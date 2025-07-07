# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from rag_agent.agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from contextlib import asynccontextmanager

# Constantes de sesión
APP_NAME   = "juris-rag-ai"
USER_ID    = "default-user"
SESSION_ID = "default-session"

# Inicia runner y sesión
session_service = InMemorySessionService()
runner = Runner(
    app_name=APP_NAME,
    agent=root_agent,
    session_service=session_service,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    yield

app = FastAPI(
    title="Jurisprudence RAG AI",
    version="1.0.0",
    lifespan=lifespan,
)

class Query(BaseModel):
    prompt: str

class Response(BaseModel):
    answer: str
    metadata: List[Dict]  # <-- aquí incluyes los metadatos por separado

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/predict", response_model=Response)
async def predict(q: Query):
    if not q.prompt.strip():
        raise HTTPException(400, "El campo 'prompt' no puede estar vacío")

    # 1) Llamo al agente
    user_msg = types.Content(role="user", parts=[types.Part(text=q.prompt)])
    events = list(runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_msg
    ))

    # 2) Extraer la respuesta final
    final_text = None
    for ev in events:
        if ev.is_final_response() and ev.content and ev.content.parts:
            final_text = ev.content.parts[0].text

    # 3) Extraer metadata de los function_response de bq_metadata
    metadata = []
    for ev in events:
        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                fr = getattr(part, "function_response", None)
                if fr and fr.name == "bq_metadata":
                    metadata.append(fr.response)

    if final_text is None:
        raise HTTPException(500, "No se obtuvo respuesta del agente")

    return Response(answer=final_text, metadata=metadata)
