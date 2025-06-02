# agent_executor.py

import pickle
from typing import List, Optional, Tuple
from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, BaseMessage
from langchain.storage import InMemoryStore
from langchain_core.runnables.base import RunnableSerializable
import os
import json

# --- Setup embeddings, vectorstore and retriever ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    embedding_function=embedding,
    persist_directory="./chroma/multivector_chroma_db_001"
)
with open("./parent_documents.pkl", "rb") as f:
    parent_documents = pickle.load(f)

store = InMemoryStore()
store.mset([(d.metadata["id"], d) for d in parent_documents])

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key="parent_id",
    search_kwargs={"k": 3}
)

# --- LLMs ---
#llm_streaming = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, streaming=True)
#llm_sync = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, streaming=False)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# --- Prompts ---
combine_prompt = PromptTemplate.from_template("""
Sos un asistente jurídico.
Usá exclusivamente el contenido de los documentos proporcionados para responder la consulta.
Si encontrás documentos relacionados, proporcioná la información correspondiente de dichos documentos.
No inventes ni infieras información que no esté presente en los textos.
Si no hay jurisprudencia relevante, indicá claramente que no la encontrás.
Tampoco menciones ningún documento si no encontraste jurisprudencia relevante.

Documentos:
{context}

Pregunta:
{question}

Respuesta:
""")

relevance_prompt = PromptTemplate.from_template("""
¿Aporta este texto información relevante para responder la pregunta siguiente?
Pregunta:
{question}

Texto:
{content}

Responde únicamente “Sí” o “No”.
""")

combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=combine_prompt)
relevance_chain = LLMChain(llm=llm, prompt=relevance_prompt)

# --- Tools ---
@tool
def retriev_documents(question: str):
    """
    Recupera documentos jurídicos relevantes desde una base de datos Chroma utilizando un retriever avanzado.

    Usá esta herramienta cuando necesites buscar fallos, jurisprudencia o documentos legales que puedan contener información útil y específica para responder una consulta jurídica del usuario.

    Esta función es ideal como primer paso para obtener contexto legal fundamentado antes de generar una respuesta. 
    Devuelve únicamente los documentos que, tras un filtrado adicional por un modelo de lenguaje, se consideran relevantes para la pregunta planteada.

    Parámetro:
        question (str): Pregunta jurídica o consulta del usuario.
    """

    retrieved_documents = retriever.invoke(question)

    relevance_prompt = PromptTemplate.from_template("""
    ¿Aporta este texto información relevante para responder la pregunta siguiente?
    Pregunta:
    {question}

    Texto:
    {content}

    Responde únicamente “Sí” o “No”.""")
    relevance_chain = LLMChain(llm=llm, prompt=relevance_prompt)
    filtered_docs = []

#    print(retrieved_documents)

    for doc in retrieved_documents:
        verdict = relevance_chain.run(question=question, content=doc.page_content)#["page_content"])
        if verdict.strip().lower().startswith("sí"):
            filtered_docs.append(doc)


    response = combine_docs_chain.invoke({
        "question": question,
        "context": filtered_docs
    })

    return response

@tool
def should_refresh_context(
    question: str,
    previous_question: str,
    last_context_summary: str = ""
) -> bool:
    """
    Decide si la pregunta actual del usuario requiere una nueva búsqueda de documentos o si debe responderse usando el contexto anterior.

    Parámetros:
        question (str): Pregunta actual del usuario.
        previous_question (str): Última pregunta respondida (o "" si es la primera).
        last_context_summary (str): Resumen (opcional) del contexto/documentos anteriores.

    Retorna:
        bool: True si hay que buscar documentos nuevos, False si puede usar el contexto anterior.
    """

    system = (
        "Eres un asistente legal experto. Debes determinar si la nueva pregunta es "
        "sobre el mismo tema que la anterior (puede responderse con los documentos actuales) "
        "o si es un tema completamente diferente (requiere buscar nueva documentación). "
        "Sé conservador: si hay duda, sugerí buscar nuevos documentos. "
        "No expliques nada, responde solo True o False."
    )
    prompt = PromptTemplate.from_template(
        "{system}\n\nPregunta anterior: {previous_question}\n"
        "Nueva pregunta: {question}\n"
        "Resumen del contexto anterior: {last_context_summary}\n"
        "\n¿Hay que buscar documentos nuevos? Responde solo True o False."
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run(
        system=system,
        question=question,
        previous_question=previous_question,
        last_context_summary=last_context_summary
    )
    return "true" in resp.lower()


@tool
def final_answer(answer: str, tools_used: list = []) -> dict:
    """
    Utiliza esta herramienta para proveer una respuesta final al usuario.
    """
    return {"answer": answer, "tools_used": tools_used}

tools = [retriev_documents, final_answer, should_refresh_context]

name2tool = {tool.name: tool.func for tool in tools}

# --- Agent Prompt ---
agent_system_prompt = (
    "Eres un asistente legal experto en análisis de fallos judiciales y documentos jurídicos. "
    "Antes de responder, SIEMPRE evalúa si la nueva pregunta del usuario trata sobre el mismo tema que la anterior o si cambió de asunto. "
    "Usa la herramienta 'should_refresh_context' para decidir esto. "
    "Si la respuesta es True, llama a 'retriev_documents' para obtener nuevos documentos relevantes. "
    "Si la respuesta es False, responde usando el contexto/documentos anteriores. "
    "Nunca inventes información y siempre respondé con evidencia."
)


prompt = ChatPromptTemplate.from_messages([
    ("system", agent_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm_agent = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

agent: RunnableSerializable = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "agent_scratchpad": lambda x: x.get("agent_scratchpad", []),
        "context_docs": lambda x: x.get("filtered_docs", None),  # Pasamos el contexto actual
    }
    | prompt
    | llm_agent.bind_tools(tools, tool_choice="any")
)

# --- Custom Agent Executor ---
class CustomAgentExecutor:
    def __init__(
        self,
        chat_history: List[BaseMessage],
        filtered_docs: Optional[List[Document]],
        max_iterations: int = 3
    ):
        self.chat_history = chat_history
        self.filtered_docs = filtered_docs
        self.max_iterations = max_iterations
        self.agent = agent

    def invoke(self, input: str, progress_callback=True) -> Tuple[str, Optional[List[Document]]]:
        """
        Ejecuta el agente iterativamente y mantiene/actualiza el contexto documental.
        Devuelve la respuesta final (str) y los documentos contextuales actualizados (list[Document] o None)
        """
        count = 0
        agent_scratchpad = []
        curr_filtered_docs = self.filtered_docs
        end_loop = False

        while count < self.max_iterations and not end_loop:

            print(f'[DEBUG] input >> {input}')
            print(f'[DEBUG] chat_history >> {self.chat_history}')
            print(f'[DEBUG] agent_scratchpad >> {agent_scratchpad}')
            tool_call = self.agent.invoke({
                "input": input,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            })
            #print(f'[DEBUG] tool_call >>> {tool_call}')
            agent_scratchpad.append(tool_call)
            #print(f'[DEBUG] agent_scratchpad >>> {agent_scratchpad}')
            
            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_call_id = tool_call.tool_calls[0]["id"]


            if tool_name == "retriev_documents" and "question" in tool_args:
                    tool_args["question"] = input

            print(f'[DEBUG] tool_name >>> {tool_name}')
            print(f'[DEBUG] tool_args >>> {tool_args}')

            tool_out = name2tool[tool_name](**tool_args)
            #print(f'[DEBUG] tool_out >>> {tool_out}')

            tool_exec = ToolMessage(
                content=f"{tool_out}",
                tool_call_id=tool_call_id
            )
            #print(f'[DEBUG] tool_exec >>> {tool_exec}')
            agent_scratchpad.append(tool_exec)
            #print(f'[DEBUG] agent_scratchpad >>> {agent_scratchpad}')

            print(f"{count}: {tool_name}({tool_args})")
            count += 1

            if tool_name == "final_answer":
                end_loop = True

        final_answer = tool_out["answer"]
        self.chat_history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer)
        ])

    
        return final_answer, curr_filtered_docs

