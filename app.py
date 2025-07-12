import streamlit as st
import requests
import os

# ——— 1) Configuración de página ———
st.set_page_config(page_title="Chat RAG", page_icon="🤖", layout="centered")

# ——— 2) URL de tu backend ———
API_URL = "https://jurisprudence-rag-ai-303029425062.us-central1.run.app"
#"http://localhost:8000"

# ——— 3) Inicializar historial en sesión ———
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # lista de dicts: {"role","content"}

# ——— 4) Mostrar historial ———
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# ——— 5) Entrada de usuario ———
user_input = st.chat_input("Escribe tu mensaje…")
if user_input:
    # A) Guarda el mensaje del usuario
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # B) Llama a tu endpoint /predict y extrae answer + metadata
    try:
        res = requests.post(
            f"{API_URL}/predict",
            json={"prompt": user_input},
            timeout=60,
        )
        res.raise_for_status()
        data = res.json()
        answer = data.get("answer", "")
        metadata = data.get("metadata", [])
    except Exception as e:
        answer = f"⚠️ Error: {e}"
        metadata = []

    # C) Muestra la respuesta
    with st.chat_message("assistant"):
        st.write(answer)

        # D) Si hay metadata, crea un expander por documento
        if metadata:
            st.markdown("**Metadatos de los documentos encontrados:**")
            for doc in metadata:
                # Usa el nombre del PDF como título del expander
                title = doc.get("expediente_n", "Documento")
                with st.expander(title):
                    st.markdown(f"- **Tribunal:** {doc.get('tribunal','—')}")
                    st.markdown(f"- **Expediente N:** {doc.get('expediente_n','—')}")
                    st.markdown(f"- **Carátula:** {doc.get('caratula','—')}")
                    st.markdown(f"- **Fecha de Sentencia:** {doc.get('fecha_sentencia','—')}")
                    st.markdown(f"- **Sala:** {doc.get('sala','—')}")
        else:
            st.markdown("_No se encontraron metadatos para los documentos._")

    # E) Guarda la respuesta en el historial (solo el texto)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
