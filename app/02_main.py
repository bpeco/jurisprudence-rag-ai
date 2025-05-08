import streamlit as st
from get_response import get_answer


st.title("🔍 AI Derecho Comercial Argentino - Fallos")
pregunta = st.text_input("Tu pregunta:")
if st.button("Consultar") and pregunta:
    with st.spinner("Pensando…"):
        respuesta, metadatas = get_answer(question=pregunta, k=3)
    st.markdown("**Respuesta:**")
    st.write(respuesta)
    if metadatas:
        st.markdown("**Fallos utilizados (metadata):**")
        for md in metadatas:
            with st.expander(f"ID: {md['Expediente']}"):
                st.write(f"- **Tribunal:** {md['Tribunal']}")
                st.write(f"- **Sala:** {md['Sala']}")
                st.write(f"- **Expediente:** {md['Expediente']}")
                st.write(f"- **Caratula:** {md['Caratula']}")
                st.write(f"- **Título:** {md['title']}")
                st.write(f"- **Fecha Sentencia:** {md['FechaSentencia']}")
