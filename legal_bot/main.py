import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from agent_executor import CustomAgentExecutor

st.set_page_config(page_title="Jedi Legal RAG Bot", page_icon="⚖️")
st.title("Jedi Legal RAG Bot ⚖️")

# --- Estado de sesión ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="¡Hola! Soy Jedi, tu asistente legal. Preguntame lo que necesites sobre jurisprudencia.")
    ]
if "filtered_docs" not in st.session_state:
    st.session_state.filtered_docs = None


for msg in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
        st.markdown(msg.content)


user_input = st.chat_input("Escribí tu pregunta jurídica...")
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("Human"):
        st.markdown(user_input) #ojo con esto que me está dando duplicados

    agent_executor = CustomAgentExecutor(
        chat_history=st.session_state.chat_history,
        filtered_docs=st.session_state.filtered_docs,  
    )
    response, new_filtered_docs = agent_executor.invoke(user_input)
    st.session_state.chat_history.append(AIMessage(content=response))

    with st.chat_message("AI"):
        st.write(response)
    st.session_state.filtered_docs = new_filtered_docs 
