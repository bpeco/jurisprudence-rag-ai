# app/pipeline/response.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import pickle
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.chains.combine_documents import create_stuff_documents_chain



# 1) Definí aquí tu prompt
PROMPT = PromptTemplate.from_template(
    """
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
"""
)



def get_answer(question: str, k: int = 3) -> tuple[str, list[dict]]:
    """
    Executes the RAG query.
    - question: The text of the question.
    - k: Number of neighboring documents to retrieve.
    Returns a tuple (answer, list_of_metadata).
    """
    # 2) Cargá el embedding y el vectorstore persistido
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory="./multivector_chroma_db_001"
    )

    with open("parent_documents.pkl", "rb") as f:
        parent_documents = pickle.load(f)

    # 🔹 Reconstruís el store
    store = InMemoryStore()
    store.mset([(d.metadata["id"], d) for d in parent_documents])

    # 🔹 Reconstruís el retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="parent_id",
        search_kwargs={"k": k} # reminder: top 3 similitud para los chunks resumidos, no para los fallos completos
    )

    # 4) Armá la cadena de RetrievalQA con tu LLM y prompt
    llm = ChatOpenAI(model='o4-mini',temperature=1.0)
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)

    # Step 1: retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # Step 2: chain to analyze retrieved documents with the question
    response = combine_docs_chain.invoke({
        "question": question,
        "context": retrieved_docs
    })


    metadatas = [doc.metadata for doc in retrieved_docs]
    return response, metadatas
