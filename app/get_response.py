# app/pipeline/response.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import pickle

# Prompts definition
COMBINE_PROMPT = PromptTemplate.from_template(
    """
Sos un asistente jurídico.
Usá exclusivamente el contenido de los documentos proporcionados para responder la consulta.
No inventes ni infieras información que no esté presente en los textos.

\n\nDocumentos:\n
{context}

\n\nPregunta:\n
{question}

\n\nRespuesta:\n
"""
)

RELEVANCE_PROMPT = PromptTemplate.from_template(
    """
¿Aporta este texto información relevante para responder la pregunta siguiente?
\n\nPregunta:\n
{question}

\n\nTexto:\n
{content}

\n\nResponde únicamente "Sí" o "No".
"""
)


def build_retriever(k: int = 3) -> MultiVectorRetriever:
    """
    Load embeddings, rebuild the vectorstore, and return a MultiVectorRetriever.

    Args:
        k: Number of nearest neighbors to retrieve.
    """
    # set up the embedding model and vector store
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory="./multivector_chroma_db_001"
    )

    with open("parent_documents.pkl", "rb") as f:
        parent_documents = pickle.load(f)

    store = InMemoryStore()
    store.mset([(d.metadata["id"], d) for d in parent_documents])

    # return the multi-vector retriever instance
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="parent_id",
        search_kwargs={"k": k}
    )


def filter_relevant_docs(
    docs: list,
    question: str,
    llm: ChatOpenAI
) -> list:
    """
    Dynamically filters documents using a relevance chain.

    Args:
        docs: List of candidate Document objects.
        question: The query text.
        llm: An instance of ChatOpenAI for making relevance calls.

    Returns:
        A list of Document instances deemed relevant by the model.
    """
    # create a small LLMChain to evaluate relevance
    relevance_chain = LLMChain(llm=llm, prompt=RELEVANCE_PROMPT)
    filtered = []
    for doc in docs:
        verdict = relevance_chain.run(
            question=question,
            content=doc.page_content
        )
        if verdict.strip().lower().startswith("sí"):
            filtered.append(doc)
    return filtered


def get_answer(
    question: str,
    k: int = 3
) -> tuple[str, list[dict]]:
    """
    Executes the RAG pipeline with prior document filtering.

    Args:
        question: The query text.
        k: Number of documents to retrieve.

    Returns:
        A tuple containing the model response and a list of metadata dicts for the used documents.
    """
    retriever = build_retriever(k)
    llm = ChatOpenAI(model="o4-mini", temperature=1.0)

    # set up the chain that stitches documents into an answer
    combine_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=COMBINE_PROMPT
    )

    # retrieve, filter documents and final answer generation
    retrieved = retriever.invoke(question)
    relevant = filter_relevant_docs(retrieved, question, llm)

    response = combine_chain.invoke({
        "question": question,
        "context": relevant
    })

    # extract metadata for traceability
    metadatas = [doc.metadata for doc in relevant]
    return response, metadatas
