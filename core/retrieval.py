# core/retrieval.py
import logging
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def build_vectorstore(articles: list):
    """Convert article dicts into FAISS vector store."""
    if not articles:
        raise ValueError("No articles provided for vectorstore.")

    docs = [
        Document(
            page_content=f"{a['title']}\n\n{a['text']}",
            metadata={"source": a["source"], "title": a["title"]}
        )
        for a in articles
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info("FAISS vectorstore built successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to build FAISS vectorstore: {e}")
        raise
