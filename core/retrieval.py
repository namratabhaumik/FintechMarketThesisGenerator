# core/retrieval.py
import logging
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

logging.getLogger("faiss.loader").setLevel(logging.ERROR)


def build_vectorstore(articles: list):
    """Convert article dicts into FAISS vector store."""
    if not articles:
        raise ValueError("No articles provided for vectorstore.")

    docs = []
    for a in articles:
        title = a.get("title", "Untitled Article")
        text = a.get("text") or a.get("content") or ""
        source = a.get("source", "unknown")

        if not text.strip():
            continue

        docs.append(
            Document(
                page_content=f"{title}\n\n{text}",
                metadata={"source": source, "title": title}
            )
        )

    if not docs:
        raise ValueError("No valid documents found to build vectorstore.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info("✅ FAISS vectorstore built successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"❌ Failed to build FAISS vectorstore: {e}")
        raise
