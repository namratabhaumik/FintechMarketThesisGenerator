"""FAISS vector store implementation."""

import logging
from typing import Any, List

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore

from config.settings import VectorStoreConfig
from core.interfaces.embeddings import IEmbeddingModel
from core.interfaces.vectorstore import IVectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(IVectorStore):
    """FAISS vector store implementation."""

    def __init__(self, config: VectorStoreConfig, embedding_model: IEmbeddingModel):
        """Initialize with configuration and embedding model.

        Args:
            config: Vector store configuration.
            embedding_model: Embedding model implementation (dependency injection).
        """
        self._config = config
        self._embedding_model = embedding_model

    def build(self, documents: List[Document]) -> VectorStore:
        """Build FAISS vectorstore from documents.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            Built FAISS vectorstore.

        Raises:
            ValueError: If no valid documents provided.
        """
        if not documents:
            raise ValueError("No documents provided for vectorstore")

        logger.info(f"Building FAISS vectorstore from {len(documents)} documents")

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks")

        # Build vectorstore with embeddings
        embeddings = self._embedding_model.get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        logger.info("FAISS vectorstore built successfully")
        return vectorstore

    def as_retriever(self, vectorstore: VectorStore, k: int) -> Any:
        """Get retriever from vectorstore.

        Args:
            vectorstore: Built vectorstore instance.
            k: Number of documents to retrieve.

        Returns:
            Retriever instance.
        """
        return vectorstore.as_retriever(search_kwargs={"k": k})
