"""Document retrieval service."""

import logging
from typing import List

from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore

from core.interfaces.vectorstore import IVectorStore

logger = logging.getLogger(__name__)


class DocumentRetrievalService:
    """Service for document retrieval.

    Single Responsibility: Vector store management and retrieval.
    Implements Dependency Inversion: Depends on IVectorStore abstraction.
    """

    def __init__(self, vectorstore: IVectorStore):
        """Initialize with vectorstore implementation.

        Args:
            vectorstore: Injected vectorstore implementation.
        """
        self._vectorstore_impl = vectorstore
        self._vectorstore_instance: VectorStore = None

    def build_vectorstore(self, documents: List[Document]) -> None:
        """Build vectorstore from documents.

        Args:
            documents: List of LangChain Document objects.

        Raises:
            ValueError: If no documents provided.
        """
        logger.info(f"Building vectorstore from {len(documents)} documents")
        self._vectorstore_instance = self._vectorstore_impl.build(documents)
        logger.info("Vectorstore built and cached")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for query.

        Args:
            query: Search query.
            k: Number of documents to retrieve.

        Returns:
            List of retrieved Document objects.

        Raises:
            RuntimeError: If vectorstore not built.
        """
        if not self._vectorstore_instance:
            raise RuntimeError(
                "Vectorstore not built. Call build_vectorstore() first."
            )

        logger.info(f"Retrieving {k} documents for query: {query}")
        retriever = self._vectorstore_impl.as_retriever(
            self._vectorstore_instance,
            k=k
        )
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents")
        return docs

    def is_built(self) -> bool:
        """Check if vectorstore is built.

        Returns:
            True if vectorstore is initialized.
        """
        return self._vectorstore_instance is not None
