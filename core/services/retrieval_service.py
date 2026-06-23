"""Document retrieval service."""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from config.settings import RetrievalConfig
from core.interfaces.vectorstore import IVectorStore

logger = logging.getLogger(__name__)


class DocumentRetrievalService:
    """Reads the persistent corpus: opens the store and runs MMR retrieval.

    On the first retrieve it lazily opens the store, then returns the
    most relevant, diverse chunks for a query. Depends on the IVectorStore
    abstraction.
    """

    def __init__(self, vectorstore: IVectorStore, config: RetrievalConfig):
        """Initialize with vectorstore implementation and retrieval config.

        Args:
            vectorstore: Injected vectorstore implementation.
            config: MMR retrieval hyperparameters (k / fetch_k / lambda_mult).
        """
        self._vectorstore_impl = vectorstore
        self._config = config
        self._vectorstore_instance: Optional[VectorStore] = None

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

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve relevant documents for query via MMR.

        Args:
            query: Search query.
            k: Number of documents to return. Defaults to the configured k; pass
                an explicit value only to override per call.

        Returns:
            List of retrieved Document objects.

        Raises:
            RuntimeError: If vectorstore not built.
        """
        if self._vectorstore_instance is None:
            # Open the existing persistent corpus for reading
            # Ingestion happens offline in Silver; a thesis request only
            # reads what the medallion already built.
            logger.info("Opening existing persistent vector store for retrieval")
            self._vectorstore_instance = self._vectorstore_impl.open()

        # Use the configured k unless the caller overrides it. fetch_k must be
        # >= k for MMR (it is the candidate pool we select k from), so widen it
        # if an override pushes k past the configured pool.
        effective_k = k if k is not None else self._config.k
        fetch_k = max(self._config.fetch_k, effective_k)

        logger.info(f"Retrieving {effective_k} documents (MMR) for query: {query}")
        try:
            docs = self._vectorstore_impl.retrieve(
                self._vectorstore_instance,
                query,
                k=effective_k,
                fetch_k=fetch_k,
                lambda_mult=self._config.lambda_mult,
                window_days=self._config.window_days,
            )
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            raise
        logger.info(f"Retrieved {len(docs)} documents")
        return docs

    def is_built(self) -> bool:
        """Check if vectorstore is built.

        Returns:
            True if vectorstore is initialized.
        """
        return self._vectorstore_instance is not None
