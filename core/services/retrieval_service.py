"""Document retrieval service."""

import logging
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import RetrievalConfig
from core.interfaces.vectorstore import IVectorStore

logger = logging.getLogger(__name__)


class DocumentRetrievalService:
    """Reads the persistent corpus: runs MMR retrieval over it.

    Stateless: the vector store implementation queries the persistent corpus
    directly, so there is no handle to open or build here. Ingestion happens
    offline in Silver; a thesis request only reads what the medallion already
    built. Depends on the IVectorStore abstraction.
    """

    def __init__(self, vectorstore: IVectorStore, config: RetrievalConfig):
        """Initialize with vectorstore implementation and retrieval config.

        Args:
            vectorstore: Injected vectorstore implementation.
            config: MMR retrieval hyperparameters (k / fetch_k / lambda_mult).
        """
        self._vectorstore_impl = vectorstore
        self._config = config

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Document]:
        """Retrieve relevant documents for query via MMR.

        Args:
            query: Search query.
            k: Number of documents to return. Defaults to the configured k; pass
                an explicit value only to override per call.
            query_embedding: Precomputed vector for `query`. Pass it when the
                caller already embedded the query (e.g. to reuse it for episodic
                recall) so retrieval does not embed a second time; None embeds
                the query inside the vector store.

        Returns:
            List of retrieved Document objects.
        """
        # Use the configured k unless the caller overrides it. fetch_k must be
        # >= k for MMR (it is the candidate pool we select k from), so widen it
        # if an override pushes k past the configured pool.
        effective_k = k if k is not None else self._config.k
        fetch_k = max(self._config.fetch_k, effective_k)

        logger.info(f"Retrieving {effective_k} documents (MMR) for query: {query}")
        try:
            docs = self._vectorstore_impl.retrieve(
                query,
                k=effective_k,
                fetch_k=fetch_k,
                lambda_mult=self._config.lambda_mult,
                window_days=self._config.window_days,
                query_embedding=query_embedding,
                min_similarity=self._config.min_similarity,
            )
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            raise
        logger.info(f"Retrieved {len(docs)} documents")
        return docs
