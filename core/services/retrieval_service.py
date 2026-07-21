"""Document retrieval service."""

import logging
from typing import List, Optional

import numpy as np
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document

from config.settings import RetrievalConfig
from core.interfaces.vectorstore import IVectorStore
from core.utils.date_intent import parse_date_intent

logger = logging.getLogger(__name__)


class DocumentRetrievalService:
    """Reads the persistent corpus in two stages: a wide analytics pool and,
    separately, the diverse subset for the LLM.

    `retrieve()` returns up to `max_articles` distinct articles (deduped by URL)
    so tag strengths, scoring, confidence and the sources list reflect the full
    weight of the market. `select_diverse()` then MMR-narrows that pool to the
    handful the summarizer actually reads, keeping token cost flat.

    Stateless: the vector store implementation queries the persistent corpus
    directly, so there is no handle to open or build here. Ingestion happens
    offline in Silver; a thesis request only reads what the medallion already
    built. Depends on the IVectorStore abstraction.
    """

    def __init__(self, vectorstore: IVectorStore, config: RetrievalConfig):
        """Initialize with vectorstore implementation and retrieval config.

        Args:
            vectorstore: Injected vectorstore implementation.
            config: retrieval hyperparameters (k / fetch_k / max_articles /
                lambda_mult / window_days / min_similarity).
        """
        self._vectorstore_impl = vectorstore
        self._config = config

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Document]:
        """Retrieve the wide analytics pool: distinct articles by relevance.

        Pulls the raw chunk pool, applies the similarity floor, dedupes by URL
        and caps to `max_articles` (all in the vector store). No MMR here - call
        `select_diverse` on the result to pick the LLM's docs.

        Args:
            query: Search query.
            query_embedding: Precomputed vector for `query`. Pass it when the
                caller already embedded the query (e.g. to reuse it for episodic
                recall) so retrieval does not embed a second time; None embeds
                the query inside the vector store.

        Returns:
            Up to `max_articles` distinct-article Documents, relevance-ordered,
            each carrying its chunk `embedding` in metadata for a later MMR pass.
        """
        # An explicit date range named in the query (e.g. "since March 2024")
        # replaces the default trailing window rather than narrowing it.
        date_intent = parse_date_intent(query)
        date_from, date_to = date_intent if date_intent else (None, None)
        window_days = None if date_intent else self._config.window_days

        logger.info(f"Retrieving up to {self._config.max_articles} articles for query: {query}")
        try:
            docs = self._vectorstore_impl.retrieve(
                query,
                fetch_k=self._config.fetch_k,
                max_articles=self._config.max_articles,
                window_days=window_days,
                query_embedding=query_embedding,
                min_similarity=self._config.min_similarity,
                date_from=date_from,
                date_to=date_to,
            )
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            raise
        logger.info(f"Retrieved {len(docs)} distinct articles")
        return docs

    def select_diverse(
        self,
        docs: List[Document],
        query_embedding: Optional[List[float]] = None,
        k: Optional[int] = None,
    ) -> List[Document]:
        """MMR-select the `k` most diverse articles for the LLM.

        Runs Maximal Marginal Relevance over the wide pool from `retrieve()`,
        using each doc's carried chunk `embedding`, so the summarizer gets
        relevant-but-varied evidence rather than near-duplicates. Returns the
        docs in MMR order; the "embedding" metadata key is dropped from the
        returned copies (it is a transient retrieval artifact, not content).

        Args:
            docs: The wide article pool (each with an "embedding" in metadata).
            query_embedding: The query vector MMR ranks relevance against.
            k: How many to select; defaults to the configured k.

        Returns:
            The selected Documents (<= k), embedding stripped.
        """
        effective_k = k if k is not None else self._config.k
        if not docs:
            return []

        candidate_embeddings = [doc.metadata.get("embedding") for doc in docs]
        # MMR needs both the query vector and every candidate's chunk vector. If
        # the query embedding failed upstream, or a doc has no carried embedding
        # (e.g. a rehydrated stored doc), fall back to the leading
        # relevance-ordered docs - which are already the best matches.
        if query_embedding is None or any(e is None for e in candidate_embeddings):
            logger.warning("select_diverse: no query/candidate vectors; falling back to top-k by relevance")
            return [_without_embedding(d) for d in docs[:effective_k]]

        selected = maximal_marginal_relevance(
            np.array(query_embedding, dtype=np.float32),
            candidate_embeddings,
            k=min(effective_k, len(docs)),
            lambda_mult=self._config.lambda_mult,
        )
        return [_without_embedding(docs[i]) for i in selected]


def _without_embedding(doc: Document) -> Document:
    """A copy of `doc` with the transient "embedding" metadata key removed."""
    metadata = {k: v for k, v in (doc.metadata or {}).items() if k != "embedding"}
    return Document(page_content=doc.page_content, metadata=metadata)
