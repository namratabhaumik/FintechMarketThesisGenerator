"""Supabase pgvector vectorstore implementation with URL-based deduplication."""

import logging
from typing import Any, List, Set

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import Client

from config.settings import VectorStoreConfig
from core.interfaces.embeddings import IEmbeddingModel
from core.interfaces.vectorstore import IVectorStore

logger = logging.getLogger(__name__)

# Postgres table that holds the embedded chunks.
TABLE = "documents"
# Name of the Supabase SQL function used for similarity search over that table.
QUERY_NAME = "match_documents"


class SupabaseVectorStoreImpl(IVectorStore):
    """pgvector-backed vectorstore. Embeddings persist across runs; only new
    article URLs are embedded on each call to build()."""

    def __init__(
        self,
        config: VectorStoreConfig,
        embedding_model: IEmbeddingModel,
        client: Client,
    ):
        # Chunk-size / overlap settings for splitting long articles.
        self._config = config
        # The model that turns text into vectors (embeddings).
        self._embeddings = embedding_model.get_embeddings()
        # Supabase client used to read existing URLs and upsert new chunks.
        self._client = client

    def build(self, documents: List[Document]) -> VectorStore:
        """Embed only the not-yet-seen documents, then return a live store.

        read URLs already in the DB --> keep only documents whose URL is
        new --> split them into chunks --> embed + upsert --> return a retriever-
        ready vector store.
        """
        # URLs already embedded in a previous run (our dedup key).
        existing = self._fetch_existing_urls()
        # new_docs: documents whose URL is NOT already in the store. Anything
        # already embedded is filtered out here so we never re-embed it.
        new_docs = [
            d for d in documents
            if d.metadata.get("url", "") not in existing
        ]

        # Something new to embed --> split, embed, and upsert it.
        if new_docs:
            logger.info(
                f"Embedding {len(new_docs)} new articles "
                f"({len(documents) - len(new_docs)} already cached)"
            )
            # Break each long article into overlapping chunks so each piece fits
            # the embedding model and retrieval can pinpoint the relevant part.
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )
            # chunks: the per-piece documents actually written to pgvector.
            chunks = splitter.split_documents(new_docs)
            logger.info(f"Upserting {len(chunks)} chunks to Supabase pgvector")
            # Embed every chunk and insert the vectors into the documents table.
            SupabaseVectorStore.from_documents(
                chunks,
                self._embeddings,
                client=self._client,
                table_name=TABLE,
                query_name=QUERY_NAME,
            )
        else:
            # Every URL was already embedded --> nothing to write this run.
            logger.info("All articles already cached — skipping embedding")

        # Return a handle to the full store (old + new chunks) for searching.
        return SupabaseVectorStore(
            client=self._client,
            embedding=self._embeddings,
            table_name=TABLE,
            query_name=QUERY_NAME,
        )

    def as_retriever(
        self, vectorstore: VectorStore, k: int, fetch_k: int, lambda_mult: float
    ) -> Any:
        # MMR retriever: fetch `fetch_k` candidates by similarity, then pick `k`
        # that are relevant (lambda_mult tunes the relevance/diversity balance).
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )

    def _fetch_existing_urls(self) -> Set[str]:
        """Return the article URLs already embedded, for internal build dedup.

        Raises on read failure instead of returning an empty set: a silent empty
        set would make build() treat every document as new and re-insert chunks,
        duplicating embeddings. 
        """
        try:
            # Pull just the metadata column from every stored chunk.
            resp = self._client.table(TABLE).select("metadata").execute()
        except Exception as e:
            # Read failed --> raise (do NOT swallow), because an empty set here
            # would make build() re-embed everything and duplicate chunks.
            raise RuntimeError(f"Failed to read existing URLs from Supabase: {e}") from e
        # Build a set of the URLs found in metadata. A set gives fast "is this
        # URL already stored?" lookups and collapses the many chunks per article
        # down to one entry per URL.
        return {
            row["metadata"].get("url", "")
            for row in resp.data
            if row.get("metadata")
        }
