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

TABLE = "documents"
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
        self._config = config
        self._embeddings = embedding_model.get_embeddings()
        self._client = client

    def build(self, documents: List[Document]) -> VectorStore:
        existing_urls = self._fetch_existing_urls()
        new_docs = [
            d for d in documents
            if d.metadata.get("url", "") not in existing_urls
        ]

        if new_docs:
            logger.info(
                f"Embedding {len(new_docs)} new articles "
                f"({len(documents) - len(new_docs)} already cached)"
            )
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )
            chunks = splitter.split_documents(new_docs)
            logger.info(f"Upserting {len(chunks)} chunks to Supabase pgvector")
            SupabaseVectorStore.from_documents(
                chunks,
                self._embeddings,
                client=self._client,
                table_name=TABLE,
                query_name=QUERY_NAME,
            )
        else:
            logger.info("All articles already cached — skipping embedding")

        return SupabaseVectorStore(
            client=self._client,
            embedding=self._embeddings,
            table_name=TABLE,
            query_name=QUERY_NAME,
        )

    def as_retriever(self, vectorstore: VectorStore, k: int) -> Any:
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def _fetch_existing_urls(self) -> Set[str]:
        try:
            resp = self._client.table(TABLE).select("metadata").execute()
            return {
                row["metadata"].get("url", "")
                for row in resp.data
                if row.get("metadata")
            }
        except Exception as e:
            logger.warning(f"Could not fetch existing URLs from Supabase: {e}")
            return set()
