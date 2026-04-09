"""Thesis generation pipeline orchestrator.

Owns the end-to-end flow of: fetching articles, building the vectorstore,
retrieving relevant docs, and generating the thesis. Each step persists
its result through IJobManager so progress is visible in real time.
"""

import logging

from api.schemas import JobStatus
from core.interfaces.job_manager import IJobManager
from dependency_injection.container import ServiceContainer

logger = logging.getLogger(__name__)


class ThesisPipelineService:
    """Orchestrates the four-step thesis generation pipeline."""

    def __init__(self, container: ServiceContainer, jm: IJobManager):
        self._container = container
        self._jm = jm

    def run(self, job_id: str) -> None:
        """Execute the full pipeline for a job. Catches and persists errors."""
        job = self._jm.get_job(job_id)
        if not job:
            return

        try:
            articles = self._fetch_articles(job_id)
            if not articles:
                return

            documents = self._build_vectorstore(job_id, articles)
            docs = self._retrieve_docs(job_id, job.query)
            if not docs:
                return

            self._generate_thesis(job_id, job.query, docs)
            self._jm.update_status(
                job_id, JobStatus.COMPLETED, "Thesis generated successfully"
            )
            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.exception(f"Job {job_id} failed")
            self._jm.update_job(job_id, error=str(e))
            self._jm.update_status(job_id, JobStatus.FAILED, f"Error: {e}")

    # --- pipeline steps ---

    def _fetch_articles(self, job_id: str):
        self._jm.update_status(
            job_id, JobStatus.FETCHING_ARTICLES, "Fetching fintech news..."
        )
        ingestion = self._container.get_ingestion_service()
        articles = ingestion.fetch_articles(query="fintech", limit=5)
        if not articles:
            self._jm.update_status(job_id, JobStatus.FAILED, "No articles found")
            self._jm.update_job(job_id, error="No articles found from RSS feeds")
            return None
        self._jm.update_job(job_id, articles=articles)
        return articles

    def _build_vectorstore(self, job_id: str, articles):
        self._jm.update_status(
            job_id, JobStatus.BUILDING_VECTORSTORE, "Building FAISS vectorstore..."
        )
        ingestion = self._container.get_ingestion_service()
        documents = ingestion.convert_to_documents(articles)
        retrieval = self._container.get_retrieval_service()
        retrieval.build_vectorstore(documents)
        return documents

    def _retrieve_docs(self, job_id: str, query: str):
        self._jm.update_status(
            job_id, JobStatus.RETRIEVING, "Retrieving relevant context..."
        )
        retrieval = self._container.get_retrieval_service()
        docs = retrieval.retrieve(query, k=5)
        if not docs:
            self._jm.update_status(
                job_id, JobStatus.FAILED, "No relevant documents found"
            )
            self._jm.update_job(
                job_id, error="No relevant documents found for query"
            )
            return None
        self._jm.update_job(job_id, retrieved_docs=docs)
        return docs

    def _generate_thesis(self, job_id: str, query: str, docs):
        self._jm.update_status(
            job_id, JobStatus.GENERATING, "Generating market thesis..."
        )
        thesis_service = self._container.get_thesis_service()
        thesis = thesis_service.generate_thesis(query, docs)
        self._jm.update_job(job_id, thesis=thesis)
        return thesis
