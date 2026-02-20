"""Dependency Injection Container for wiring dependencies."""

import logging
from typing import Dict, Optional, Type

from config.settings import AppConfig
from core.implementations.article_sources.rss_source import RSSArticleSource
from core.implementations.embeddings.huggingface_embeddings import (
    HuggingFaceEmbeddingModel,
)
from core.implementations.llm.gemini_llm import GeminiLanguageModel
from core.implementations.scrapers.beautifulsoup_scraper import BeautifulSoupScraper
from core.implementations.vectorstores.faiss_store import FAISSVectorStore
from core.interfaces.article_source import IArticleSource
from core.interfaces.embeddings import IEmbeddingModel
from core.interfaces.llm import ILanguageModel
from core.interfaces.scraper import IWebScraper
from core.interfaces.vectorstore import IVectorStore
from core.services.ingestion_service import ArticleIngestionService
from core.services.retrieval_service import DocumentRetrievalService
from core.services.thesis_generator_service import ThesisGeneratorService

# To add a new LLM provider, see README.md
LLM_PROVIDER_REGISTRY: Dict[str, Type[ILanguageModel]] = {
    "gemini": GeminiLanguageModel,
}

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Dependency Injection Container.

    Follows Dependency Inversion Principle: All dependencies flow through abstractions.
    Implements lazy loading of singletons for efficiency.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize container with configuration.

        Args:
            config: Application configuration. If None, loads from environment.
        """
        self._config = config or AppConfig.from_env()

        # Lazy-loaded interface implementations (singletons)
        self._scraper: Optional[IWebScraper] = None
        self._article_source: Optional[IArticleSource] = None
        self._embedding_model: Optional[IEmbeddingModel] = None
        self._vectorstore: Optional[IVectorStore] = None
        self._llm: Optional[ILanguageModel] = None

        # Services
        self._ingestion_service: Optional[ArticleIngestionService] = None
        self._retrieval_service: Optional[DocumentRetrievalService] = None
        self._thesis_service: Optional[ThesisGeneratorService] = None

        logger.info("ServiceContainer initialized")

    # === Factory Methods for Core Interfaces ===

    def get_scraper(self) -> IWebScraper:
        """Get or create web scraper implementation.

        Returns:
            IWebScraper implementation (BeautifulSoupScraper).
        """
        if not self._scraper:
            logger.info("Creating BeautifulSoupScraper")
            self._scraper = BeautifulSoupScraper(self._config.scraper)
        return self._scraper

    def get_article_source(self) -> IArticleSource:
        """Get or create article source implementation.

        Returns:
            IArticleSource implementation (RSSArticleSource).
        """
        if not self._article_source:
            logger.info("Creating RSSArticleSource")
            scraper = self.get_scraper()
            self._article_source = RSSArticleSource(
                feeds=self._config.rss_feeds,
                scraper=scraper
            )
        return self._article_source

    def get_embedding_model(self) -> IEmbeddingModel:
        """Get or create embedding model implementation.

        Returns:
            IEmbeddingModel implementation based on configuration.

        Raises:
            ValueError: If configured embedding provider is unknown.
        """
        if not self._embedding_model:
            logger.info(f"Creating {self._config.embedding.provider} embedding model")

            if self._config.embedding.provider == "huggingface":
                self._embedding_model = HuggingFaceEmbeddingModel(
                    self._config.embedding
                )
            else:
                raise ValueError(
                    f"Unknown embedding provider: {self._config.embedding.provider}"
                )

        return self._embedding_model

    def get_vectorstore(self) -> IVectorStore:
        """Get or create vectorstore implementation.

        Returns:
            IVectorStore implementation based on configuration.

        Raises:
            ValueError: If configured vectorstore provider is unknown.
        """
        if not self._vectorstore:
            logger.info(f"Creating {self._config.vectorstore.provider} vectorstore")

            if self._config.vectorstore.provider == "faiss":
                embedding_model = self.get_embedding_model()
                self._vectorstore = FAISSVectorStore(
                    self._config.vectorstore,
                    embedding_model
                )
            else:
                raise ValueError(
                    f"Unknown vectorstore provider: {self._config.vectorstore.provider}"
                )

        return self._vectorstore

    def get_llm(self) -> ILanguageModel:
        """Get or create LLM implementation.

        Selects the concrete strategy from LLM_PROVIDER_REGISTRY based on
        the configured provider. To add a new LLM, register it in the registry.

        Returns:
            ILanguageModel implementation based on configuration.

        Raises:
            ValueError: If configured LLM provider is not in the registry.
        """
        if not self._llm:
            provider = self._config.llm.provider
            llm_class = LLM_PROVIDER_REGISTRY.get(provider)

            if not llm_class:
                raise ValueError(
                    f"Unknown LLM provider: '{provider}'. "
                    f"Supported: {list(LLM_PROVIDER_REGISTRY.keys())}"
                )

            logger.info(f"Creating {provider} LLM ({llm_class.__name__})")
            self._llm = llm_class(self._config.llm)

        return self._llm

    # === Service Factories ===

    def get_ingestion_service(self) -> ArticleIngestionService:
        """Get or create article ingestion service.

        Returns:
            ArticleIngestionService instance.
        """
        if not self._ingestion_service:
            logger.info("Creating ArticleIngestionService")
            article_source = self.get_article_source()
            self._ingestion_service = ArticleIngestionService(article_source)
        return self._ingestion_service

    def get_retrieval_service(self) -> DocumentRetrievalService:
        """Get or create document retrieval service.

        Returns:
            DocumentRetrievalService instance.
        """
        if not self._retrieval_service:
            logger.info("Creating DocumentRetrievalService")
            vectorstore = self.get_vectorstore()
            self._retrieval_service = DocumentRetrievalService(vectorstore)
        return self._retrieval_service

    def get_thesis_service(self) -> ThesisGeneratorService:
        """Get or create thesis generator service.

        Returns:
            ThesisGeneratorService instance.
        """
        if not self._thesis_service:
            logger.info("Creating ThesisGeneratorService")
            llm = self.get_llm()
            self._thesis_service = ThesisGeneratorService(llm)
        return self._thesis_service
