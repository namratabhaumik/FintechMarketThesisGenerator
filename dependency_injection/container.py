"""Dependency Injection Container for wiring dependencies."""

import logging
from typing import Dict, Optional, Type

from config.settings import AppConfig
from core.implementations.article_sources.rss_source import RSSArticleSource
from core.implementations.embeddings.huggingface_embeddings import (
    HuggingFaceEmbeddingModel,
)
from core.implementations.keyword_scoring_strategy import KeywordCountScoringStrategy
from core.implementations.llm.ai_gateway import AIGateway
from core.implementations.llm.cache_manager import CacheManager
from core.implementations.llm.cost_tracker import CostTracker
from core.implementations.llm.gemini_llm import GeminiLanguageModel
from core.implementations.llm.local_summarizer import LocalSummarizerModel
from core.implementations.llm.llm_wrapper import LLMWrapper
from core.implementations.scrapers.beautifulsoup_scraper import BeautifulSoupScraper
from core.implementations.vectorstores.faiss_store import FAISSVectorStore
from core.interfaces.article_source import IArticleSource
from core.interfaces.embeddings import IEmbeddingModel
from core.interfaces.llm import ILanguageModel
from core.interfaces.scraper import IWebScraper
from core.interfaces.scoring_strategy import IScoringStrategy
from core.interfaces.thesis_structurer import IThesisStructurer
from core.interfaces.vectorstore import IVectorStore
from core.services.approval_service import ApprovalService
from core.services.ingestion_service import ArticleIngestionService
from core.services.opportunity_scoring_service import OpportunityScoringService
from core.services.retrieval_service import DocumentRetrievalService
from core.services.thesis_generator_service import ThesisGeneratorService
from core.services.thesis_structuring_service import ThesisStructuringService

# To add a new LLM provider, see README.md
LLM_PROVIDER_REGISTRY: Dict[str, Type[ILanguageModel]] = {
    "gemini": GeminiLanguageModel,
    "local": LocalSummarizerModel,
}

# To add a new embedding provider, see README.md
EMBEDDING_PROVIDER_REGISTRY: Dict[str, Type[IEmbeddingModel]] = {
    "huggingface": HuggingFaceEmbeddingModel,
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
        self._scoring_strategy: Optional[IScoringStrategy] = None
        self._thesis_structurer: Optional[IThesisStructurer] = None

        # AI Gateway components (singletons)
        self._cache_manager: Optional[CacheManager] = None
        self._cost_tracker: Optional[CostTracker] = None

        # Services
        self._ingestion_service: Optional[ArticleIngestionService] = None
        self._retrieval_service: Optional[DocumentRetrievalService] = None
        self._approval_service: Optional[ApprovalService] = None
        self._opportunity_scoring_service: Optional[OpportunityScoringService] = None
        self._thesis_service: Optional[ThesisGeneratorService] = None

        # Agents
        self._refinement_graph: Optional[object] = None

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
            ValueError: If configured embedding provider is not in the registry.
        """
        if not self._embedding_model:
            provider = self._config.embedding.provider
            embedding_class = EMBEDDING_PROVIDER_REGISTRY.get(provider)

            if not embedding_class:
                raise ValueError(
                    f"Unknown embedding provider: '{provider}'. "
                    f"Supported: {list(EMBEDDING_PROVIDER_REGISTRY.keys())}"
                )

            logger.info(f"Creating {provider} embedding model ({embedding_class.__name__})")
            self._embedding_model = embedding_class(self._config.embedding)

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

    def get_cache_manager(self) -> CacheManager:
        """Get or create cache manager for AI Gateway.

        Returns:
            CacheManager instance.
        """
        if not self._cache_manager:
            logger.info("Creating CacheManager")
            self._cache_manager = CacheManager(
                ttl_seconds=self._config.ai_gateway.cache_ttl_seconds
            )
        return self._cache_manager

    def get_cost_tracker(self) -> CostTracker:
        """Get or create cost tracker for AI Gateway.

        Returns:
            CostTracker instance.
        """
        if not self._cost_tracker:
            logger.info("Creating CostTracker")
            self._cost_tracker = CostTracker()
        return self._cost_tracker

    def get_llm(self) -> ILanguageModel:
        """Get or create LLM implementation.

        For Gemini: Wraps with fallback to Local using LLMWrapper for resilience.
        For Local: Returns directly without wrapper.
        If AI Gateway enabled: Wraps with AIGateway for cost optimization.

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
            primary_llm = llm_class(self._config.llm)

            # Wrap Gemini with fallback to Local for resilience
            if provider == "gemini":
                logger.info("Wrapping Gemini with fallback to Local")
                fallback_llm = LocalSummarizerModel(self._config.llm)
                self._llm = LLMWrapper(
                    primary_llm=primary_llm,
                    fallback_llm=fallback_llm,
                    max_retries=2
                )
            else:
                self._llm = primary_llm

            # Wrap with AI Gateway if enabled
            if self._config.ai_gateway.enabled:
                logger.info("Wrapping LLM with AI Gateway for cost optimization")
                cache_manager = self.get_cache_manager()
                cost_tracker = self.get_cost_tracker()

                # Create fallback LLM for gateway if not already created
                fallback_llm = fallback_llm if provider == "gemini" else LocalSummarizerModel(self._config.llm)

                self._llm = AIGateway(
                    primary_llm=self._llm,
                    fallback_llm=fallback_llm,
                    config=self._config.ai_gateway,
                    cache_manager=cache_manager,
                    cost_tracker=cost_tracker,
                )

        return self._llm

    def get_scoring_strategy(self) -> IScoringStrategy:
        """Get or create scoring strategy implementation.

        Returns:
            IScoringStrategy implementation (KeywordCountScoringStrategy).
        """
        if not self._scoring_strategy:
            logger.info("Creating KeywordCountScoringStrategy")
            self._scoring_strategy = KeywordCountScoringStrategy()
        return self._scoring_strategy

    def get_thesis_structurer(self) -> IThesisStructurer:
        """Get or create thesis structurer implementation.

        Returns:
            IThesisStructurer implementation (ThesisStructuringService).
        """
        if not self._thesis_structurer:
            logger.info("Creating ThesisStructuringService")
            scoring_strategy = self.get_scoring_strategy()
            self._thesis_structurer = ThesisStructuringService(
                scoring_strategy=scoring_strategy,
                max_results=3
            )
        return self._thesis_structurer

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

    def get_approval_service(self) -> ApprovalService:
        """Get or create approval service.

        Returns:
            ApprovalService instance for human approval workflows.
        """
        if not self._approval_service:
            logger.info("Creating ApprovalService")
            self._approval_service = ApprovalService()
        return self._approval_service

    def get_opportunity_scoring_service(self) -> OpportunityScoringService:
        """Get or create opportunity scoring service.

        Returns:
            OpportunityScoringService instance.
        """
        if not self._opportunity_scoring_service:
            logger.info("Creating OpportunityScoringService")
            self._opportunity_scoring_service = OpportunityScoringService()
        return self._opportunity_scoring_service

    def get_thesis_service(self) -> ThesisGeneratorService:
        """Get or create thesis generator service.

        Returns:
            ThesisGeneratorService instance.
        """
        if not self._thesis_service:
            logger.info("Creating ThesisGeneratorService")
            llm = self.get_llm()
            structurer = self.get_thesis_structurer()
            scoring_service = self.get_opportunity_scoring_service()
            self._thesis_service = ThesisGeneratorService(
                llm=llm,
                structuring_service=structurer,
                scoring_service=scoring_service,
            )
        return self._thesis_service

    def get_refinement_graph(self) -> object:
        """Get or create the compiled LangGraph refinement graph with real tool calling.

        Returns:
            Compiled LangGraph StateGraph ready for invocation.

        Raises:
            NotImplementedError: If Gemini API key is not configured.
        """
        if not self._refinement_graph:
            if not self._config.llm.api_key:
                raise NotImplementedError(
                    "Refinement graph requires a Gemini API key (GOOGLE_API_KEY)."
                )

            logger.info("Creating LangGraph refinement graph with real tool calling")
            from core.agents.refinement_graph import build_refinement_graph

            self._refinement_graph = build_refinement_graph(
                thesis_service=self.get_thesis_service(),
                structuring_service=self.get_thesis_structurer(),
                scoring_service=self.get_opportunity_scoring_service(),
                gemini_api_key=self._config.llm.api_key,
                model_name=self._config.llm.model_name,
            )
        return self._refinement_graph
