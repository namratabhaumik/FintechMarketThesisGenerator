"""Dependency Injection Container for wiring dependencies."""

import logging
from typing import Dict, Optional, Type

from config.settings import AppConfig
from core.implementations.article_sources.rss_source import RSSArticleSource
from core.implementations.classifiers.huggingface_classifier import (
    HuggingFaceFintechClassifier,
)
from core.implementations.classifiers.ollama_classifier import OllamaFintechClassifier
from core.implementations.embeddings.fastembed_embeddings import (
    FastEmbedEmbeddingModel,
)
from finthesis_internal.keyword_scoring_strategy import KeywordCountScoringStrategy
from core.implementations.llm.ai_gateway import AIGateway
from core.implementations.llm.cache_manager import CacheManager
from core.implementations.llm.cost_tracker import CostTracker
from core.implementations.llm.gemini_llm import GeminiLanguageModel
from core.implementations.llm.local_summarizer import LocalSummarizerModel
from core.implementations.llm.llm_wrapper import LLMWrapper
from core.implementations.scrapers.beautifulsoup_scraper import BeautifulSoupScraper
from core.implementations.repositories.supabase_article_repository import (
    SupabaseArticleRepository,
)
from core.implementations.repositories.supabase_silver_repository import (
    SupabaseSilverRepository,
)
from core.implementations.repositories.supabase_trend_repository import (
    SupabaseTrendRepository,
)
from core.implementations.repositories.supabase_untagged_repository import (
    SupabaseUntaggedRepository,
)
from core.implementations.vectorstores.faiss_store import FAISSVectorStore
from core.implementations.vectorstores.supabase_vector_store import SupabaseVectorStoreImpl
from core.interfaces.article_repository import IArticleRepository
from core.interfaces.article_source import IArticleSource
from core.interfaces.silver_repository import ISilverRepository
from core.interfaces.trend_repository import ITrendRepository
from core.interfaces.untagged_repository import IUntaggedRepository
from core.interfaces.embeddings import IEmbeddingModel
from core.interfaces.llm import ILanguageModel
from core.interfaces.relevance_classifier import IRelevanceClassifier
from core.interfaces.scraper import IWebScraper
from core.interfaces.scoring_strategy import IScoringStrategy
from core.interfaces.thesis_structurer import IThesisStructurer
from core.interfaces.vectorstore import IVectorStore
from core.services.approval_service import ApprovalService
from core.services.gold_service import GoldService
from core.services.ingestion_service import ArticleIngestionService
from core.services.silver_service import SilverService
from finthesis_internal.category_mappings import ThemeMappings
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService
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
    "fastembed": FastEmbedEmbeddingModel,
}

# Fintech relevance classifier backends (model configurable via CLASSIFIER_MODEL)
CLASSIFIER_PROVIDER_REGISTRY: Dict[str, Type[IRelevanceClassifier]] = {
    "ollama": OllamaFintechClassifier,
    "huggingface": HuggingFaceFintechClassifier,
}


def _build_faiss_store(app_config: "AppConfig", embedding_model) -> IVectorStore:
    return FAISSVectorStore(app_config.vectorstore, embedding_model)


def _build_supabase_store(app_config: "AppConfig", embedding_model) -> IVectorStore:
    if not app_config.supabase.enabled:
        raise ValueError(
            "VECTORSTORE_PROVIDER=supabase requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
        )
    from supabase import create_client
    client = create_client(app_config.supabase.url, app_config.supabase.service_role_key)
    return SupabaseVectorStoreImpl(app_config.vectorstore, embedding_model, client)


# To add a new vectorstore provider, add an entry here and a factory function above
VECTORSTORE_PROVIDER_REGISTRY = {
    "faiss": _build_faiss_store,
    "supabase": _build_supabase_store,
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
        self._relevance_classifier: Optional[IRelevanceClassifier] = None
        self._article_source: Optional[IArticleSource] = None
        self._embedding_model: Optional[IEmbeddingModel] = None
        self._vectorstore: Optional[IVectorStore] = None
        self._article_repository: Optional[IArticleRepository] = None
        self._silver_repository: Optional[ISilverRepository] = None
        self._trend_repository: Optional[ITrendRepository] = None
        self._untagged_repository: Optional[IUntaggedRepository] = None
        self._llm: Optional[ILanguageModel] = None
        self._scoring_strategy: Optional[IScoringStrategy] = None
        self._thesis_structurer: Optional[IThesisStructurer] = None

        # AI Gateway components (singletons)
        self._cache_manager: Optional[CacheManager] = None
        self._cost_tracker: Optional[CostTracker] = None

        # Services
        self._ingestion_service: Optional[ArticleIngestionService] = None
        self._silver_service: Optional[SilverService] = None
        self._gold_service: Optional[GoldService] = None
        self._retrieval_service: Optional[DocumentRetrievalService] = None
        self._approval_service: Optional[ApprovalService] = None
        self._opportunity_scoring_service: Optional[OpportunityScoringService] = None
        self._thesis_service: Optional[ThesisGeneratorService] = None

        # Agents
        self._refinement_graph: Optional[object] = None
        self._langfuse_handler: Optional[object] = None

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

    def get_relevance_classifier(self) -> IRelevanceClassifier:
        """Get or create the fintech relevance classifier.

        Returns:
            IRelevanceClassifier implementation based on configuration.

        Raises:
            ValueError: If the configured classifier provider is unknown.
        """
        if not self._relevance_classifier:
            provider = self._config.classifier.provider
            classifier_class = CLASSIFIER_PROVIDER_REGISTRY.get(provider)

            if not classifier_class:
                raise ValueError(
                    f"Unknown classifier provider: '{provider}'. "
                    f"Supported: {list(CLASSIFIER_PROVIDER_REGISTRY.keys())}"
                )

            logger.info(f"Creating {provider} classifier ({classifier_class.__name__})")
            self._relevance_classifier = classifier_class(self._config.classifier)
        return self._relevance_classifier

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
                scraper=scraper,
                classifier=self.get_relevance_classifier(),
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
            provider = self._config.vectorstore.provider
            factory = VECTORSTORE_PROVIDER_REGISTRY.get(provider)

            if not factory:
                raise ValueError(
                    f"Unknown vectorstore provider: '{provider}'. "
                    f"Supported: {list(VECTORSTORE_PROVIDER_REGISTRY.keys())}"
                )

            logger.info(f"Creating {provider} vectorstore")
            embedding_model = self.get_embedding_model()
            self._vectorstore = factory(self._config, embedding_model)

        return self._vectorstore

    def get_article_repository(self) -> IArticleRepository:
        """Get or create the Bronze-layer raw article repository.

        Returns:
            IArticleRepository backed by Supabase.

        Raises:
            ValueError: If Supabase is not configured.
        """
        if not self._article_repository:
            if not self._config.supabase.enabled:
                raise ValueError(
                    "The Bronze article repository requires SUPABASE_URL and "
                    "SUPABASE_SERVICE_ROLE_KEY"
                )
            from supabase import create_client

            logger.info("Creating SupabaseArticleRepository (Bronze)")
            client = create_client(
                self._config.supabase.url, self._config.supabase.service_role_key
            )
            self._article_repository = SupabaseArticleRepository(client)
        return self._article_repository

    def get_silver_repository(self) -> ISilverRepository:
        """Get or create the Silver-layer verdict repository.

        Returns:
            ISilverRepository backed by Supabase.

        Raises:
            ValueError: If Supabase is not configured.
        """
        if not self._silver_repository:
            if not self._config.supabase.enabled:
                raise ValueError(
                    "The Silver verdict repository requires SUPABASE_URL and "
                    "SUPABASE_SERVICE_ROLE_KEY"
                )
            from supabase import create_client

            logger.info("Creating SupabaseSilverRepository (Silver verdicts)")
            client = create_client(
                self._config.supabase.url, self._config.supabase.service_role_key
            )
            self._silver_repository = SupabaseSilverRepository(client)
        return self._silver_repository

    def get_trend_repository(self) -> ITrendRepository:
        """Get or create the Gold-layer trend metrics repository.

        Returns:
            ITrendRepository backed by Supabase.

        Raises:
            ValueError: If Supabase is not configured.
        """
        if not self._trend_repository:
            if not self._config.supabase.enabled:
                raise ValueError(
                    "The Gold trend repository requires SUPABASE_URL and "
                    "SUPABASE_SERVICE_ROLE_KEY"
                )
            from supabase import create_client

            logger.info("Creating SupabaseTrendRepository (Gold trends)")
            client = create_client(
                self._config.supabase.url, self._config.supabase.service_role_key
            )
            self._trend_repository = SupabaseTrendRepository(client)
        return self._trend_repository

    def get_untagged_repository(self) -> IUntaggedRepository:
        """Get or create the untagged-article capture repository (Gold side-table).

        Returns:
            IUntaggedRepository backed by Supabase.

        Raises:
            ValueError: If Supabase is not configured.
        """
        if not self._untagged_repository:
            if not self._config.supabase.enabled:
                raise ValueError(
                    "The untagged-article repository requires SUPABASE_URL and "
                    "SUPABASE_SERVICE_ROLE_KEY"
                )
            from supabase import create_client

            logger.info("Creating SupabaseUntaggedRepository (untagged capture)")
            client = create_client(
                self._config.supabase.url, self._config.supabase.service_role_key
            )
            self._untagged_repository = SupabaseUntaggedRepository(client)
        return self._untagged_repository

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

    def get_silver_service(self) -> SilverService:
        """Get or create the Silver service.

        Reads Bronze, classifies + scrapes, and embeds into the persistent
        vector store. Requires the Supabase pgvector store (it needs
        cross-run persistence and existing_urls()).

        Raises:
            ValueError: If the vectorstore provider is not "supabase".
        """
        if not self._silver_service:
            if self._config.vectorstore.provider != "supabase":
                raise ValueError(
                    "Silver requires VECTORSTORE_PROVIDER=supabase "
                    "(a persistent corpus); in-memory FAISS cannot back it."
                )
            logger.info("Creating SilverService")
            self._silver_service = SilverService(
                repository=self.get_article_repository(),
                silver_repository=self.get_silver_repository(),
                classifier=self.get_relevance_classifier(),
                scraper=self.get_scraper(),
                scoring_strategy=self.get_scoring_strategy(),
                theme_categories=ThemeMappings.get_mapping().categories,
                vectorstore=self.get_vectorstore(),
            )
        return self._silver_service

    def get_gold_service(self) -> GoldService:
        """Get or create the Gold aggregation service.

        Aggregates the fintech corpus into per-theme weekly trend metrics.
        Requires Supabase (Bronze, Silver and trend stores).

        Raises:
            ValueError: If Supabase is not configured.
        """
        if not self._gold_service:
            if not self._config.supabase.enabled:
                raise ValueError(
                    "Gold aggregation requires Supabase (SUPABASE_URL and "
                    "SUPABASE_SERVICE_ROLE_KEY)"
                )
            logger.info("Creating GoldService")
            self._gold_service = GoldService(
                article_repository=self.get_article_repository(),
                silver_repository=self.get_silver_repository(),
                trend_repository=self.get_trend_repository(),
                untagged_repository=self.get_untagged_repository(),
            )
        return self._gold_service

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
            self._opportunity_scoring_service = OpportunityScoringService(
                scoring_strategy=self.get_scoring_strategy()
            )
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

    def get_refinement_graph(self) -> tuple:
        """Get or create the compiled LangGraph refinement graph with real tool calling.

        Returns:
            Tuple of (compiled graph, langfuse handler or None).

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

            self._refinement_graph, self._langfuse_handler = build_refinement_graph(
                thesis_service=self.get_thesis_service(),
                scoring_service=self.get_opportunity_scoring_service(),
                gemini_api_key=self._config.llm.api_key,
                model_name=self._config.llm.model_name,
            )
        return self._refinement_graph, self._langfuse_handler
