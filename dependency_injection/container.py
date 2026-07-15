"""Dependency Injection Container for wiring dependencies."""

import logging
import threading
from typing import Dict, Optional, Type

from config.settings import AppConfig
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
from core.implementations.llm.supabase_cache_manager import SupabaseCacheManager
from core.implementations.llm.cost_tracker import CostTracker
from core.implementations.llm.gemini_llm import GeminiLanguageModel
from core.implementations.llm.local_summarizer import LocalSummarizerModel
from core.implementations.llm.llm_wrapper import LLMWrapper
from core.implementations.scrapers.beautifulsoup_scraper import BeautifulSoupScraper
from core.implementations.repositories.supabase_article_repository import (
    SupabaseArticleRepository,
)
from core.implementations.repositories.supabase_article_content_repository import (
    SupabaseArticleContentRepository,
)
from core.implementations.repositories.supabase_quarantine_repository import (
    SupabaseQuarantineRepository,
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
from core.implementations.vectorstores.supabase_vector_store import SupabaseVectorStoreImpl
from core.interfaces.cache import ICacheManager
from core.interfaces.article_content_repository import IArticleContentRepository
from core.interfaces.article_repository import IArticleRepository
from core.interfaces.quarantine_repository import IQuarantineRepository
from core.interfaces.silver_repository import ISilverRepository
from core.interfaces.trend_repository import ITrendRepository
from core.interfaces.untagged_repository import IUntaggedRepository
from core.interfaces.embeddings import IEmbeddingModel
from core.interfaces.llm import ILanguageModel
from core.interfaces.relevance_classifier import IRelevanceClassifier
from core.interfaces.scraper import IWebScraper
from core.interfaces.scoring_strategy import IScoringStrategy
from core.interfaces.vectorstore import IVectorStore
from core.services.gold_service import GoldService
from core.services.silver_service import SilverService
from finthesis_internal.category_mappings import (
    ThemeMappings,
    RiskMappings,
    SignalMappings,
)
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService
from core.services.retrieval_service import DocumentRetrievalService
from core.services.thesis_generator_service import ThesisGeneratorService

# These registries map a config string (e.g. LLM_PROVIDER=gemini) to the class
# that implements it. get_llm/get_embedding_model/get_relevance_classifier look
# up the configured name here, so swapping a backend is a config change, not a
# code change.
# To add a new LLM provider: implement ILanguageModel, register it here, and
# add its API key/model env vars to PROVIDER_API_KEY_ENV/PROVIDER_MODEL_ENV
# in config/settings.py.
LLM_PROVIDER_REGISTRY: Dict[str, Type[ILanguageModel]] = {
    "gemini": GeminiLanguageModel,
    "local": LocalSummarizerModel,
}

# To add a new embedding provider: implement IEmbeddingModel and register it here.
EMBEDDING_PROVIDER_REGISTRY: Dict[str, Type[IEmbeddingModel]] = {
    "fastembed": FastEmbedEmbeddingModel,
}

# Fintech relevance classifier backends (model configurable via CLASSIFIER_MODEL)
CLASSIFIER_PROVIDER_REGISTRY: Dict[str, Type[IRelevanceClassifier]] = {
    "ollama": OllamaFintechClassifier,
    "huggingface": HuggingFaceFintechClassifier,
}


# The vector store needs a live Supabase client to build, so the registry maps
# the provider name to a factory function rather than a bare class. Supabase
# (persistent pgvector) is the only provider
def _build_supabase_store(app_config: "AppConfig", embedding_model) -> IVectorStore:
    # Persistent pgvector store --> needs Supabase creds, so refuse early if missing.
    if not app_config.supabase.enabled:
        raise ValueError(
            "VECTORSTORE_PROVIDER=supabase requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
        )
    from supabase import create_client
    client = create_client(app_config.supabase.url, app_config.supabase.service_role_key)
    return SupabaseVectorStoreImpl(app_config.vectorstore, embedding_model, client)


# To add a new vectorstore provider, add an entry here and a factory function above
VECTORSTORE_PROVIDER_REGISTRY = {
    "supabase": _build_supabase_store,
}

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Dependency Injection Container.

    Follows Dependency Inversion Principle: All dependencies flow through abstractions.
    Implements lazy loading of singletons for efficiency.

    How every get_* method works (the lazy-accessor pattern):
        call get_x() --> is the cached private field empty?
            --> yes: build the object once --> store it in the field --> return it
            --> no: just return the already-built field (same singleton every time)
    A get_* that needs other parts simply calls their get_* accessors, so the
    whole wiring graph builds itself on demand from the leaves up. Nothing is
    constructed until something actually asks for it.

    Where the pieces sit in the data flow:
        Bronze (raw landing) --> article source + Bronze article repository
        --> Silver (refine: classify/scrape/embed + verdict/content/quarantine)
        --> Gold (aggregate: per-theme weekly trends + untagged side-table)
        --> LLM / scoring / retrieval / thesis / agent layers consume the corpus
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize container with configuration.

        Args:
            config: Application configuration. If None, loads from environment.
        """
        self._config = config or AppConfig.from_env()

        # Guards the lazy check-then-set in the getters reached from request
        # handling: routes run embed/retrieve/graph-build in worker threads
        # (asyncio.to_thread), so two concurrent FIRST requests could otherwise
        # both see an empty slot and double-build a singleton (e.g. two ONNX
        # embedding models in memory at once). Re-entrant because getters nest
        # (get_retrieval_service -> get_vectorstore -> get_embedding_model).
        # Offline-pipeline getters (scraper, repos, Silver/Gold services) run
        # single-threaded and stay unguarded.
        self._lock = threading.RLock()

        # Lazy-loaded interface implementations (singletons)
        self._scraper: Optional[IWebScraper] = None
        self._relevance_classifier: Optional[IRelevanceClassifier] = None
        self._embedding_model: Optional[IEmbeddingModel] = None
        self._vectorstore: Optional[IVectorStore] = None
        self._article_repository: Optional[IArticleRepository] = None
        self._silver_repository: Optional[ISilverRepository] = None
        self._content_repository: Optional[IArticleContentRepository] = None
        self._quarantine_repository: Optional[IQuarantineRepository] = None
        self._trend_repository: Optional[ITrendRepository] = None
        self._untagged_repository: Optional[IUntaggedRepository] = None
        self._llm: Optional[ILanguageModel] = None
        self._scoring_strategy: Optional[IScoringStrategy] = None

        # AI Gateway components (singletons)
        self._cache_manager: Optional[ICacheManager] = None
        self._cost_tracker: Optional[CostTracker] = None

        # Services
        self._silver_service: Optional[SilverService] = None
        self._gold_service: Optional[GoldService] = None
        self._retrieval_service: Optional[DocumentRetrievalService] = None
        self._opportunity_scoring_service: Optional[OpportunityScoringService] = None
        self._thesis_service: Optional[ThesisGeneratorService] = None

        # Agents
        self._refinement_graph: Optional[object] = None
        self._langfuse_handler: Optional[object] = None

        logger.info("ServiceContainer initialized")

    # === Factory Methods for Core Interfaces ===

    def get_scraper(self) -> IWebScraper:
        """Get or create web scraper implementation.

        Builds the HTML fetcher (BeautifulSoupScraper) that pulls full article
        text from a URL. Leaf component, depends only on config. Used by the
        Bronze article source and by Silver scraping.

        Returns:
            IWebScraper implementation (BeautifulSoupScraper).
        """
        if not self._scraper:
            logger.info("Creating BeautifulSoupScraper")
            self._scraper = BeautifulSoupScraper(self._config.scraper)
        return self._scraper

    def get_relevance_classifier(self) -> IRelevanceClassifier:
        """Get or create the fintech relevance classifier.

        Builds the model that decides "is this article fintech-relevant?"
        (ollama or huggingface, chosen by CLASSIFIER_PROVIDER). Leaf component,
        depends only on config. Used in Bronze (gate at the source) and again in
        Silver (the per-article verdict).

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

    def get_embedding_model(self) -> IEmbeddingModel:
        """Get or create embedding model implementation.

        Builds the model that turns article text into vectors (fastembed by
        default). Leaf component, depends only on config. Feeds the vector store,
        which Silver writes to and retrieval reads from.

        Returns:
            IEmbeddingModel implementation based on configuration.

        Raises:
            ValueError: If configured embedding provider is not in the registry.
        """
        if not self._embedding_model:
            with self._lock:
                if not self._embedding_model:
                    provider = self._config.embedding.provider
                    embedding_class = EMBEDDING_PROVIDER_REGISTRY.get(provider)

                    if not embedding_class:
                        raise ValueError(
                            f"Unknown embedding provider: '{provider}'. "
                            f"Supported: {list(EMBEDDING_PROVIDER_REGISTRY.keys())}"
                        )

                    logger.info(
                        f"Creating {provider} embedding model ({embedding_class.__name__})"
                    )
                    self._embedding_model = embedding_class(self._config.embedding)

        return self._embedding_model

    def get_vectorstore(self) -> IVectorStore:
        """Get or create vectorstore implementation.

        Builds the searchable store of embedded articles in Supabase. Wires in 
        get_embedding_model. Silver writes embeddings here;
        retrieval queries it.

        Returns:
            IVectorStore implementation based on configuration.

        Raises:
            ValueError: If configured vectorstore provider is unknown.
        """
        if not self._vectorstore:
            with self._lock:
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

        Stores the raw landed articles (the Bronze table). Builds its own
        Supabase client from config. Pattern for every repository below:
        Supabase enabled? --> yes: make client --> wrap in the repo --> cache it
        --> no: raise ValueError (these stores have no in-memory fallback).

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

        Stores the per-article verdict (kept / rejected, theme, score) produced
        during Silver refinement. Supabase-backed; same gating as Bronze above.

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

    def get_content_repository(self) -> IArticleContentRepository:
        """Get or create the validated article-content repository (Silver record).

        Stores the cleaned, scraped article body that passed Silver checks (the
        text that later gets embedded and aggregated). Supabase-backed; same
        gating as above.

        Returns:
            IArticleContentRepository backed by Supabase.

        Raises:
            ValueError: If Supabase is not configured.
        """
        if not self._content_repository:
            if not self._config.supabase.enabled:
                raise ValueError(
                    "The article-content repository requires SUPABASE_URL and "
                    "SUPABASE_SERVICE_ROLE_KEY"
                )
            from supabase import create_client

            logger.info("Creating SupabaseArticleContentRepository (Silver record)")
            client = create_client(
                self._config.supabase.url, self._config.supabase.service_role_key
            )
            self._content_repository = SupabaseArticleContentRepository(client)
        return self._content_repository

    def get_quarantine_repository(self) -> IQuarantineRepository:
        """Get or create the Silver dead-letter / quarantine repository.

        Stores articles that failed Silver processing (scrape error, bad data)
        so they are parked rather than lost - a dead-letter bin. Supabase-backed;
        same gating as above.

        Returns:
            IQuarantineRepository backed by Supabase.

        Raises:
            ValueError: If Supabase is not configured.
        """
        if not self._quarantine_repository:
            if not self._config.supabase.enabled:
                raise ValueError(
                    "The quarantine repository requires SUPABASE_URL and "
                    "SUPABASE_SERVICE_ROLE_KEY"
                )
            from supabase import create_client

            logger.info("Creating SupabaseQuarantineRepository (Silver dead-letter)")
            client = create_client(
                self._config.supabase.url, self._config.supabase.service_role_key
            )
            self._quarantine_repository = SupabaseQuarantineRepository(client)
        return self._quarantine_repository

    def get_trend_repository(self) -> ITrendRepository:
        """Get or create the Gold-layer trend metrics repository.

        Stores the final Gold output: per-theme weekly trend metrics that the
        aggregation step writes. Supabase-backed; same gating as above.

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

        Stores Gold articles that matched no theme - a side-table so they are
        counted/visible instead of silently dropped during aggregation.
        Supabase-backed; same gating as above.

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

    def get_cache_manager(self) -> ICacheManager:
        """Get or create cache manager for AI Gateway.

        Builds the response cache (TTL from config) the AI Gateway uses to avoid
        repeating identical LLM calls. cache_type selects the backend: "memory"
        (per-process, cleared on restart) or "supabase" (persistent, shared).
        The version tag defaults to the primary model name so a model swap busts
        the cache; set AI_GATEWAY_CACHE_VERSION to also bust after a prompt change.

        Returns:
            ICacheManager instance.
        """
        if not self._cache_manager:
            gw = self._config.ai_gateway
            version = gw.cache_version or self._config.llm.model_name

            if gw.cache_type == "supabase":
                if not self._config.supabase.enabled:
                    raise ValueError(
                        "The persistent (supabase) LLM cache requires SUPABASE_URL "
                        "and SUPABASE_SERVICE_ROLE_KEY"
                    )
                from supabase import create_client

                logger.info("Creating SupabaseCacheManager (persistent LLM cache)")
                client = create_client(
                    self._config.supabase.url, self._config.supabase.service_role_key
                )
                self._cache_manager = SupabaseCacheManager(
                    client,
                    ttl_seconds=gw.cache_ttl_seconds,
                    version=version,
                )
            else:
                logger.info("Creating in-memory CacheManager")
                self._cache_manager = CacheManager(
                    ttl_seconds=gw.cache_ttl_seconds,
                    version=version,
                )
        return self._cache_manager

    def get_cost_tracker(self) -> CostTracker:
        """Get or create cost tracker for AI Gateway.

        Builds the token/cost accountant the AI Gateway uses to tally LLM spend.
        Leaf component. Only wired in when the gateway is enabled.

        Returns:
            CostTracker instance.
        """
        if not self._cost_tracker:
            logger.info("Creating CostTracker")
            self._cost_tracker = CostTracker()
        return self._cost_tracker

    def get_llm(self) -> ILanguageModel:
        """Get or create LLM implementation.

        Builds the text-generation model used to write theses, then layers on
        resilience and cost controls:
            pick provider from registry --> build primary model
            --> provider is gemini? wrap it in LLMWrapper with a Local fallback
                (so a Gemini outage degrades instead of failing)
            --> AI Gateway enabled? wrap again in AIGateway (cache + cost tracker)
            --> otherwise return the bare model
        Top-level component consumed by the thesis service and the agent graph.

        For Gemini: Wraps with fallback to Local using LLMWrapper for resilience.
        For Local: Returns directly without wrapper.
        If AI Gateway enabled: Wraps with AIGateway for cost optimization.

        Returns:
            ILanguageModel implementation based on configuration.

        Raises:
            ValueError: If configured LLM provider is not in the registry.
        """
        if not self._llm:
            with self._lock:
                if not self._llm:
                    self._llm = self._build_llm()
        return self._llm

    def _build_llm(self) -> ILanguageModel:
        """Construct the layered LLM (provider + fallback + optional gateway).

        Called only under self._lock from get_llm."""
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
            llm: ILanguageModel = LLMWrapper(
                primary_llm=primary_llm,
                fallback_llm=fallback_llm,
                max_retries=2
            )
        else:
            llm = primary_llm

        # Wrap with AI Gateway if enabled
        if self._config.ai_gateway.enabled:
            logger.info("Wrapping LLM with AI Gateway for cost optimization")
            cache_manager = self.get_cache_manager()
            cost_tracker = self.get_cost_tracker()

            # Create fallback LLM for gateway if not already created
            fallback_llm = fallback_llm if provider == "gemini" else LocalSummarizerModel(self._config.llm)

            llm = AIGateway(
                primary_llm=llm,
                fallback_llm=fallback_llm,
                config=self._config.ai_gateway,
                cache_manager=cache_manager,
                cost_tracker=cost_tracker,
            )

        return llm

    def get_scoring_strategy(self) -> IScoringStrategy:
        """Get or create scoring strategy implementation.

        Builds the keyword-count scorer that ranks how strongly text matches a
        theme. Leaf component, no dependencies. Reused by Silver, the thesis
        structurer, and opportunity scoring.

        Returns:
            IScoringStrategy implementation (KeywordCountScoringStrategy).
        """
        if not self._scoring_strategy:
            logger.info("Creating KeywordCountScoringStrategy")
            self._scoring_strategy = KeywordCountScoringStrategy()
        return self._scoring_strategy

    # === Service Factories ===

    def get_silver_service(self) -> SilverService:
        """Get or create the Silver service.

        The Silver driver and the widest fan-out in this container. It reads
        Bronze, then for each article:
            classify --> scrape body --> score by theme
            --> keep: write verdict + content, embed into the vector store
            --> fail: park in the quarantine (dead-letter) repository
        Wires in the Bronze article repo, all three Silver repos (verdict /
        content / quarantine), the classifier, scraper, scoring strategy, theme
        mappings, and the vector store.

        Reads Bronze, classifies + scrapes, and embeds into the persistent
        vector store. Requires the Supabase pgvector store (it needs
        cross-run persistence and existing_urls()).

        Raises:
            ValueError: If the vectorstore provider is not "supabase".
        """
        if not self._silver_service:
            if self._config.vectorstore.provider != "supabase":
                raise ValueError(
                    "Silver requires VECTORSTORE_PROVIDER=supabase: the corpus "
                    "must persist across runs."
                )
            logger.info("Creating SilverService")
            self._silver_service = SilverService(
                repository=self.get_article_repository(),
                silver_repository=self.get_silver_repository(),
                content_repository=self.get_content_repository(),
                quarantine_repository=self.get_quarantine_repository(),
                classifier=self.get_relevance_classifier(),
                scraper=self.get_scraper(),
                scoring_strategy=self.get_scoring_strategy(),
                theme_categories=ThemeMappings.get_mapping().categories,
                risk_categories=RiskMappings.get_mapping().categories,
                signal_categories=SignalMappings.get_mapping().categories,
                vectorstore=self.get_vectorstore(),
            )
        return self._silver_service

    def get_gold_service(self) -> GoldService:
        """Get or create the Gold aggregation service.

        The Gold driver:
            read Silver content + verdicts --> roll up per theme per week
            --> write trend metrics --> theme-less articles go to the untagged
                side-table (so nothing is silently dropped)
        Wires in the Silver content and verdict repos plus the Gold trend and
        untagged repos.

        Aggregates the fintech corpus into per-theme weekly trend metrics.
        Requires Supabase (Silver content/verdict and trend stores).

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
                content_repository=self.get_content_repository(),
                silver_repository=self.get_silver_repository(),
                trend_repository=self.get_trend_repository(),
                untagged_repository=self.get_untagged_repository(),
            )
        return self._gold_service

    def get_retrieval_service(self) -> DocumentRetrievalService:
        """Get or create document retrieval service.

        The read side of the corpus: queries the vector store for articles
        relevant to a prompt. Wires in get_vectorstore. Feeds the thesis flow.

        Returns:
            DocumentRetrievalService instance.
        """
        if not self._retrieval_service:
            with self._lock:
                if not self._retrieval_service:
                    logger.info("Creating DocumentRetrievalService")
                    vectorstore = self.get_vectorstore()
                    self._retrieval_service = DocumentRetrievalService(
                        vectorstore, self._config.retrieval
                    )
        return self._retrieval_service

    def get_opportunity_scoring_service(self) -> OpportunityScoringService:
        """Get or create opportunity scoring service.

        Scores how strong an investment opportunity a thesis represents, from the
        grounded Silver tag strengths passed in at scoring time. Used by the
        thesis service and the agent graph.

        Returns:
            OpportunityScoringService instance.
        """
        if not self._opportunity_scoring_service:
            logger.info("Creating OpportunityScoringService")
            self._opportunity_scoring_service = OpportunityScoringService()
        return self._opportunity_scoring_service

    def get_thesis_service(self) -> ThesisGeneratorService:
        """Get or create thesis generator service.

        The top-level service that produces a market thesis:
            have the LLM write the narrative --> derive grounded tags from the
            retrieved docs' Silver metadata --> score the opportunity
        Wires in get_llm and the opportunity scoring service. Consumed directly
        and by the agent refinement graph.

        Returns:
            ThesisGeneratorService instance.
        """
        if not self._thesis_service:
            with self._lock:
                if not self._thesis_service:
                    logger.info("Creating ThesisGeneratorService")
                    llm = self.get_llm()
                    scoring_service = self.get_opportunity_scoring_service()
                    self._thesis_service = ThesisGeneratorService(
                        llm=llm,
                        scoring_service=scoring_service,
                        trend_repository=self.get_trend_repository(),
                        retrieval_window_days=self._config.retrieval.window_days,
                    )
        return self._thesis_service

    def get_refinement_graph(self) -> tuple:
        """Get or create the compiled LangGraph refinement graph with real tool calling.

        Builds the agentic loop that iteratively improves a thesis by calling
        tools. Wires in get_thesis_service, and is gated on a Gemini API key (the
        agent needs real tool calling, so a local-only setup cannot run it). The
        refine path carries the numbers forward unchanged, so it does not need the
        scoring service. Returns the graph plus an optional Langfuse tracing handler.

        Returns:
            Tuple of (compiled graph, langfuse handler or None).

        Raises:
            NotImplementedError: If Gemini API key is not configured.
        """
        if not self._refinement_graph:
            with self._lock:
                if not self._refinement_graph:
                    if not self._config.llm.api_key:
                        raise NotImplementedError(
                            "Refinement graph requires a Gemini API key (GOOGLE_API_KEY)."
                        )

                    logger.info("Creating LangGraph refinement graph with real tool calling")
                    from core.agents.refinement_graph import build_refinement_graph

                    self._refinement_graph, self._langfuse_handler = build_refinement_graph(
                        thesis_service=self.get_thesis_service(),
                        gemini_api_key=self._config.llm.api_key,
                        model_name=self._config.llm.model_name,
                        timeout=self._config.llm.timeout,
                        max_output_tokens=self._config.llm.max_output_tokens,
                    )
        return self._refinement_graph, self._langfuse_handler
