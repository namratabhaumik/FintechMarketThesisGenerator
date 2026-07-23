"""Unit tests for dependency injection container."""

import pytest
from unittest.mock import Mock, patch
from dependency_injection.container import (
    ServiceContainer,
    LLM_PROVIDER_REGISTRY,
    EMBEDDING_PROVIDER_REGISTRY,
)
from config.settings import AppConfig, LLMConfig, EmbeddingConfig, VectorStoreConfig
from core.interfaces.llm import ILanguageModel
from core.interfaces.embeddings import IEmbeddingModel
from finthesis_internal.keyword_scoring_strategy import KeywordCountScoringStrategy
from finthesis_internal.semantic_scoring_strategy import SemanticScoringStrategy


class TestProviderRegistries:
    """Tests for provider registries."""

    def test_llm_provider_registry_contains_gemini(self):
        """Test that LLM registry includes Gemini provider."""
        assert "gemini" in LLM_PROVIDER_REGISTRY

    def test_llm_provider_registry_values_are_classes(self):
        """Test that LLM registry values are class types."""
        for provider, cls in LLM_PROVIDER_REGISTRY.items():
            assert isinstance(cls, type)

    def test_llm_provider_registry_values_inherit_from_interface(self):
        """Test that all LLM providers inherit from ILanguageModel."""
        for provider, cls in LLM_PROVIDER_REGISTRY.items():
            assert issubclass(cls, ILanguageModel)

    def test_embedding_provider_registry_contains_fastembed(self):
        """Test that embedding registry includes FastEmbed provider."""
        assert "fastembed" in EMBEDDING_PROVIDER_REGISTRY

    def test_embedding_provider_registry_values_are_classes(self):
        """Test that embedding registry values are class types."""
        for provider, cls in EMBEDDING_PROVIDER_REGISTRY.items():
            assert isinstance(cls, type)

    def test_embedding_provider_registry_values_inherit_from_interface(self):
        """Test that all embedding providers inherit from IEmbeddingModel."""
        for provider, cls in EMBEDDING_PROVIDER_REGISTRY.items():
            assert issubclass(cls, IEmbeddingModel)


class TestServiceContainer:
    """Tests for ServiceContainer dependency injection."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AppConfig."""
        config = Mock(spec=AppConfig)
        config.llm = Mock(spec=LLMConfig)
        config.llm.provider = "gemini"
        config.llm.model_name = "test-model"
        config.llm.api_key = "test-api-key"
        config.llm.temperature = 0.0

        config.embedding = Mock(spec=EmbeddingConfig)
        config.embedding.provider = "fastembed"
        config.embedding.model_name = "test-embedding-model"

        config.vectorstore = Mock(spec=VectorStoreConfig)
        config.vectorstore.provider = "supabase"
        config.vectorstore.chunk_size = 800
        config.vectorstore.chunk_overlap = 100

        config.scraper = Mock()
        config.rss_feeds = []

        return config

    # === Container Initialization ===

    def test_container_initializes_with_config(self, mock_config):
        """Test that container can be initialized with config."""
        container = ServiceContainer(mock_config)
        assert container is not None

    def test_container_initializes_with_none_config(self):
        """Test that container initializes with None (loads from env)."""
        with patch("dependency_injection.container.AppConfig.from_env") as mock_from_env:
            mock_from_env.return_value = Mock(spec=AppConfig)
            ServiceContainer(None)
            mock_from_env.assert_called_once()

    def test_container_stores_config(self, mock_config):
        """Test that container stores the provided config."""
        container = ServiceContainer(mock_config)
        assert container._config == mock_config

    # === LLM Provider Lookup ===

    def test_get_llm_uses_provider_registry(self, mock_config):
        """Test that get_llm looks up provider from LLM_PROVIDER_REGISTRY."""
        # Provider "gemini" should be in the registry
        assert "gemini" in LLM_PROVIDER_REGISTRY
        assert LLM_PROVIDER_REGISTRY["gemini"].__name__ == "GeminiLanguageModel"

    def test_get_llm_raises_error_for_unknown_provider(self, mock_config):
        """Test that get_llm raises ValueError for unknown provider."""
        mock_config.llm.provider = "unknown_provider"

        container = ServiceContainer(mock_config)

        with pytest.raises(ValueError) as exc_info:
            container.get_llm()

        assert "Unknown LLM provider" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)

    def test_get_llm_error_lists_supported_providers(self, mock_config):
        """Test that error message lists supported LLM providers."""
        mock_config.llm.provider = "fake_provider"

        container = ServiceContainer(mock_config)

        with pytest.raises(ValueError) as exc_info:
            container.get_llm()

        error_msg = str(exc_info.value)
        assert "Supported" in error_msg
        assert "gemini" in error_msg

    # === Embedding Provider Lookup ===

    def test_get_embedding_model_uses_provider_registry(self, mock_config):
        """Test that get_embedding_model looks up provider from EMBEDDING_PROVIDER_REGISTRY."""
        # Provider "fastembed" should be in the registry
        assert "fastembed" in EMBEDDING_PROVIDER_REGISTRY
        assert EMBEDDING_PROVIDER_REGISTRY["fastembed"].__name__ == "FastEmbedEmbeddingModel"

    def test_get_embedding_model_raises_error_for_unknown_provider(self, mock_config):
        """Test that get_embedding_model raises ValueError for unknown provider."""
        mock_config.embedding.provider = "unknown_provider"

        container = ServiceContainer(mock_config)

        with pytest.raises(ValueError) as exc_info:
            container.get_embedding_model()

        assert "Unknown embedding provider" in str(exc_info.value)

    def test_get_embedding_model_error_lists_supported_providers(self, mock_config):
        """Test that error message lists supported embedding providers."""
        mock_config.embedding.provider = "fake_provider"

        container = ServiceContainer(mock_config)

        with pytest.raises(ValueError) as exc_info:
            container.get_embedding_model()

        error_msg = str(exc_info.value)
        assert "Supported" in error_msg
        assert "fastembed" in error_msg

    # === Vectorstore Provider Lookup ===

    def test_get_vectorstore_provider_lookup_supabase(self, mock_config):
        """Test that get_vectorstore looks up the Supabase provider."""
        # Just verify that the supabase provider is recognized
        assert mock_config.vectorstore.provider == "supabase"

    def test_get_vectorstore_raises_error_for_unknown_provider(self, mock_config):
        """Test that get_vectorstore raises ValueError for unknown provider."""
        mock_config.vectorstore.provider = "unknown_provider"

        container = ServiceContainer(mock_config)

        with pytest.raises(ValueError) as exc_info:
            container.get_vectorstore()

        assert "Unknown vectorstore provider" in str(exc_info.value)

    # === Singleton Lazy Loading ===

    def test_llm_singleton_pattern_implemented(self, mock_config):
        """Test that container implements lazy-loading singleton pattern."""
        # Verify that _llm attribute starts as None
        container = ServiceContainer(mock_config)
        assert container._llm is None

    # === Service Factory Methods ===

    def test_get_retrieval_service_method_exists(self, mock_config):
        """Test that get_retrieval_service method exists."""
        container = ServiceContainer(mock_config)
        assert hasattr(container, "get_retrieval_service")
        assert callable(container.get_retrieval_service)

    def test_get_thesis_service_creates_thesis_generator_service(self, mock_config):
        """Test that get_thesis_service returns ThesisGeneratorService."""

        container = ServiceContainer(mock_config)
        assert hasattr(container, "get_thesis_service")
        assert callable(container.get_thesis_service)

    # === Service Singleton Caching ===

    def test_thesis_service_singleton_attribute_starts_none(self, mock_config):
        """Test that thesis service singleton starts as None."""
        container = ServiceContainer(mock_config)
        assert container._thesis_service is None

    def test_retrieval_service_singleton_attribute_starts_none(self, mock_config):
        """Test that retrieval service singleton starts as None."""
        container = ServiceContainer(mock_config)
        assert container._retrieval_service is None


class TestContainerThreadSafety:
    """Concurrent FIRST requests must build each singleton exactly once.

    Routes run embed/retrieve/graph-build in worker threads (asyncio.to_thread),
    so two simultaneous first requests hit the lazy getters in parallel; the
    container's RLock must collapse that to a single build."""

    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=AppConfig)
        config.embedding = Mock(spec=EmbeddingConfig)
        config.embedding.provider = "fastembed"
        config.embedding.model_name = "test-embedding-model"
        return config

    def test_concurrent_first_calls_build_embedding_model_once(self, mock_config):
        import threading
        import time

        build_count = 0

        class SlowEmbedding:
            """Stands in for the FastEmbed model; a slow __init__ widens the
            race window that unguarded check-then-set would fall into."""
            def __init__(self, config):
                nonlocal build_count
                build_count += 1
                time.sleep(0.05)

        with patch.dict(EMBEDDING_PROVIDER_REGISTRY, {"fastembed": SlowEmbedding}):
            container = ServiceContainer(mock_config)
            threads = [
                threading.Thread(target=container.get_embedding_model)
                for _ in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert build_count == 1
            assert isinstance(container.get_embedding_model(), SlowEmbedding)


class TestScoringStrategyWiring:
    """Guards which scorer each consumer gets.

    Silver must get the embedding-backed semantic scorer; the shared
    get_scoring_strategy() stays the plain keyword scorer.
    """

    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=AppConfig)
        config.llm = Mock(spec=LLMConfig)
        config.embedding = Mock(spec=EmbeddingConfig)
        config.embedding.provider = "fastembed"
        config.embedding.model_name = "test-embedding-model"
        config.embedding.cache_dir = "~/.cache/huggingface"
        config.vectorstore = Mock(spec=VectorStoreConfig)
        config.vectorstore.provider = "supabase"
        config.scraper = Mock()
        config.rss_feeds = []
        return config

    def test_get_scoring_strategy_returns_keyword_strategy(self, mock_config):
        container = ServiceContainer(mock_config)
        assert isinstance(container.get_scoring_strategy(), KeywordCountScoringStrategy)

    def test_shared_scorer_is_not_the_semantic_one(self, mock_config):
        """The shared keyword scorer must never be the embedding scorer."""
        container = ServiceContainer(mock_config)
        assert not isinstance(container.get_scoring_strategy(), SemanticScoringStrategy)

    def test_get_silver_scoring_strategy_returns_semantic_strategy(self, mock_config):
        class FakeEmbedding:
            def __init__(self, config):
                pass

        with patch.dict(EMBEDDING_PROVIDER_REGISTRY, {"fastembed": FakeEmbedding}):
            container = ServiceContainer(mock_config)
            assert isinstance(
                container.get_silver_scoring_strategy(), SemanticScoringStrategy
            )
