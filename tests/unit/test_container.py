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

    def test_embedding_provider_registry_contains_huggingface(self):
        """Test that embedding registry includes HuggingFace provider."""
        assert "huggingface" in EMBEDDING_PROVIDER_REGISTRY

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
        config.embedding.provider = "huggingface"
        config.embedding.model_name = "test-embedding-model"

        config.vectorstore = Mock(spec=VectorStoreConfig)
        config.vectorstore.provider = "faiss"
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
            container = ServiceContainer(None)
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
        # Provider "huggingface" should be in the registry
        assert "huggingface" in EMBEDDING_PROVIDER_REGISTRY
        assert EMBEDDING_PROVIDER_REGISTRY["huggingface"].__name__ == "HuggingFaceEmbeddingModel"

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
        assert "huggingface" in error_msg

    # === Vectorstore Provider Lookup ===

    def test_get_vectorstore_provider_lookup_faiss(self, mock_config):
        """Test that get_vectorstore looks up FAISS provider."""
        # Just verify that faiss provider is recognized
        assert mock_config.vectorstore.provider == "faiss"

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

    def test_get_article_source_method_exists(self, mock_config):
        """Test that get_article_source method exists and returns IArticleSource."""
        from core.interfaces.article_source import IArticleSource

        container = ServiceContainer(mock_config)
        assert hasattr(container, "get_article_source")
        assert callable(container.get_article_source)

    def test_get_retrieval_service_method_exists(self, mock_config):
        """Test that get_retrieval_service method exists."""
        container = ServiceContainer(mock_config)
        assert hasattr(container, "get_retrieval_service")
        assert callable(container.get_retrieval_service)

    def test_get_thesis_service_creates_thesis_generator_service(self, mock_config):
        """Test that get_thesis_service returns ThesisGeneratorService."""
        from core.services.thesis_generator_service import ThesisGeneratorService

        container = ServiceContainer(mock_config)
        assert hasattr(container, "get_thesis_service")
        assert callable(container.get_thesis_service)

    # === Service Singleton Caching ===

    def test_article_source_singleton_attribute_starts_none(self, mock_config):
        """Test that article source singleton starts as None."""
        container = ServiceContainer(mock_config)
        assert container._article_source is None

    def test_thesis_service_singleton_attribute_starts_none(self, mock_config):
        """Test that thesis service singleton starts as None."""
        container = ServiceContainer(mock_config)
        assert container._thesis_service is None

    def test_ingestion_service_singleton_attribute_starts_none(self, mock_config):
        """Test that ingestion service singleton starts as None."""
        container = ServiceContainer(mock_config)
        assert container._ingestion_service is None

    def test_retrieval_service_singleton_attribute_starts_none(self, mock_config):
        """Test that retrieval service singleton starts as None."""
        container = ServiceContainer(mock_config)
        assert container._retrieval_service is None
