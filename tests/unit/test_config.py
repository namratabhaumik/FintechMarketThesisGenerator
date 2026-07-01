"""Unit tests for application configuration."""

import os
import pytest
from config.settings import AppConfig, EmbeddingConfig, LLMConfig, VectorStoreConfig


class TestAppConfigFromEnv:
    """Tests for AppConfig.from_env() configuration loading."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Clean up relevant environment variables before and after tests."""
        # Store original values
        original = {
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER"),
            "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER"),
            "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
            "VECTORSTORE_PROVIDER": os.getenv("VECTORSTORE_PROVIDER"),
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "CLASSIFIER_PROVIDER": os.getenv("CLASSIFIER_PROVIDER"),
            "CLASSIFIER_MODEL": os.getenv("CLASSIFIER_MODEL"),
            "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL"),
        }
        # Default classifier provider is "ollama", which needs no token. Tests
        # that exercise the huggingface provider set CLASSIFIER_PROVIDER/HF_TOKEN.
        os.environ.pop("CLASSIFIER_PROVIDER", None)
        os.environ.pop("CLASSIFIER_MODEL", None)
        os.environ.pop("HF_TOKEN", None)
        yield
        # Restore original values
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # === Valid Configuration Tests ===

    def test_load_valid_gemini_config(self):
        """Test loading valid Gemini configuration from environment."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_api_key_123"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        config = AppConfig.from_env()

        assert config.llm.provider == "gemini"
        assert config.llm.model_name == "gemini-2.0-flash"
        assert config.llm.api_key == "test_api_key_123"
        assert config.embedding.provider == "huggingface"
        assert config.embedding.model_name == "all-MiniLM-L6-v2"

    def test_load_config_with_default_vectorstore(self):
        """Test that vectorstore defaults to 'supabase' if not specified."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
        # Don't set VECTORSTORE_PROVIDER

        config = AppConfig.from_env()

        assert config.vectorstore.provider == "supabase"

    def test_load_config_with_custom_vectorstore(self):
        """Test loading with custom vectorstore provider."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
        os.environ["VECTORSTORE_PROVIDER"] = "custom_provider"

        config = AppConfig.from_env()

        assert config.vectorstore.provider == "custom_provider"

    def test_llm_config_has_default_temperature(self):
        """Test that LLMConfig uses default temperature."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        config = AppConfig.from_env()

        assert config.llm.temperature == 0.0

    # === Missing Environment Variable Tests ===

    def test_missing_llm_provider(self):
        """Test error when LLM_PROVIDER is missing."""
        os.environ.pop("LLM_PROVIDER", None)
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        with pytest.raises(EnvironmentError) as exc_info:
            AppConfig.from_env()

        assert "LLM_PROVIDER" in str(exc_info.value)

    def test_missing_embedding_provider(self):
        """Test error when EMBEDDING_PROVIDER is missing."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ.pop("EMBEDDING_PROVIDER", None)
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        with pytest.raises(EnvironmentError) as exc_info:
            AppConfig.from_env()

        assert "EMBEDDING_PROVIDER" in str(exc_info.value)

    def test_missing_embedding_model(self):
        """Test error when EMBEDDING_MODEL is missing."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ.pop("EMBEDDING_MODEL", None)

        with pytest.raises(EnvironmentError) as exc_info:
            AppConfig.from_env()

        assert "EMBEDDING_MODEL" in str(exc_info.value)

    def test_missing_api_key_for_provider(self):
        """Test error when provider API key env var is missing."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ.pop("GOOGLE_API_KEY", None)  # This is the API key for Gemini
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        with pytest.raises(EnvironmentError) as exc_info:
            AppConfig.from_env()

        assert "GOOGLE_API_KEY" in str(exc_info.value)

    def test_missing_model_name_for_provider(self):
        """Test error when provider model name env var is missing."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ.pop("GEMINI_MODEL", None)  # This is the model name for Gemini
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        with pytest.raises(EnvironmentError) as exc_info:
            AppConfig.from_env()

        assert "GEMINI_MODEL" in str(exc_info.value)

    def test_missing_hf_token_for_huggingface_provider(self):
        """HF_TOKEN is required only when CLASSIFIER_PROVIDER=huggingface."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
        os.environ["CLASSIFIER_PROVIDER"] = "huggingface"
        os.environ.pop("HF_TOKEN", None)

        with pytest.raises(EnvironmentError) as exc_info:
            AppConfig.from_env()

        assert "HF_TOKEN" in str(exc_info.value)

    def test_classifier_defaults_to_ollama_without_token(self):
        """Default classifier provider is ollama and needs no HF token."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
        os.environ.pop("CLASSIFIER_PROVIDER", None)
        os.environ.pop("HF_TOKEN", None)

        config = AppConfig.from_env()

        assert config.classifier.provider == "ollama"
        assert config.classifier.base_url == "http://localhost:11434"
        assert config.classifier.model == "qwen2.5:7b"

    def test_classifier_model_default_for_huggingface(self):
        """The huggingface provider defaults to the Qwen 7B HF repo id."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
        os.environ["CLASSIFIER_PROVIDER"] = "huggingface"
        os.environ["HF_TOKEN"] = "hf_x"

        config = AppConfig.from_env()

        assert config.classifier.model == "Qwen/Qwen2.5-7B-Instruct"

    def test_classifier_model_override(self):
        """CLASSIFIER_MODEL overrides the per-provider default."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
        os.environ["CLASSIFIER_MODEL"] = "llama3.2:3b"

        config = AppConfig.from_env()

        assert config.classifier.model == "llama3.2:3b"

    def test_hf_token_loaded_into_classifier_config(self):
        """Test that HF_TOKEN is loaded into the classifier config."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
        os.environ["CLASSIFIER_PROVIDER"] = "huggingface"
        os.environ["HF_TOKEN"] = "hf_secret_123"

        config = AppConfig.from_env()

        assert config.classifier.provider == "huggingface"
        assert config.classifier.api_key == "hf_secret_123"

    def test_ollama_api_key_loaded_into_classifier_config(self):
        """OLLAMA_API_KEY is loaded as the ollama classifier's bearer token."""
        os.environ["LLM_PROVIDER"] = "local"
        os.environ["EMBEDDING_PROVIDER"] = "fastembed"
        os.environ["EMBEDDING_MODEL"] = "jinaai/jina-embeddings-v2-small-en"
        os.environ["CLASSIFIER_PROVIDER"] = "ollama"
        os.environ["OLLAMA_API_KEY"] = "ollama_cloud_key"
        os.environ.pop("HF_TOKEN", None)

        config = AppConfig.from_env()

        assert config.classifier.provider == "ollama"
        assert config.classifier.api_key == "ollama_cloud_key"

    def test_multiple_missing_vars_in_error_message(self):
        """Test that all missing vars are listed in error message."""
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("EMBEDDING_PROVIDER", None)
        os.environ.pop("EMBEDDING_MODEL", None)

        with pytest.raises(EnvironmentError) as exc_info:
            AppConfig.from_env()

        error_msg = str(exc_info.value)
        assert "LLM_PROVIDER" in error_msg
        assert "EMBEDDING_PROVIDER" in error_msg
        assert "EMBEDDING_MODEL" in error_msg

    # === Unknown Provider Tests ===

    def test_unknown_llm_provider(self):
        """Test error when unknown LLM provider is specified."""
        os.environ["LLM_PROVIDER"] = "unknown_provider"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        with pytest.raises(ValueError) as exc_info:
            AppConfig.from_env()

        assert "Unknown LLM_PROVIDER" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)

    def test_error_message_lists_supported_providers(self):
        """Test that error message lists supported providers."""
        os.environ["LLM_PROVIDER"] = "fake_provider"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        with pytest.raises(ValueError) as exc_info:
            AppConfig.from_env()

        assert "Supported providers" in str(exc_info.value)
        assert "gemini" in str(exc_info.value)

    # === Provider-Specific Env Var Resolution Tests ===

    def test_provider_api_key_env_resolution_gemini(self):
        """Test that 'gemini' provider resolves to GOOGLE_API_KEY."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "gemini_key_123"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        config = AppConfig.from_env()

        assert config.llm.api_key == "gemini_key_123"

    def test_provider_model_env_resolution_gemini(self):
        """Test that 'gemini' provider resolves to GEMINI_MODEL."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        config = AppConfig.from_env()

        assert config.llm.model_name == "gemini-2.0-flash"

    # === Configuration Object Types ===

    def test_config_returns_llm_config_instance(self):
        """Test that config.llm is an LLMConfig instance."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        config = AppConfig.from_env()

        assert isinstance(config.llm, LLMConfig)

    def test_config_returns_embedding_config_instance(self):
        """Test that config.embedding is an EmbeddingConfig instance."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        config = AppConfig.from_env()

        assert isinstance(config.embedding, EmbeddingConfig)

    def test_config_returns_vectorstore_config_instance(self):
        """Test that config.vectorstore is a VectorStoreConfig instance."""
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.0-flash"
        os.environ["GOOGLE_API_KEY"] = "test_key"
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"

        config = AppConfig.from_env()

        assert isinstance(config.vectorstore, VectorStoreConfig)
