"""Application configuration management."""

from dataclasses import dataclass, field
from typing import Dict, List
import os

# Registry: maps provider name → env var for its API key.
# To add a new provider: add one entry here and a matching entry in PROVIDER_MODEL_ENV.
# Example: "openai": "OPENAI_API_KEY"
PROVIDER_API_KEY_ENV: Dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
}

# Registry: maps provider name → env var for its model name.
# Example: "openai": "OPENAI_MODEL"
PROVIDER_MODEL_ENV: Dict[str, str] = {
    "gemini": "GEMINI_MODEL",
}


@dataclass
class RSSFeedConfig:
    """Configuration for an RSS feed."""
    name: str
    url: str
    enabled: bool = True


@dataclass
class ScraperConfig:
    """Web scraper configuration."""
    timeout: int = 10
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    provider: str
    model_name: str


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str
    model_name: str
    api_key: str
    temperature: float = 0.0


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    provider: str = "faiss"
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass
class AppConfig:
    """Application-wide configuration."""
    embedding: EmbeddingConfig
    llm: LLMConfig
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    scraper: ScraperConfig = field(default_factory=ScraperConfig)

    rss_feeds: List[RSSFeedConfig] = field(default_factory=lambda: [
        RSSFeedConfig(
            name="TechCrunch Fintech",
            url="https://techcrunch.com/category/fintech/feed/",
            enabled=True
        )
    ])

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load and validate configuration from environment variables.

        Uses PROVIDER_API_KEY_ENV and PROVIDER_MODEL_ENV registries so that
        adding a new LLM provider requires no changes here.

        Raises:
            EnvironmentError: If any required environment variable is missing.
            ValueError: If LLM_PROVIDER is not a recognised provider.
        """
        missing = []

        llm_provider = os.getenv("LLM_PROVIDER")
        embed_provider = os.getenv("EMBEDDING_PROVIDER")
        embed_model = os.getenv("EMBEDDING_MODEL")

        if not llm_provider:
            missing.append("LLM_PROVIDER")
        if not embed_provider:
            missing.append("EMBEDDING_PROVIDER")
        if not embed_model:
            missing.append("EMBEDDING_MODEL")

        # Validate provider is known before resolving its env vars
        if llm_provider and llm_provider not in PROVIDER_API_KEY_ENV:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{llm_provider}'. "
                f"Supported providers: {list(PROVIDER_API_KEY_ENV.keys())}"
            )

        # Resolve provider-specific env var names from registries
        api_key = model_name = None
        if llm_provider:
            api_key_env = PROVIDER_API_KEY_ENV[llm_provider]
            model_env = PROVIDER_MODEL_ENV[llm_provider]

            api_key = os.getenv(api_key_env)
            model_name = os.getenv(model_env)

            if not api_key:
                missing.append(api_key_env)
            if not model_name:
                missing.append(model_env)

        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please set them in your .env file."
            )

        vs_provider = os.getenv("VECTORSTORE_PROVIDER", "faiss")

        return cls(
            llm=LLMConfig(
                provider=llm_provider,
                model_name=model_name,
                api_key=api_key,
            ),
            embedding=EmbeddingConfig(
                provider=embed_provider,
                model_name=embed_model,
            ),
            vectorstore=VectorStoreConfig(provider=vs_provider),
        )
