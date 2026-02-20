"""Application configuration management."""

from dataclasses import dataclass, field
from typing import List
import os


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

        Raises:
            EnvironmentError: If any required environment variable is missing.
        """
        missing = []

        llm_provider = os.getenv("LLM_PROVIDER")
        llm_model = os.getenv("GEMINI_MODEL")
        embed_provider = os.getenv("EMBEDDING_PROVIDER")
        embed_model = os.getenv("EMBEDDING_MODEL")

        # Resolve API key based on provider
        api_key = None
        if llm_provider == "gemini" or llm_provider is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                missing.append("GOOGLE_API_KEY")

        if not llm_provider:
            missing.append("LLM_PROVIDER")
        if not llm_model:
            missing.append("GEMINI_MODEL")
        if not embed_provider:
            missing.append("EMBEDDING_PROVIDER")
        if not embed_model:
            missing.append("EMBEDDING_MODEL")

        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please set them in your .env file."
            )

        vs_provider = os.getenv("VECTORSTORE_PROVIDER", "faiss")

        return cls(
            llm=LLMConfig(
                provider=llm_provider,
                model_name=llm_model,
                api_key=api_key,
            ),
            embedding=EmbeddingConfig(
                provider=embed_provider,
                model_name=embed_model,
            ),
            vectorstore=VectorStoreConfig(provider=vs_provider),
        )
