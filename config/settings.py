"""Application configuration management."""

from dataclasses import dataclass, field
from typing import List, Optional
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
    provider: str = "huggingface"
    model_name: str = "all-MiniLM-L6-v2"


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "gemini"
    model_name: str = "gemini-3-flash-preview"
    temperature: float = 0.0
    api_key: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    provider: str = "faiss"
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass
class AppConfig:
    """Application-wide configuration."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
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
        """Load configuration from environment variables."""
        config = cls()

        # Load API keys and model names from environment
        if api_key := os.getenv("GOOGLE_API_KEY"):
            config.llm.api_key = api_key

        if model := os.getenv("GEMINI_MODEL"):
            config.llm.model_name = model

        if embed_model := os.getenv("EMBEDDING_MODEL"):
            config.embedding.model_name = embed_model

        if llm_provider := os.getenv("LLM_PROVIDER"):
            config.llm.provider = llm_provider

        if embed_provider := os.getenv("EMBEDDING_PROVIDER"):
            config.embedding.provider = embed_provider

        if vs_provider := os.getenv("VECTORSTORE_PROVIDER"):
            config.vectorstore.provider = vs_provider

        return config
