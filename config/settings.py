"""Application configuration management."""

from dataclasses import dataclass, field
from typing import Dict, List
import os

# Registry: maps provider name → env var for its API key.
# To add a new provider: add one entry here and a matching entry in PROVIDER_MODEL_ENV.
# Example: "openai": "OPENAI_API_KEY"
# Note: "local" provider does not require an API key.
PROVIDER_API_KEY_ENV: Dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "local": "",  # Local provider doesn't need an API key
}

# Registry: maps provider name → env var for its model name.
# Example: "openai": "OPENAI_MODEL"
# Note: "local" provider always uses "local-extractor" model.
PROVIDER_MODEL_ENV: Dict[str, str] = {
    "gemini": "GEMINI_MODEL",
    "local": "LOCAL_MODEL",
}

# Fixed set of refinement feedback reasons.
FEEDBACK_OPTIONS: List[str] = [
    "Too many risks, not enough opportunities",
    "Missing recent market trends",
    "Investment signals are too vague",
    "Opportunity score seems too low",
    "Analysis is too broad, be more specific",
    "Need stronger evidence for key themes",
]


@dataclass
class RSSFeedConfig:
    """Configuration for an RSS feed."""
    name: str
    url: str
    enabled: bool = True


@dataclass
class ClassifierConfig:
    """Fintech relevance classifier configuration.

    `provider` selects the backend: "ollama" (local) or "huggingface" (hosted).
    `model` is the chat model to run, so each user can pick one that
    fits their hardware; from_env resolves a provider-appropriate default.
    `api_key` is the HF token (huggingface only); `base_url` is the Ollama
    server (ollama only).
    """
    provider: str = "ollama"
    model: str = ""
    api_key: str = ""
    base_url: str = "http://localhost:11434"
    timeout: int = 30


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
    # FastEmbed's ONNX model cache dir. Defaults to the HuggingFace cache (the
    # models are HF-hosted) so it matches the daily-ingest actions/cache path.
    cache_dir: str = "~/.cache/huggingface"


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str
    model_name: str
    api_key: str
    temperature: float = 0.0
    # upper limit with headroom above the observed max (llm timeout + token limit)
    timeout: int = 120
    max_output_tokens: int = 4096
    timeout: int = 60
    max_output_tokens: int = 2048


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    provider: str = "supabase"
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass
class RetrievalConfig:
    """MMR (Maximal Marginal Relevance) retrieval config.

    It first pulls `fetch_k` candidates by similarity, then selects `k` of them 
    by the MMR objective so the returned chunks are not near-duplicates of each 
    other. `lambda_mult` is the relevance/diversity dial

    `window_days` is a trailing recency window: retrieval only considers articles
    published within the last `window_days` from the query time (a sliding window
    that moves as time advances). The corpus is sparse and historic, so the
    default is a broad year. Set it to 0 to disable the filter and search the
    whole corpus.

    `min_similarity` cosine floor applied to `fetch_k` candidates before
    MMR. Chunks scoring below it are dropped (value specific to EMBEDDING_MODEL & corpus):
    ~0.67 off-topic .. ~0.84 on-topic.
    """
    k: int = 5
    fetch_k: int = 20
    lambda_mult: float = 0.5
    window_days: int = 365
    min_similarity: float = 0.72


@dataclass
class SupabaseConfig:
    """Supabase connection configuration."""
    url: str = ""
    service_role_key: str = ""
    enabled: bool = False


@dataclass
class AIGatewayConfig:
    """AI Gateway configuration for cost optimization."""
    enabled: bool = True
    strategy: str = "hybrid"  # cost_optimized | quality_first | hybrid
    cache_enabled: bool = True
    cache_type: str = "memory"  # memory | sqlite
    cache_ttl_seconds: int = 604800  # 7 days
    cost_limit_daily: float = 5.0
    cost_limit_monthly: float = 100.0
    track_metrics: bool = True


@dataclass
class AppConfig:
    """Application-wide configuration."""
    embedding: EmbeddingConfig
    llm: LLMConfig
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    scraper: ScraperConfig = field(default_factory=ScraperConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    supabase: SupabaseConfig = field(default_factory=SupabaseConfig)
    ai_gateway: AIGatewayConfig = field(default_factory=AIGatewayConfig)

    rss_feeds: List[RSSFeedConfig] = field(default_factory=lambda: [
        RSSFeedConfig(
            name="TechCrunch",
            url="https://techcrunch.com/feed/",
            enabled=True
        ),
        RSSFeedConfig(
            name="TechCrunch Fintech (category)",
            url="https://techcrunch.com/category/fintech/feed/",
            enabled=True
        ),
        RSSFeedConfig(
            name="TechCrunch Fintech (tag)",
            url="https://techcrunch.com/tag/fintech/feed/",
            enabled=True
        ),
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

            # Local provider doesn't need an API key
            if api_key_env:
                api_key = os.getenv(api_key_env)
                if not api_key:
                    missing.append(api_key_env)
            else:
                # Local provider, use empty string
                api_key = ""

            model_name = os.getenv(model_env)
            if not model_name:
                # Local provider has a default model name
                if llm_provider == "local":
                    model_name = "local-extractor"
                else:
                    missing.append(model_env)

        # Fintech relevance classifier — always on. Default provider is local
        # Ollama (no token); HF_TOKEN is required only for the huggingface provider.
        classifier_provider = os.getenv("CLASSIFIER_PROVIDER", "ollama").lower()
        hf_token = os.getenv("HF_TOKEN", "")
        # Bearer token for a hosted Ollama endpoint (e.g. Ollama Cloud). Empty
        # for a local server, which needs no auth.
        ollama_api_key = os.getenv("OLLAMA_API_KEY", "")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Default model differs by backend (Ollama tag vs HF repo id).
        default_model = (
            "Qwen/Qwen2.5-7B-Instruct"
            if classifier_provider == "huggingface"
            else "qwen2.5:7b"
        )
        classifier_model = os.getenv("CLASSIFIER_MODEL", default_model)
        if classifier_provider == "huggingface" and not hf_token:
            missing.append("HF_TOKEN")

        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please set them in your .env file."
            )

        vs_provider = os.getenv("VECTORSTORE_PROVIDER", "supabase")

        # Retrieval config defaults for MMR
        retrieval = RetrievalConfig(
            k=int(os.getenv("RETRIEVAL_K", "5")),
            fetch_k=int(os.getenv("RETRIEVAL_FETCH_K", "20")),
            lambda_mult=float(os.getenv("RETRIEVAL_LAMBDA_MULT", "0.5")),
            window_days=int(os.getenv("RETRIEVAL_WINDOW_DAYS", "365")),
            min_similarity=float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.72")),
        )

        # Supabase configuration (optional — falls back to in-memory if not set)
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        supabase_enabled = bool(supabase_url and supabase_service_role_key)

        # Load AI Gateway configuration
        ai_gateway_enabled = os.getenv("AI_GATEWAY_ENABLED", "true").lower() == "true"
        ai_gateway_strategy = os.getenv("AI_GATEWAY_STRATEGY", "hybrid")
        ai_gateway_cache_enabled = os.getenv("AI_GATEWAY_CACHE_ENABLED", "true").lower() == "true"
        ai_gateway_cache_type = os.getenv("AI_GATEWAY_CACHE_TYPE", "memory")
        ai_gateway_cache_ttl = int(os.getenv("AI_GATEWAY_CACHE_TTL_SECONDS", "604800"))
        ai_gateway_cost_limit_daily = float(os.getenv("AI_GATEWAY_COST_LIMIT_DAILY", "5.0"))
        ai_gateway_cost_limit_monthly = float(os.getenv("AI_GATEWAY_COST_LIMIT_MONTHLY", "100.0"))
        ai_gateway_track_metrics = os.getenv("AI_GATEWAY_TRACK_METRICS", "true").lower() == "true"

        return cls(
            llm=LLMConfig(
                provider=llm_provider,
                model_name=model_name,
                api_key=api_key,
                timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
                max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096")),
            ),
            embedding=EmbeddingConfig(
                provider=embed_provider,
                model_name=embed_model,
                cache_dir=os.getenv("FASTEMBED_CACHE_DIR", "~/.cache/huggingface"),
            ),
            classifier=ClassifierConfig(
                provider=classifier_provider,
                # HF uses the HF token; Ollama uses the cloud bearer token.
                model=classifier_model,
                api_key=(hf_token if classifier_provider == "huggingface" else ollama_api_key),
                base_url=ollama_base_url,
            ),
            vectorstore=VectorStoreConfig(provider=vs_provider),
            retrieval=retrieval,
            supabase=SupabaseConfig(
                url=supabase_url,
                service_role_key=supabase_service_role_key,
                enabled=supabase_enabled,
            ),
            ai_gateway=AIGatewayConfig(
                enabled=ai_gateway_enabled,
                strategy=ai_gateway_strategy,
                cache_enabled=ai_gateway_cache_enabled,
                cache_type=ai_gateway_cache_type,
                cache_ttl_seconds=ai_gateway_cache_ttl,
                cost_limit_daily=ai_gateway_cost_limit_daily,
                cost_limit_monthly=ai_gateway_cost_limit_monthly,
                track_metrics=ai_gateway_track_metrics,
            ),
        )
