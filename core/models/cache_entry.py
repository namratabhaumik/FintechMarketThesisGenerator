"""Cache entry model for AI Gateway response caching."""

from dataclasses import dataclass, field
from datetime import datetime

# Model marker for a summary produced by the local extractive fallback during an
# LLM outage or daily-cost-limit hit with a short TTL.
FALLBACK_MODEL = "local-extractor-fallback"
FALLBACK_TTL_SECONDS = 600


def effective_ttl(model: str, default_ttl_seconds: int) -> int:
    """TTL to age an entry by: short for degraded fallback summaries, else default."""
    return FALLBACK_TTL_SECONDS if model == FALLBACK_MODEL else default_ttl_seconds


@dataclass
class CacheEntry:
    """Represents a cached LLM response."""

    key: str
    response: str
    model: str
    input_tokens: int
    output_tokens: int
    created_at: datetime = field(default_factory=datetime.now)

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry has expired based on TTL."""
        age_seconds = (datetime.now() - self.created_at).total_seconds()
        return age_seconds > ttl_seconds

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "key": self.key,
            "response": self.response,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "created_at": self.created_at.isoformat(),
        }
