"""Cache entry model for AI Gateway response caching."""

from dataclasses import dataclass, field
from datetime import datetime


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
