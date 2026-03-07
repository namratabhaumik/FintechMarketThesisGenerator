"""Cost tracking model for AI Gateway."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CostMetric:
    """Represents a single LLM API call cost."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    cache_hit: bool = False
    session_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "cache_hit": self.cache_hit,
            "session_id": self.session_id,
        }
