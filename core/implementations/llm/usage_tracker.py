"""Usage tracker: per-day primary-call counts + token totals (no dollars).

This tracker exists only to:
  1. power the gateway's call-budget guardrail (fall back to the free local
     summarizer once the daily primary-call budget is hit), and
  2. expose lightweight in-process usage metrics.

Only real primary-provider calls count against the budget - cache hits and
local-summarizer calls are free and excluded.
"""

import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd

from core.models.cost_metric import CostMetric

logger = logging.getLogger(__name__)

# Provider labels that never consume the primary-call budget.
_NON_BILLABLE_PROVIDERS = {"local", "cache"}


class UsageTracker:
    """Tracks primary LLM call counts and token usage per day (no dollars)."""

    def __init__(self):
        """Initialize usage tracker."""
        self._metrics: List[CostMetric] = []

    def record_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cache_hit: bool = False,
    ) -> int:
        """Record an LLM call.

        Args:
            provider: Provider label ("primary", "local", "cache", ...).
            model: Resolved model name.
            input_tokens: Estimated input tokens.
            output_tokens: Estimated output tokens.
            latency_ms: Latency in milliseconds.
            cache_hit: Whether this was served from cache.

        Returns:
            Today's primary (billable) call count - exactly what the routing
            guardrail reads, so the caller can log the post-call budget position.
        """
        self._metrics.append(
            CostMetric(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=0.0,  # dollars live in Langfuse, not here
                latency_ms=latency_ms,
                cache_hit=cache_hit,
            )
        )

        count = self.get_daily_calls()
        if cache_hit:
            logger.debug(f"Cache hit recorded: {provider}/{model}")
        else:
            logger.debug(
                f"Call recorded: {provider}/{model} "
                f"({input_tokens}+{output_tokens} tok); daily primary calls={count}"
            )
        return count

    def get_daily_calls(self) -> int:
        """Count today's real primary calls (excludes cache hits and local)."""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return sum(
            1
            for m in self._metrics
            if m.timestamp >= today_start
            and not m.cache_hit
            and m.provider not in _NON_BILLABLE_PROVIDERS
        )

    def get_metrics(self) -> Dict:
        """Get usage metrics (calls + tokens, no dollars).

        Returns:
            Dictionary with call counts and token usage per provider.
        """
        if not self._metrics:
            return {
                "total_calls": 0,
                "cache_hits": 0,
                "cache_hit_rate": 0,
                "daily_primary_calls": 0,
                "avg_latency_ms": 0.0,
                "by_provider": {},
            }

        df = pd.DataFrame([m.to_dict() for m in self._metrics])
        cache_hits = int(df["cache_hit"].sum())
        avg_latency = df["latency_ms"].mean()

        by_provider = (
            df.groupby("provider")
            .agg(
                calls=("provider", "count"),
                input_tokens=("input_tokens", "sum"),
                output_tokens=("output_tokens", "sum"),
            )
            .to_dict(orient="index")
        )

        return {
            "total_calls": len(self._metrics),
            "cache_hits": cache_hits,
            "cache_hit_rate": round((cache_hits / len(self._metrics) * 100), 2),
            "daily_primary_calls": self.get_daily_calls(),
            "avg_latency_ms": round(avg_latency, 2),
            "by_provider": by_provider,
        }

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics = []
        logger.info("Usage metrics cleared")