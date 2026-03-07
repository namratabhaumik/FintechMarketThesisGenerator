"""Cost tracker for AI Gateway API call tracking."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from core.models.cost_metric import CostMetric

logger = logging.getLogger(__name__)

# Provider pricing (per 1M tokens)
PROVIDER_PRICING = {
    "gemini": {
        "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
        "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    },
    "local": {
        "local-extractor": {"input": 0.0, "output": 0.0},
    },
}


class CostTracker:
    """Tracks costs of LLM API calls."""

    def __init__(self):
        """Initialize cost tracker."""
        self._metrics: List[CostMetric] = []

    def calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for an API call.

        Args:
            provider: Provider name (e.g., "gemini", "local").
            model: Model name (e.g., "gemini-2.0-flash").
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD.
        """
        if provider not in PROVIDER_PRICING:
            logger.warning(f"Unknown provider: {provider}")
            return 0.0

        provider_models = PROVIDER_PRICING[provider]
        if model not in provider_models:
            logger.warning(f"Unknown model: {model} for provider: {provider}")
            return 0.0

        pricing = provider_models[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    def record_call(self, provider: str, model: str, input_tokens: int,
                    output_tokens: int, latency_ms: float, cache_hit: bool = False) -> float:
        """Record an LLM API call.

        Args:
            provider: Provider name.
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            latency_ms: Latency in milliseconds.
            cache_hit: Whether this was a cache hit.

        Returns:
            Cost in USD.
        """
        cost = self.calculate_cost(provider, model, input_tokens, output_tokens)

        metric = CostMetric(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )

        self._metrics.append(metric)

        if cache_hit:
            logger.debug(f"Cache hit: saved ${cost:.6f}")
        else:
            logger.debug(f"API call recorded: {provider}/{model} - ${cost:.6f}")

        return cost

    def get_daily_spend(self) -> float:
        """Get total spend for today.

        Returns:
            Total cost in USD for the current day.
        """
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        return sum(
            m.cost for m in self._metrics
            if m.timestamp >= today_start
        )

    def get_monthly_spend(self) -> float:
        """Get total spend for the current month.

        Returns:
            Total cost in USD for the current month.
        """
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        return sum(
            m.cost for m in self._metrics
            if m.timestamp >= month_start
        )

    def get_metrics(self) -> Dict[str, any]:
        """Get comprehensive cost metrics.

        Returns:
            Dictionary with cost and usage metrics.
        """
        if not self._metrics:
            return {
                "total_calls": 0,
                "cache_hits": 0,
                "daily_spend": 0.0,
                "monthly_spend": 0.0,
                "avg_latency_ms": 0.0,
                "by_provider": {},
            }

        cache_hits = sum(1 for m in self._metrics if m.cache_hit)
        avg_latency = sum(m.latency_ms for m in self._metrics) / len(self._metrics)

        # Group by provider
        by_provider = {}
        for metric in self._metrics:
            if metric.provider not in by_provider:
                by_provider[metric.provider] = {
                    "calls": 0,
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            by_provider[metric.provider]["calls"] += 1
            by_provider[metric.provider]["cost"] += metric.cost
            by_provider[metric.provider]["input_tokens"] += metric.input_tokens
            by_provider[metric.provider]["output_tokens"] += metric.output_tokens

        return {
            "total_calls": len(self._metrics),
            "cache_hits": cache_hits,
            "cache_hit_rate": round((cache_hits / len(self._metrics) * 100), 2) if self._metrics else 0,
            "daily_spend": round(self.get_daily_spend(), 4),
            "monthly_spend": round(self.get_monthly_spend(), 4),
            "avg_latency_ms": round(avg_latency, 2),
            "by_provider": by_provider,
        }

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()
        logger.info("Cost metrics cleared")
