"""Abstract interface for the trend metrics store (medallion: Gold layer)."""

from abc import ABC, abstractmethod
from typing import List

from core.models.trend_metric import TrendMetric


class ITrendRepository(ABC):
    """Stores per-theme weekly trend metrics (the Gold layer).

    Metrics are recomputed from Silver each run, so writes overwrite the count
    for an existing (week, theme) rather than accumulating.
    """

    @abstractmethod
    def upsert(self, metrics: List[TrendMetric]) -> int:
        """Insert or overwrite metrics, keyed by (week_start, theme).

        Returns:
            The number of metric rows written.
        """
        pass

    @abstractmethod
    def fetch_all(self) -> List[TrendMetric]:
        """Return all stored metrics, most recent week first."""
        pass
