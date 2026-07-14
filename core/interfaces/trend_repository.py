"""Abstract interface for the trend metrics store (medallion: Gold layer)."""

from abc import ABC, abstractmethod
from typing import List, Optional

from core.models.trend_metric import TrendMetric


class ITrendRepository(ABC):
    """Gold layer: the aggregated metrics store (per-category, per-week counts).

    This is the end of the pipeline. Each run --> Gold counts up the Silver tags
    by (dimension, category) and week --> writes those counts here for the app to
    read. Dimension is "theme", "risk", or "signal".

    Counts are recomputed from scratch each run, so a write for a given
    (week, dimension, category) overwrites the old count rather than adding to
    it. That keeps the numbers correct even if the same run executes more than
    once.
    """

    @abstractmethod
    def upsert(self, metrics: List[TrendMetric]) -> int:
        """Insert new rows or overwrite existing ones, keyed by
        (week_start, dimension, category).

        "Upsert" = update if the row already exists, else insert.

        Returns:
            How many metric rows were written.
        """
        pass

    @abstractmethod
    def fetch_all(self) -> List[TrendMetric]:
        """Return all stored metrics, most recent week first."""
        pass

    @abstractmethod
    def fetch_recent(self, window_weeks: Optional[int]) -> List[TrendMetric]:
        """Return the metrics needed for a `window_weeks`-week confidence window.

        Scopes the read to the last `window_weeks` Gold weeks (ending at the
        latest present week) so the transfer stays bounded as history grows.
        `window_weeks=None` (whole-corpus retrieval) returns everything, same as
        fetch_all. The scoped result is equivalent to fetch_all for confidence
        purposes: the window is exactly that range, so nothing counted is lost.
        """
        pass
