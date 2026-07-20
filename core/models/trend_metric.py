"""Trend metric model"""

from dataclasses import dataclass, field
from datetime import date
from typing import List


@dataclass
class TrendMetric:
    """Coverage volume for one tag category in one week.

    Generalized across the three Silver tag dimensions: `dimension` is one of
    "theme" / "risk" / "signal", and `category` is the specific label within it
    (e.g. dimension="risk", category="Regulatory Risk"). `week_start` is the
    Monday of the publish week. `article_count` is how many fintech articles
    published that week carried that category (an article can count toward more
    than one category, and toward categories in all three dimensions).
    """

    week_start: date    # Monday of the publish week; the time bucket.
    dimension: str      # Which tag dimension: "theme", "risk", or "signal".
    category: str       # The specific label within that dimension.
    article_count: int  # How many fintech articles that week carried this category.
    load_ids: List[str] = field(default_factory=list) # distinct Bronze ingestion runs whose articles fed this count.

    def __post_init__(self):
        # A metric needs both a dimension and a category to be meaningful, and a
        # coverage count can never be negative - guard so a malformed metric
        # never reaches Gold.
        if not self.dimension or not self.dimension.strip():
            raise ValueError("TrendMetric dimension cannot be empty")
        if not self.category or not self.category.strip():
            raise ValueError("TrendMetric category cannot be empty")
        if self.article_count < 0:
            raise ValueError("TrendMetric article_count cannot be negative")