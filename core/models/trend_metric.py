"""Trend metric model"""

from dataclasses import dataclass
from datetime import date


@dataclass
class TrendMetric:
    """Coverage volume for one fintech theme in one week.

    `week_start` is the Monday of the publish week. `article_count` is how many
    fintech articles published that week matched the theme (an article can count
    toward more than one theme).
    """

    week_start: date    # Monday of the publish week; the time bucket.
    theme: str          # The fintech theme this count is for.
    article_count: int  # How many fintech articles that week matched the theme.

    def __post_init__(self):
        # A metric with no theme is meaningless, and a coverage count can never
        # be negative - guard both so a malformed metric never reaches Gold.
        if not self.theme or not self.theme.strip():
            raise ValueError("TrendMetric theme cannot be empty")
        if self.article_count < 0:
            raise ValueError("TrendMetric article_count cannot be negative")