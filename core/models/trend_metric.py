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

    week_start: date
    theme: str
    article_count: int

    def __post_init__(self):
        if not self.theme or not self.theme.strip():
            raise ValueError("TrendMetric theme cannot be empty")
        if self.article_count < 0:
            raise ValueError("TrendMetric article_count cannot be negative")