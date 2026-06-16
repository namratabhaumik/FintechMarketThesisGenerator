"""Article data models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Article:
    """Represents a fetched article."""
    title: str
    text: str
    source: str
    published_at: datetime
    url: Optional[str] = None

    def __post_init__(self):
        """Validate article data."""
        if not self.title or not self.title.strip():
            raise ValueError("Article title cannot be empty")
        if not self.text or not self.text.strip():
            raise ValueError("Article text cannot be empty")
        if not self.source or not self.source.strip():
            raise ValueError("Article source cannot be empty")
        if not isinstance(self.published_at, datetime):
            raise ValueError("Article published_at must be a datetime")
