"""Article data models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Article:
    """Represents a fetched article."""
    title: str
    text: str
    source: str
    url: Optional[str] = None

    def __post_init__(self):
        """Validate article data."""
        if not self.title or not self.title.strip():
            raise ValueError("Article title cannot be empty")
        if not self.text or not self.text.strip():
            raise ValueError("Article text cannot be empty")
        if not self.source or not self.source.strip():
            raise ValueError("Article source cannot be empty")
