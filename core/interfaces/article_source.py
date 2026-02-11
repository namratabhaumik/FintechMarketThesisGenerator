"""Abstract interface for article sources."""

from abc import ABC, abstractmethod
from typing import List

from core.models.article import Article


class IArticleSource(ABC):
    """Protocol for fetching articles from various sources."""

    @abstractmethod
    def fetch_articles(self, query: str, limit: int) -> List[Article]:
        """Fetch articles matching query, up to limit."""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Return the name of this article source."""
        pass
