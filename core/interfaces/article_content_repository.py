"""Abstract interface for the validated article-content store"""

from abc import ABC, abstractmethod
from typing import List

from core.models.article import Article


class IArticleContentRepository(ABC):
    """Silver layer: store of full, validated article text (article_content).

    Silver scrapes full text --> text passes validation --> save it here
    (one row per URL) --> embedding step later reads from here, NOT the web.

    This is a durable copy of the text, so embedding is replayable.
    If embedding fails or the model changes --> re-embed from this store --> no
    need to scrape the web again. Each URL is stored only once (deduped by URL).
    """

    @abstractmethod
    def save(self, articles: List[Article]) -> int:
        """Save validated articles, ignoring any URL already stored.

        for each article --> if its URL is new, store it --> if its URL is
        already here, skip it.

        Args:
            articles: Articles whose full text passed validation.

        Returns:
            How many articles were newly stored (skipped duplicates not counted).
        """
        pass

    @abstractmethod
    def fetch_all(self) -> List[Article]:
        """Return every validated article in the store.

        The embedding step calls this to get the text it needs to embed.
        """
        pass