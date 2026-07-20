"""Article data models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Article:
    """A fetched article with its full text - the Silver working model.

    This is the cleaned, validated form an article takes once it has been
    scraped (unlike RawArticle, which is the thin Bronze feed entry). It is what
    gets embedded into the vector store and, via article_content, what Gold reads
    publish dates from.
    """
    title: str          # Headline.
    text: str           # Full scraped + cleaned article body.
    source: str         # Where it came from (e.g. the site's domain).
    published_at: datetime  # When it was published; used to place it on the time axis.
    url: Optional[str] = None  # Canonical link; optional because some sources lack one.
    # Lineage: the Bronze ingestion run this article originated from, carried
    # through from the RawArticle.
    load_id: Optional[str] = None

    def __post_init__(self):
        """Reject half-formed articles at construction time.

        Every required field must be present and non-blank, and published_at must
        be a real datetime - so no downstream step has to defend against an empty
        title, missing body, or a bad date.
        """
        if not self.title or not self.title.strip():
            raise ValueError("Article title cannot be empty")
        if not self.text or not self.text.strip():
            raise ValueError("Article text cannot be empty")
        if not self.source or not self.source.strip():
            raise ValueError("Article source cannot be empty")
        if not isinstance(self.published_at, datetime):
            raise ValueError("Article published_at must be a datetime")
