"""Raw article model"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class RawArticle:
    """This is the Bronze record: what the RSS feed gave us (title, summary,
    link, publish date), with no full-text scrape, no fintech classification
    and no embedding. Those happen later in the Silver layer, which reads
    these rows. `published_at` is required so every row sits on the time axis.
    """

    title: str
    url: str
    published_at: datetime
    summary: str = ""
    source: str = ""
    feed_name: str = ""

    def __post_init__(self):
        if not self.title or not self.title.strip():
            raise ValueError("RawArticle title cannot be empty")
        if not self.url or not self.url.strip():
            raise ValueError("RawArticle url cannot be empty")
        if not isinstance(self.published_at, datetime):
            raise ValueError("RawArticle published_at must be a datetime")
