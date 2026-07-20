"""Raw article model"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RawArticle:
    """This is the Bronze record: what the RSS feed gave us (title, summary,
    link, publish date), with no full-text scrape, no fintech classification
    and no embedding. Those happen later in the Silver layer, which reads
    these rows. `published_at` is required so every row sits on the time axis.
    """

    title: str          # Headline from the feed.
    url: str            # Link to the article; also the dedup/primary key in Bronze.
    published_at: datetime  # Feed <pubDate>; required so every row sits on the time axis.
    summary: str = ""   # The short RSS blurb (not the full article text).
    source: str = ""    # Domain the link points to.
    feed_name: str = ""  # Which configured feed this entry came from (provenance).
    # Lineage: the ingestion run that landed this row (set at save time). Carried
    # forward into Silver so derived rows trace back to their load.
    load_id: Optional[str] = None

    def __post_init__(self):
        # Guard the two fields nothing downstream can work without: a title to
        # show and a URL to dedup/scrape by. A non-datetime published_at is also
        # rejected so the time axis is never polluted with a bad value.
        if not self.title or not self.title.strip():
            raise ValueError("RawArticle title cannot be empty")
        if not self.url or not self.url.strip():
            raise ValueError("RawArticle url cannot be empty")
        if not isinstance(self.published_at, datetime):
            raise ValueError("RawArticle published_at must be a datetime")
