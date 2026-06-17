"""Quarantine record model (Silver dead-letter)."""

from dataclasses import dataclass

# Reasons a record is quarantined.
SCRAPE_FAILED = "scrape_failed"
INVALID_ARTICLE = "invalid_article"


@dataclass
class QuarantineRecord:
    """A Bronze article that failed Silver enrichment.

    Holds only inspection fields - Bronze stays the source of truth, so a replay
    re-reads the article from articles_raw and re-attempts it.
    """

    url: str
    reason: str
    detail: str = ""
    title: str = ""

    def __post_init__(self):
        if not self.url or not self.url.strip():
            raise ValueError("QuarantineRecord url cannot be empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("QuarantineRecord reason cannot be empty")