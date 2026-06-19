"""Quarantine record model (Silver dead-letter)."""

from dataclasses import dataclass

# The two reasons Silver parks an article instead of embedding it:
SCRAPE_FAILED = "scrape_failed"      # the scraper returned no usable text.
INVALID_ARTICLE = "invalid_article"  # the scraped text failed Article validation.


@dataclass
class QuarantineRecord:
    """A Bronze article that failed Silver enrichment.

    Holds only inspection fields - Bronze stays the source of truth, so a replay
    re-reads the article from articles_raw and re-attempts it.
    """

    url: str            # Which Bronze article failed; the key used to replay it.
    reason: str         # One of the SCRAPE_FAILED / INVALID_ARTICLE constants above.
    detail: str = ""    # Free-text note (e.g. the validation error) for a human.
    title: str = ""     # Carried along so the row is readable without a Bronze join.

    def __post_init__(self):
        # A quarantine row is useless without the URL to replay and a reason to
        # explain why it landed here, so both are required.
        if not self.url or not self.url.strip():
            raise ValueError("QuarantineRecord url cannot be empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("QuarantineRecord reason cannot be empty")