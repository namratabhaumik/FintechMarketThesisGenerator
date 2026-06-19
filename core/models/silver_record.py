"""Silver verdict model"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SilverVerdict:
    """The outcome of processing one Bronze article in the Silver layer.

    Recorded for every processed URL - accepted or rejected - so a later run
    never re-classifies an article it has already decided on. `fintech_relevant`
    is False for articles the classifier rejected (which are therefore never
    scraped or embedded). `themes` are the fintech themes matched on the full
    scraped text (empty for rejected articles, or for accepted ones matching no
    theme); the Gold layer aggregates trends from them.
    """

    url: str            # The Bronze article this verdict is about; the dedup key.
    fintech_relevant: bool  # True = accepted (scraped/embedded); False = rejected.
    # Fintech themes matched on the full text. Empty when rejected, or when
    # accepted but matching no theme. Gold rolls these up into weekly trends.
    themes: List[str] = field(default_factory=list)

    def __post_init__(self):
        # The URL is the identity of the verdict, so it must be present.
        if not self.url or not self.url.strip():
            raise ValueError("SilverVerdict url cannot be empty")