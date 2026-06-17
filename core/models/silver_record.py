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

    url: str
    fintech_relevant: bool
    themes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.url or not self.url.strip():
            raise ValueError("SilverVerdict url cannot be empty")