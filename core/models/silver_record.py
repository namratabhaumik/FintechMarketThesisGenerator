"""Silver verdict model"""

from dataclasses import dataclass


@dataclass
class SilverVerdict:
    """The outcome of processing one Bronze article in the Silver layer.

    Recorded for every processed URL - accepted or rejected - so a later run
    never re-classifies an article it has already decided on. `fintech_relevant`
    is False for articles the classifier rejected (which are therefore never
    scraped or embedded).
    """

    url: str
    fintech_relevant: bool

    def __post_init__(self):
        if not self.url or not self.url.strip():
            raise ValueError("SilverVerdict url cannot be empty")