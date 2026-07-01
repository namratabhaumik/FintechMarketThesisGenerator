"""Silver verdict model"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SilverVerdict:
    """The outcome of processing one Bronze article in the Silver layer.

    Recorded for every processed URL - accepted or rejected - so a later run
    never re-classifies an article it has already decided on. `fintech_relevant`
    is False for articles the classifier rejected (which are therefore never
    scraped or embedded).

    The three tag lists - `themes`, `risks`, `signals` - are all matched
    deterministically on the full scraped text at Silver time (same keyword
    scoring, three different category maps). They are empty for rejected
    articles, or for accepted ones that matched nothing in a given dimension.
    Tagging all three here (rather than later, on a llm generated thesis summary) is what lets
    Gold accumulate them into historic trends and lets the thesis stay grounded
    in what articles actually reported.
    """

    url: str            # The Bronze article this verdict is about; the dedup key.
    fintech_relevant: bool  # True = accepted (scraped/embedded); False = rejected.
    # Fintech sub-sector themes matched on the full text (e.g. "Digital Payments").
    themes: List[str] = field(default_factory=list)
    # Risk categories explicitly present in the text (e.g. "Regulatory Risk").
    risks: List[str] = field(default_factory=list)
    # Investment-signal categories present in the text (e.g. "Payment Infrastructure").
    signals: List[str] = field(default_factory=list)

    def __post_init__(self):
        # The URL is the identity of the verdict, so it must be present.
        if not self.url or not self.url.strip():
            raise ValueError("SilverVerdict url cannot be empty")