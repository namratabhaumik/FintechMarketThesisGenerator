"""Tag base rate model"""

from dataclasses import dataclass


@dataclass
class TagBaseRate:
    """How common one tag category is across the whole fintech corpus.

    The denominator behind lift-ranked tags. A thesis ranks the tags its
    retrieved articles carry, and a raw count answers "which tag is most common
    here?" - but a wide pool is a large slice of the corpus, so the most common
    tag there is usually just the most common tag everywhere. Dividing the pool's
    rate by this corpus-wide rate answers the question actually being displayed:
    "which tag is over-represented in THIS query's evidence?".

    Same (dimension, category) vocabulary as TrendMetric, but aggregated over the
    whole corpus instead of bucketed by week. Recomputed from scratch each Gold
    run, so it tracks the corpus as it grows.
    """

    dimension: str        # Which tag dimension: "theme", "risk", or "signal".
    category: str         # The specific label within that dimension.
    article_count: int    # Fintech articles carrying this category, corpus-wide.
    total_articles: int   # Fintech articles considered - the shared denominator.

    def __post_init__(self):
        if not self.dimension or not self.dimension.strip():
            raise ValueError("TagBaseRate dimension cannot be empty")
        if not self.category or not self.category.strip():
            raise ValueError("TagBaseRate category cannot be empty")
        if self.article_count < 0:
            raise ValueError("TagBaseRate article_count cannot be negative")
        if self.total_articles < 0:
            raise ValueError("TagBaseRate total_articles cannot be negative")
        # A category cannot appear in more articles than exist.
        if self.article_count > self.total_articles:
            raise ValueError(
                "TagBaseRate article_count cannot exceed total_articles "
                f"({self.article_count} > {self.total_articles})"
            )

    @property
    def rate(self) -> float:
        """Share of the corpus carrying this category, 0.0 when the corpus is empty."""
        if self.total_articles <= 0:
            return 0.0
        return self.article_count / self.total_articles
