"""Aggregate the fintech corpus into per-category weekly trends.

Reads the fintech (accepted) articles and their Silver tags (themes, risks and
signals, all assigned on the full scraped text), buckets them by the Monday of
their publish week, and stores per-(week, dimension, category) coverage counts.
Every tag dimension is accumulated the same way, so risks and signals become
historic trends just like themes. Articles that matched no theme are captured
for later taxonomy analysis. Recomputed from current data each run.

Gold reads only from Silver (the verdict store for tags, the article-content
store for publish dates), never from Bronze
"""

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, Tuple

from core.interfaces.article_content_repository import IArticleContentRepository
from core.interfaces.silver_repository import ISilverRepository
from core.interfaces.trend_repository import ITrendRepository
from core.interfaces.untagged_repository import IUntaggedRepository
from core.models.trend_metric import TrendMetric
from core.utils.data_quality import check_gold

logger = logging.getLogger(__name__)


class GoldService:
    """Builds per-category weekly trend metrics (Gold) from the fintech corpus,
    across all three Silver tag dimensions (theme / risk / signal)."""

    def __init__(
        self,
        content_repository: IArticleContentRepository,
        silver_repository: ISilverRepository,
        trend_repository: ITrendRepository,
        untagged_repository: IUntaggedRepository,
    ):
        self._content_repository = content_repository
        self._silver_repository = silver_repository
        self._trend_repository = trend_repository
        self._untagged_repository = untagged_repository

    def build(self) -> int:
        """Recompute and store all trend metrics.

        Returns:
            The number of (week, dimension, category) metric rows written.

        read accepted-article tags from Silver --> walk all stored article
        content --> for each article, add 1 to every (week, dimension, category)
        tally across all three dimensions --> turn tallies into TrendMetric rows
        --> save metrics + untagged.
        """
        # URL --> {themes, risks, signals}, for fintech-accepted articles only
        # (Silver's verdict view of "what is relevant and what it carries"). A
        # URL missing here was rejected or not yet decided, so Gold ignores it.
        tags_by_url = self._silver_repository.fintech_tags()
        # Single in-memory pass over the Silver content store (already just the
        # accepted subset), joined to the verdict tags by URL. No Bronze read:
        # Gold is a pure Silver -> Gold transform. Fine at this scale; if volume
        # ever makes the full fetch hurt, push this join into a DB view, not a
        # growing .in_() list.
        # buckets: (week, dimension, category) -> count, accumulated across all
        # three tag dimensions in one pass.
        buckets: Dict[Tuple[date, str, str], int] = defaultdict(int)
        untagged = []
        fintech = 0
        for article in self._content_repository.fetch_all():
            tags = tags_by_url.get(article.url or "")
            if tags is None:
                continue  # no fintech verdict for this URL
            fintech += 1
            week = self._week_start(article.published_at)
            # Theme-taxonomy gaps are still captured separately (untagged), but
            # the article's risks/signals are still tallied below regardless.
            if not tags["themes"]:
                untagged.append(article)
            # Tally every category the article carries, in every dimension.
            for dimension in ("theme", "risk", "signal"):
                for category in tags[f"{dimension}s"]:
                    buckets[(week, dimension, category)] += 1
        logger.info(f"Gold: aggregating {fintech} fintech articles")

        metrics = [
            TrendMetric(
                week_start=week,
                dimension=dimension,
                category=category,
                article_count=count,
            )
            for (week, dimension, category), count in sorted(buckets.items())
        ]
        check_gold(
            fintech_verdicts=len(tags_by_url),
            fintech_with_content=fintech,
            metrics=metrics,
        )
        self._trend_repository.upsert(metrics)
        self._untagged_repository.save(untagged)
        logger.info(
            f"Gold: wrote {len(metrics)} (week, dimension, category) metrics "
            f"across {len({w for w, _, _ in buckets})} weeks; "
            f"{len(untagged)} untagged"
        )
        return len(metrics)

    @staticmethod
    def _week_start(published_at: datetime) -> date:
        """Monday of the article's publish week."""
        d = published_at.date()
        return d - timedelta(days=d.weekday())
