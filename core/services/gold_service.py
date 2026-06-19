"""Aggregate the fintech corpus into per-theme weekly trends.

Reads the fintech (accepted) articles and their themes (assigned by Silver on
the full scraped text), buckets them by the Monday of their publish week, and
stores per-(week, theme) coverage counts. Articles that matched no theme are
captured for later taxonomy analysis. Recomputed from current data each run.

Gold reads only from Silver (the verdict store for themes, the article-content
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
    """Builds per-theme weekly trend metrics (Gold) from the fintech corpus."""

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
            The number of (week, theme) metric rows written.

        read accepted-article themes from Silver --> walk all stored
        article content --> for each article, add 1 to every (week, theme) tally
        --> turn tallies into TrendMetric rows --> save metrics + untagged.
        """
        # URL --> its themes, for fintech-accepted articles only (Silver's
        # verdict view of "what is relevant and what it is about"). A URL missing
        # here was rejected or not yet decided, so Gold ignores it.
        themes_by_url = self._silver_repository.fintech_themes()
        # Single in-memory pass over the Silver content store (already just the
        # accepted subset), joined to the verdict themes by URL. No Bronze read:
        # Gold is a pure Silver -> Gold transform. Fine at this scale; if volume
        # ever makes the full fetch hurt, push this join into a DB view, not a
        # growing .in_() list.
        buckets: Dict[Tuple[date, str], int] = defaultdict(int)
        untagged = []
        fintech = 0
        for article in self._content_repository.fetch_all():
            themes = themes_by_url.get(article.url or "")
            if themes is None:
                continue  # no fintech verdict for this URL
            fintech += 1
            if not themes:
                # Accepted but matched no theme: capture for later taxonomy
                # analysis instead of dropping it.
                untagged.append(article)
                continue
            week = self._week_start(article.published_at)
            for theme in themes:
                buckets[(week, theme)] += 1
        logger.info(f"Gold: aggregating {fintech} fintech articles")

        metrics = [
            TrendMetric(week_start=week, theme=theme, article_count=count)
            for (week, theme), count in sorted(buckets.items())
        ]
        check_gold(
            fintech_verdicts=len(themes_by_url),
            fintech_with_content=fintech,
            metrics=metrics,
        )
        self._trend_repository.upsert(metrics)
        self._untagged_repository.save(untagged)
        logger.info(
            f"Gold: wrote {len(metrics)} (week, theme) metrics "
            f"across {len({w for w, _ in buckets})} weeks; "
            f"{len(untagged)} untagged"
        )
        return len(metrics)

    @staticmethod
    def _week_start(published_at: datetime) -> date:
        """Monday of the article's publish week."""
        d = published_at.date()
        return d - timedelta(days=d.weekday())
