"""Aggregate the fintech corpus into per-theme weekly trends.

Reads the fintech (accepted) articles and their themes (assigned by Silver on
the full scraped text), buckets them by the Monday of their publish week, and
stores per-(week, theme) coverage counts. Articles that matched no theme are
captured for later taxonomy analysis. Recomputed from current data each run.
"""

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, Tuple

from core.interfaces.article_repository import IArticleRepository
from core.interfaces.silver_repository import ISilverRepository
from core.interfaces.trend_repository import ITrendRepository
from core.interfaces.untagged_repository import IUntaggedRepository
from core.models.trend_metric import TrendMetric

logger = logging.getLogger(__name__)


class GoldService:
    """Builds per-theme weekly trend metrics (Gold) from the fintech corpus."""

    def __init__(
        self,
        article_repository: IArticleRepository,
        silver_repository: ISilverRepository,
        trend_repository: ITrendRepository,
        untagged_repository: IUntaggedRepository,
    ):
        self._article_repository = article_repository
        self._silver_repository = silver_repository
        self._trend_repository = trend_repository
        self._untagged_repository = untagged_repository

    def build(self) -> int:
        """Recompute and store all trend metrics.

        Returns:
            The number of (week, theme) metric rows written.
        """
        themes_by_url = self._silver_repository.fintech_themes()
        # In-memory filter: pull all Bronze rows and keep the fintech URLs. Fine
        # at this scale. If volume ever makes the full fetch hurt, push this join 
        # into a DB view, not a growing .in_() list.
        articles = [
            a for a in self._article_repository.fetch_all() if a.url in themes_by_url
        ]
        logger.info(f"Gold: aggregating {len(articles)} fintech articles")

        buckets: Dict[Tuple[date, str], int] = defaultdict(int)
        untagged = []
        for article in articles:
            themes = themes_by_url[article.url]
            if not themes:
                # Accepted but matched no theme: capture for later taxonomy
                # analysis instead of dropping it.
                untagged.append(article)
                continue
            week = self._week_start(article.published_at)
            for theme in themes:
                buckets[(week, theme)] += 1

        metrics = [
            TrendMetric(week_start=week, theme=theme, article_count=count)
            for (week, theme), count in sorted(buckets.items())
        ]
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
