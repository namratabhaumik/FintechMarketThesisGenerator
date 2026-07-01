"""Supabase-backed trend metrics store"""

import logging
from datetime import date
from typing import List

from supabase import Client

from core.interfaces.trend_repository import ITrendRepository
from core.models.trend_metric import TrendMetric

logger = logging.getLogger(__name__)

# Gold-layer table: one article count per (week, dimension, category) bucket.
TABLE = "trend_metrics"


class SupabaseTrendRepository(ITrendRepository):
    """Stores trend metrics in a Supabase `trend_metrics` table.

    Medallion role: Gold. This is the aggregated output - per-category weekly
    counts rolled up from the Silver tags, across all three dimensions (theme /
    risk / signal) - that the app charts as trends.

    Upserts on the (week_start, dimension, category) primary key. It does NOT
    pass ignore_duplicates, so the default applies: a conflict UPDATEs the
    existing row. That means a recompute overwrites the previous count for a
    bucket.
    """

    def __init__(self, client: Client):
        # Live Supabase connection used for every query below.
        self._client = client

    def upsert(self, metrics: List[TrendMetric]) -> int:
        # each TrendMetric --> shape into a row keyed by week + dimension +
        # category with its article count --> collect into `rows`. week_start is
        # a date, so it is serialized to an ISO string.
        rows = [
            {
                "week_start": m.week_start.isoformat(),
                "dimension": m.dimension,
                "category": m.category,
                "article_count": m.article_count,
            }
            for m in metrics
        ]
        # Nothing to write --> return 0.
        if not rows:
            return 0
        # Upsert on the (week_start, dimension, category) key --> existing bucket
        # gets its count OVERWRITTEN (no ignore_duplicates), so recompute
        # refreshes numbers rather than skipping them.
        self._client.table(TABLE).upsert(
            rows, on_conflict="week_start,dimension,category"
        ).execute()
        logger.info(f"Gold: upserted {len(rows)} trend metrics")
        return len(rows)

    def fetch_all(self) -> List[TrendMetric]:
        # Read every bucket, newest week first --> rebuild each as a TrendMetric.
        resp = (
            self._client.table(TABLE)
            .select("*")
            .order("week_start", desc=True)
            .execute()
        )
        rows: list = resp.data or []
        # DB row --> TrendMetric, parsing week_start back into a date.
        return [
            TrendMetric(
                week_start=date.fromisoformat(row["week_start"]),
                dimension=row["dimension"],
                category=row["category"],
                article_count=row["article_count"],
            )
            for row in rows
        ]