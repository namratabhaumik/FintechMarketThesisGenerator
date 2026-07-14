"""Supabase-backed trend metrics store"""

import logging
from datetime import date, timedelta
from typing import List, Optional

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
        return [self._to_metric(row) for row in (resp.data or [])]

    def fetch_recent(self, window_weeks: Optional[int]) -> List[TrendMetric]:
        # Confidence only looks at the last `window_weeks` Gold weeks ending at
        # the latest present week, so scope the read to that range instead of
        # scanning all of Gold.
        # window_weeks None (whole-corpus retrieval) genuinely needs everything.
        # The scoped set yields the SAME covered_weeks/as_of a full read would:
        # the confidence window IS exactly [as_of - (window_weeks-1), as_of], and
        # _gold_confidence_inputs already discards anything outside it (& window),
        # so nothing it counts is left out.
        if window_weeks is None:
            return self.fetch_all()
        as_of = self._latest_week()
        if as_of is None:
            return []
        cutoff = as_of - timedelta(weeks=window_weeks - 1)
        resp = (
            self._client.table(TABLE)
            .select("*")
            .gte("week_start", cutoff.isoformat())
            .order("week_start", desc=True)
            .execute()
        )
        return [self._to_metric(row) for row in (resp.data or [])]

    def _latest_week(self) -> Optional[date]:
        # Cheap probe (LIMIT 1) for the newest week_start - the anchor the
        # confidence window is measured back from. None when Gold is empty.
        resp = (
            self._client.table(TABLE)
            .select("week_start")
            .order("week_start", desc=True)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return date.fromisoformat(rows[0]["week_start"]) if rows else None

    @staticmethod
    def _to_metric(row: dict) -> TrendMetric:
        # DB row --> TrendMetric, parsing week_start back into a date.
        return TrendMetric(
            week_start=date.fromisoformat(row["week_start"]),
            dimension=row["dimension"],
            category=row["category"],
            article_count=row["article_count"],
        )