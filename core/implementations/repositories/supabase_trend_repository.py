"""Supabase-backed trend metrics store"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from supabase import Client

from core.implementations.repositories.paging import fetch_paged
from core.interfaces.trend_repository import ITrendRepository
from core.models.tag_base_rate import TagBaseRate
from core.models.trend_metric import TrendMetric

logger = logging.getLogger(__name__)

# Gold-layer table: one article count per (week, dimension, category) bucket.
TABLE = "trend_metrics"
# Gold-layer table: one corpus-wide share per (dimension, category).
BASE_RATE_TABLE = "tag_base_rates"


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
        # Lineage: one computed_at per recompute (one upsert call) stamps every
        # bucket this run writes. Gold aggregates across many ingestion loads, so
        # a per-load id can't map to a bucket; the useful question is "which 
        # recompute produced this current count".
        computed_at = datetime.now(timezone.utc).isoformat()
        # each TrendMetric --> shape into a row keyed by week + dimension +
        # category with its article count --> collect into `rows`. week_start is
        # a date, so it is serialized to an ISO string.
        rows = [
            {
                "week_start": m.week_start.isoformat(),
                "dimension": m.dimension,
                "category": m.category,
                "article_count": m.article_count,
                "computed_at": computed_at,
                "load_ids": m.load_ids,
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
        # Paged: one row per (week, dimension, category), so Gold grows this
        # table fastest of all - a truncated read would silently drop the oldest
        # weeks out of the trend history. Sorting on the full upsert key
        # (week_start, dimension, category) makes it unique, so paging is stable.
        rows = fetch_paged(
            lambda: self._client.table(TABLE)
            .select("*")
            .order("week_start", desc=True)
            .order("dimension")
            .order("category")
        )
        return [self._to_metric(row) for row in rows]

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
        # Paged like fetch_all. The window bounds the rows but does not cap them:
        # a wide window over a taxonomy with many categories still multiplies out
        # past the row limit, and confidence is computed from what comes back.
        rows = fetch_paged(
            lambda: self._client.table(TABLE)
            .select("*")
            .gte("week_start", cutoff.isoformat())
            .order("week_start", desc=True)
            .order("dimension")
            .order("category")
        )
        return [self._to_metric(row) for row in rows]

    def upsert_base_rates(self, rates: List[TagBaseRate]) -> int:
        # One computed_at per recompute, same lineage story as upsert().
        computed_at = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "dimension": r.dimension,
                "category": r.category,
                "article_count": r.article_count,
                "total_articles": r.total_articles,
                "computed_at": computed_at,
            }
            for r in rates
        ]
        if not rows:
            return 0
        self._client.table(BASE_RATE_TABLE).upsert(
            rows, on_conflict="dimension,category"
        ).execute()
        # A category that left the corpus (taxonomy edit, or its last article
        # dropped) would otherwise keep its old row and stay a live lift
        # denominator. Upsert cannot remove it, so clear anything this recompute
        # did not just stamp.
        self._client.table(BASE_RATE_TABLE).delete().neq(
            "computed_at", computed_at
        ).execute()
        logger.info(f"Gold: upserted {len(rows)} tag base rates")
        return len(rows)

    def fetch_base_rates(self) -> List[TagBaseRate]:
        # Small by construction: one row per (dimension, category), not per week.
        # Paged anyway - these are the lift denominators, and a silently short
        # read would inflate lift for whichever categories fell off the end.
        # (dimension, category) is the upsert key, so it orders uniquely.
        rows = fetch_paged(
            lambda: self._client.table(BASE_RATE_TABLE)
            .select("*")
            .order("dimension")
            .order("category")
        )
        return [self._to_base_rate(row) for row in rows]

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
        # Annotated because postgrest types .data as JSON?, which mypy will not
        # let us index; the same annotation is used at the other read sites.
        rows: list = resp.data or []
        return date.fromisoformat(rows[0]["week_start"]) if rows else None

    @staticmethod
    def _to_base_rate(row: dict) -> TagBaseRate:
        # DB row --> TagBaseRate. `rate` is derived, never stored.
        return TagBaseRate(
            dimension=row["dimension"],
            category=row["category"],
            article_count=row["article_count"],
            total_articles=row["total_articles"],
        )

    @staticmethod
    def _to_metric(row: dict) -> TrendMetric:
        # DB row --> TrendMetric, parsing week_start back into a date.
        return TrendMetric(
            week_start=date.fromisoformat(row["week_start"]),
            dimension=row["dimension"],
            category=row["category"],
            article_count=row["article_count"],
            load_ids=row.get("load_ids") or [],
        )