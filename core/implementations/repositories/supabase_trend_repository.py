"""Supabase-backed trend metrics store"""

import logging
from datetime import date
from typing import List

from supabase import Client

from core.interfaces.trend_repository import ITrendRepository
from core.models.trend_metric import TrendMetric

logger = logging.getLogger(__name__)

TABLE = "trend_metrics"


class SupabaseTrendRepository(ITrendRepository):
    """Stores trend metrics in a Supabase `trend_metrics` table.

    Upserts on the (week_start, theme) primary key. It does NOT pass
    ignore_duplicates, so the default applies: a conflict UPDATEs the existing
    row. That means a recompute overwrites the previous count for a bucket
    """

    def __init__(self, client: Client):
        self._client = client

    def upsert(self, metrics: List[TrendMetric]) -> int:
        rows = [
            {
                "week_start": m.week_start.isoformat(),
                "theme": m.theme,
                "article_count": m.article_count,
            }
            for m in metrics
        ]
        if not rows:
            return 0
        self._client.table(TABLE).upsert(rows, on_conflict="week_start,theme").execute()
        logger.info(f"Gold: upserted {len(rows)} trend metrics")
        return len(rows)

    def fetch_all(self) -> List[TrendMetric]:
        resp = (
            self._client.table(TABLE)
            .select("*")
            .order("week_start", desc=True)
            .execute()
        )
        rows: list = resp.data or []
        return [
            TrendMetric(
                week_start=date.fromisoformat(row["week_start"]),
                theme=row["theme"],
                article_count=row["article_count"],
            )
            for row in rows
        ]