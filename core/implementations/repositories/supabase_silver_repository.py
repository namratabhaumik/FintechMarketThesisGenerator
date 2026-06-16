"""Supabase-backed Silver verdict store"""

import logging
from typing import List, Set

from supabase import Client

from core.interfaces.silver_repository import ISilverRepository
from core.models.silver_record import SilverVerdict

logger = logging.getLogger(__name__)

TABLE = "articles_silver"


class SupabaseSilverRepository(ISilverRepository):
    """Records Silver processing verdicts in a Supabase `articles_silver` table.

    Deduped by the table's UNIQUE(url) constraint, so re-recording a URL is a
    no-op and the build can run idempotently.
    """

    def __init__(self, client: Client):
        self._client = client

    def processed_urls(self) -> Set[str]:
        resp = self._client.table(TABLE).select("url").execute()
        rows: list = resp.data or []
        return {row["url"] for row in rows}

    def record(self, verdicts: List[SilverVerdict]) -> int:
        rows = [
            {"url": v.url, "fintech_relevant": v.fintech_relevant} for v in verdicts
        ]
        if not rows:
            return 0
        resp = (
            self._client.table(TABLE)
            .upsert(rows, on_conflict="url", ignore_duplicates=True)
            .execute()
        )
        recorded = len(resp.data or [])
        logger.info(f"Silver: recorded {recorded} new verdicts")
        return recorded