"""Supabase-backed Silver dead-letter / quarantine store."""

import logging
from typing import List, Set

from supabase import Client

from core.interfaces.quarantine_repository import IQuarantineRepository
from core.models.quarantine_record import QuarantineRecord

logger = logging.getLogger(__name__)

TABLE = "quarantine"


class SupabaseQuarantineRepository(IQuarantineRepository):
    """Parks failed Silver records in a Supabase `quarantine` table.

    Deduped by the UNIQUE(url) constraint via ignore_duplicates, so a URL that
    is already parked is not re-added on a later run.
    """

    def __init__(self, client: Client):
        self._client = client

    def add(self, records: List[QuarantineRecord]) -> int:
        rows = [
            {
                "url": r.url,
                "reason": r.reason,
                "detail": r.detail,
                "title": r.title,
            }
            for r in records
        ]
        if not rows:
            return 0
        resp = (
            self._client.table(TABLE)
            .upsert(rows, on_conflict="url", ignore_duplicates=True)
            .execute()
        )
        added = len(resp.data or [])
        logger.info(f"Silver: quarantined {added} new records")
        return added

    def quarantined_urls(self) -> Set[str]:
        resp = self._client.table(TABLE).select("url").execute()
        rows: list = resp.data or []
        return {row["url"] for row in rows}
