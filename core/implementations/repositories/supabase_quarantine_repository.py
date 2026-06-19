"""Supabase-backed Silver dead-letter / quarantine store."""

import logging
from typing import List, Set

from supabase import Client

from core.interfaces.quarantine_repository import IQuarantineRepository
from core.models.quarantine_record import QuarantineRecord

logger = logging.getLogger(__name__)

# Silver-layer dead-letter table: articles that failed processing are parked here.
TABLE = "quarantine"


class SupabaseQuarantineRepository(IQuarantineRepository):
    """Parks failed Silver records in a Supabase `quarantine` table.

    Medallion role: Silver (the failure path). When an article can't be cleaned
    or validated, instead of dropping it silently we record why here, keyed by
    URL, so the rest of the pipeline can skip it and the failure stays auditable.

    Deduped by the UNIQUE(url) constraint via ignore_duplicates, so a URL that
    is already parked is not re-added on a later run.
    """

    def __init__(self, client: Client):
        # Live Supabase connection used for every query below.
        self._client = client

    def add(self, records: List[QuarantineRecord]) -> int:
        # each QuarantineRecord --> shape into a row capturing the URL +
        # the failure reason/detail/title --> collect into `rows`.
        rows = [
            {
                "url": r.url,
                "reason": r.reason,
                "detail": r.detail,
                "title": r.title,
            }
            for r in records
        ]
        # Nothing failed this run --> return 0.
        if not rows:
            return 0
        # Upsert --> URL already quarantined, ignore_duplicates skips it --> only
        # newly parked URLs come back in resp.data.
        resp = (
            self._client.table(TABLE)
            .upsert(rows, on_conflict="url", ignore_duplicates=True)
            .execute()
        )
        added = len(resp.data or [])
        logger.info(f"Silver: quarantined {added} new records")
        return added

    def quarantined_urls(self) -> Set[str]:
        # Fetch just the URL column --> build a set so callers can quickly test
        # "is this URL already quarantined?" and skip it.
        resp = self._client.table(TABLE).select("url").execute()
        rows: list = resp.data or []
        return {row["url"] for row in rows}
