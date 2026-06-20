"""Supabase-backed Silver verdict store"""

import logging
from typing import Dict, List, Set

from supabase import Client

from core.interfaces.silver_repository import ISilverRepository
from core.models.silver_record import SilverVerdict

logger = logging.getLogger(__name__)

# Silver-layer table: one verdict per article (is it fintech, which themes).
TABLE = "articles_silver"


class SupabaseSilverRepository(ISilverRepository):
    """Records Silver processing verdicts in a Supabase `articles_silver` table.

    Medallion role: Silver. For each article the classifier produces a verdict -
    whether it's fintech-relevant and which themes it carries - and that verdict
    is stored here, keyed by URL. Verdicts are frozen point-in-time facts.

    Deduped by the table's UNIQUE(url) constraint, so re-recording a URL is a
    no-op and the build can run idempotently.
    """

    def __init__(self, client: Client):
        # Live Supabase connection used for every query below.
        self._client = client

    def processed_urls(self) -> Set[str]:
        # Fetch just the URL column --> set of URLs already given a verdict, so
        # the build can skip them and stay idempotent.
        resp = self._client.table(TABLE).select("url").execute()
        rows: list = resp.data or []
        return {row["url"] for row in rows}

    def fintech_themes(self) -> Dict[str, List[str]]:
        # Pull url + themes, but only for rows flagged fintech_relevant=True.
        resp = (
            self._client.table(TABLE)
            .select("url, themes")
            .eq("fintech_relevant", True)
            .execute()
        )
        rows: list = resp.data or []
        # Build url --> themes-list map; null themes fall back to an empty list
        # so callers never have to handle None.
        return {row["url"]: (row.get("themes") or []) for row in rows}

    def fintech_tags(self) -> Dict[str, Dict[str, List[str]]]:
        # Pull all three tag dimensions for the accepted (fintech) rows.
        resp = (
            self._client.table(TABLE)
            .select("url, themes, risks, signals")
            .eq("fintech_relevant", True)
            .execute()
        )
        rows: list = resp.data or []
        # url --> {themes, risks, signals}; each null dimension falls back to []
        return {
            row["url"]: {
                "themes": row.get("themes") or [],
                "risks": row.get("risks") or [],
                "signals": row.get("signals") or [],
            }
            for row in rows
        }

    def record(self, verdicts: List[SilverVerdict]) -> int:
        # each SilverVerdict --> shape into a row holding the URL, the
        # relevance flag, and all three tag dimensions --> collect into `rows`.
        rows = [
            {
                "url": v.url,
                "fintech_relevant": v.fintech_relevant,
                "themes": v.themes,
                "risks": v.risks,
                "signals": v.signals,
            }
            for v in verdicts
        ]
        # No verdicts this run --> return 0.
        if not rows:
            return 0
        # Upsert --> URL already has a verdict, ignore_duplicates keeps the
        # original (verdicts are frozen) --> only new verdicts come back.
        resp = (
            self._client.table(TABLE)
            .upsert(rows, on_conflict="url", ignore_duplicates=True)
            .execute()
        )
        recorded = len(resp.data or [])
        logger.info(f"Silver: recorded {recorded} new verdicts")
        return recorded