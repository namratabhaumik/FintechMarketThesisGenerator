"""Supabase-backed store for untagged fintech articles (Gold side-table)."""

import logging
from typing import List

from supabase import Client

from core.interfaces.untagged_repository import IUntaggedRepository
from core.models.article import Article

logger = logging.getLogger(__name__)

# Gold-layer side-table: fintech articles that matched no theme.
TABLE = "untagged_articles"


class SupabaseUntaggedRepository(IUntaggedRepository):
    """Records theme-less fintech articles in a Supabase `untagged_articles` table.

    Medallion role: Gold (side-table). During trend rollup, a fintech article
    that matched none of the known themes is parked here with its full text, so
    gaps in the theme taxonomy can be spotted and fixed from real content.

    Deduped by the UNIQUE(url) constraint via ignore_duplicates, so re-running
    Gold re-records nothing. Stores the full scraped text (the same text themes
    were matched against) so taxonomy gaps can be analysed from the real content.
    """

    def __init__(self, client: Client):
        # Live Supabase connection used for every query below.
        self._client = client

    def save(self, articles: List[Article]) -> int:
        # each untagged Article --> shape into a row keeping its full text
        # (so the miss can be inspected later) --> collect into `rows`.
        rows = [
            {
                "url": a.url,
                "title": a.title,
                "text": a.text,
                "published_at": a.published_at.isoformat(),
            }
            for a in articles
        ]
        # Nothing untagged this run --> return 0.
        if not rows:
            return 0
        # Upsert --> URL already recorded, ignore_duplicates skips it --> only
        # newly untagged articles come back in resp.data.
        resp = (
            self._client.table(TABLE)
            .upsert(rows, on_conflict="url", ignore_duplicates=True)
            .execute()
        )
        recorded = len(resp.data or [])
        logger.info(f"Gold: recorded {recorded} new untagged articles")
        return recorded
