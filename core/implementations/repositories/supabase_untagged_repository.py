"""Supabase-backed store for untagged fintech articles (Gold side-table)."""

import logging
from typing import List

from supabase import Client

from core.interfaces.untagged_repository import IUntaggedRepository
from core.models.article import Article

logger = logging.getLogger(__name__)

TABLE = "untagged_articles"


class SupabaseUntaggedRepository(IUntaggedRepository):
    """Records theme-less fintech articles in a Supabase `untagged_articles` table.

    Deduped by the UNIQUE(url) constraint via ignore_duplicates, so re-running
    Gold re-records nothing. Stores the full scraped text (the same text themes
    were matched against) so taxonomy gaps can be analysed from the real content.
    """

    def __init__(self, client: Client):
        self._client = client

    def save(self, articles: List[Article]) -> int:
        rows = [
            {
                "url": a.url,
                "title": a.title,
                "text": a.text,
                "published_at": a.published_at.isoformat(),
            }
            for a in articles
        ]
        if not rows:
            return 0
        resp = (
            self._client.table(TABLE)
            .upsert(rows, on_conflict="url", ignore_duplicates=True)
            .execute()
        )
        recorded = len(resp.data or [])
        logger.info(f"Gold: recorded {recorded} new untagged articles")
        return recorded
