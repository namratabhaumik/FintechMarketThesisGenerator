"""Supabase-backed raw article store"""

import logging
from datetime import datetime
from typing import List

from postgrest.types import CountMethod
from supabase import Client

from core.interfaces.article_repository import IArticleRepository
from core.models.raw_article import RawArticle

logger = logging.getLogger(__name__)

TABLE = "articles_raw"


class SupabaseArticleRepository(IArticleRepository):
    """Append-only raw article store backed by a Supabase `articles_raw` table.

    Dedup is handled by the table's UNIQUE(url) constraint: save() upserts with
    ignore_duplicates, so re-ingesting an already-stored URL is a no-op and
    only genuinely new rows are returned (and counted).
    """

    def __init__(self, client: Client):
        self._client = client

    def save(self, articles: List[RawArticle]) -> int:
        rows = [self._to_row(a) for a in articles]
        if not rows:
            return 0
        resp = (
            self._client.table(TABLE)
            .upsert(rows, on_conflict="url", ignore_duplicates=True)
            .execute()
        )
        inserted = len(resp.data or [])
        logger.info(
            f"Bronze: {inserted} new articles stored "
            f"({len(rows) - inserted} already present)"
        )
        return inserted

    def fetch_all(self) -> List[RawArticle]:
        resp = (
            self._client.table(TABLE)
            .select("*")
            .order("published_at", desc=True)
            .execute()
        )
        return [self._to_raw_article(row) for row in (resp.data or [])]

    def count(self) -> int:
        resp = (
            self._client.table(TABLE)
            .select("id", count=CountMethod.exact)
            .execute()
        )
        return resp.count or 0

    def _to_raw_article(self, row: dict) -> RawArticle:
        return RawArticle(
            title=row["title"],
            url=row["url"],
            published_at=datetime.fromisoformat(row["published_at"]),
            summary=row.get("summary", ""),
            source=row.get("source", ""),
            feed_name=row.get("feed_name", ""),
        )

    def _to_row(self, article: RawArticle) -> dict:
        return {
            "url": article.url,
            "feed_name": article.feed_name,
            "source": article.source,
            "title": article.title,
            "summary": article.summary,
            "published_at": article.published_at.isoformat(),
        }