"""Supabase-backed raw article store"""

import logging
from datetime import datetime
from typing import List

from postgrest.types import CountMethod
from supabase import Client

from core.interfaces.article_repository import IArticleRepository
from core.models.raw_article import RawArticle

logger = logging.getLogger(__name__)

# Bronze-layer table: raw RSS entries landed verbatim, one row per URL.
TABLE = "articles_raw"


class SupabaseArticleRepository(IArticleRepository):
    """Append-only raw article store backed by a Supabase `articles_raw` table.

    Medallion role: Bronze. This is the landing zone where freshly scraped RSS
    entries are stored exactly as fetched, before any cleaning or classifying.

    Dedup is handled by the table's UNIQUE(url) constraint: save() upserts with
    ignore_duplicates, so re-ingesting an already-stored URL is a no-op and
    only genuinely new rows are returned (and counted).
    """

    def __init__(self, client: Client):
        # Live Supabase connection used for every query below.
        self._client = client

    def save(self, articles: List[RawArticle]) -> int:
        # each RawArticle --> _to_row() shapes it into a DB dict --> rows.
        rows = [self._to_row(a) for a in articles]
        # Empty batch --> nothing to write --> return 0.
        if not rows:
            return 0
        # Upsert --> URL already in Bronze, ignore_duplicates skips it (no
        # overwrite) --> resp.data holds only the genuinely new rows.
        resp = (
            self._client.table(TABLE)
            .upsert(rows, on_conflict="url", ignore_duplicates=True)
            .execute()
        )
        # Newly stored rows; the rest of the batch were already present.
        inserted = len(resp.data or [])
        logger.info(
            f"Bronze: {inserted} new articles stored "
            f"({len(rows) - inserted} already present)"
        )
        return inserted

    def fetch_all(self) -> List[RawArticle]:
        # Read every Bronze row, newest first --> rebuild each as a RawArticle.
        resp = (
            self._client.table(TABLE)
            .select("*")
            .order("published_at", desc=True)
            .execute()
        )
        return [self._to_raw_article(row) for row in (resp.data or [])]

    def count(self) -> int:
        # Ask Postgres for an exact row count without pulling the rows.
        resp = (
            self._client.table(TABLE)
            .select("id", count=CountMethod.exact)
            .execute()
        )
        return resp.count or 0

    def _to_raw_article(self, row: dict) -> RawArticle:
        # DB row --> RawArticle model. .get() with a default guards rows where an
        # optional field is missing/null; published_at is parsed back to datetime.
        return RawArticle(
            title=row["title"],
            url=row["url"],
            published_at=datetime.fromisoformat(row["published_at"]),
            summary=row.get("summary", ""),
            source=row.get("source", ""),
            feed_name=row.get("feed_name", ""),
        )

    def _to_row(self, article: RawArticle) -> dict:
        # RawArticle model --> DB dict matching the Bronze columns; datetime is
        # serialized to an ISO string for storage.
        return {
            "url": article.url,
            "feed_name": article.feed_name,
            "source": article.source,
            "title": article.title,
            "summary": article.summary,
            "published_at": article.published_at.isoformat(),
        }