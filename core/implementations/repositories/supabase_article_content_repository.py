"""Supabase-backed validated article-content store"""

import logging
from datetime import datetime
from typing import List

from supabase import Client

from core.interfaces.article_content_repository import IArticleContentRepository
from core.models.article import Article

logger = logging.getLogger(__name__)

TABLE = "article_content"


class SupabaseArticleContentRepository(IArticleContentRepository):
    """Stores validated articles in a Supabase `article_content` table.

    Deduped by the UNIQUE(url) constraint via ignore_duplicates, so the text is
    written once per URL and re-saving is a no-op.
    """

    def __init__(self, client: Client):
        self._client = client

    def save(self, articles: List[Article]) -> int:
        rows = [
            {
                "url": a.url,
                "title": a.title,
                "text": a.text,
                "source": a.source,
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
        saved = len(resp.data or [])
        logger.info(f"Silver: persisted {saved} new article-content records")
        return saved

    def fetch_all(self) -> List[Article]:
        resp = self._client.table(TABLE).select("*").execute()
        rows: list = resp.data or []
        return [
            Article(
                title=row["title"],
                text=row["text"],
                source=row["source"],
                url=row["url"],
                published_at=datetime.fromisoformat(row["published_at"]),
            )
            for row in rows
        ]