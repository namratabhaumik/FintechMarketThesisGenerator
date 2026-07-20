"""Supabase-backed validated article-content store"""

import logging
from datetime import datetime
from typing import List

from supabase import Client

from core.interfaces.article_content_repository import IArticleContentRepository
from core.models.article import Article

logger = logging.getLogger(__name__)

# Silver-layer table holding each article's validated full text.
TABLE = "article_content"


class SupabaseArticleContentRepository(IArticleContentRepository):
    """Stores validated articles in a Supabase `article_content` table.

    Medallion role: Silver. After an article is scraped and passes validation,
    its cleaned body lands here so later layers can read real content without
    re-scraping. One row per article, keyed by URL.

    Deduped by the UNIQUE(url) constraint via ignore_duplicates, so the text is
    written once per URL and re-saving is a no-op.
    """

    def __init__(self, client: Client):
        # Live Supabase connection used for every query below.
        self._client = client

    def save(self, articles: List[Article]) -> int:
        # take Article models --> shape each into a DB row dict (only the
        # columns this table holds) --> collect into `rows`.
        rows = [
            {
                "url": a.url,
                "title": a.title,
                "text": a.text,
                "source": a.source,
                # Supabase wants a string, so the datetime is serialized to ISO.
                "published_at": a.published_at.isoformat(),
                "load_id": a.load_id,
            }
            for a in articles
        ]
        # Nothing to save --> return 0 and skip the network call.
        if not rows:
            return 0
        # Send rows up --> on a URL already present, ignore_duplicates makes it a
        # no-op (no overwrite) --> only brand-new rows come back in resp.data.
        resp = (
            self._client.table(TABLE)
            .upsert(rows, on_conflict="url", ignore_duplicates=True)
            .execute()
        )
        # Count only the rows actually inserted this run.
        saved = len(resp.data or [])
        logger.info(f"Silver: persisted {saved} new article-content records")
        return saved

    def fetch_all(self) -> List[Article]:
        # Pull every stored row, then turn each DB row back into an Article.
        resp = self._client.table(TABLE).select("*").execute()
        rows: list = resp.data or []
        # for each row --> rebuild an Article, parsing the ISO string back
        # into a datetime.
        return [
            Article(
                title=row["title"],
                text=row["text"],
                source=row["source"],
                url=row["url"],
                published_at=datetime.fromisoformat(row["published_at"]),
                load_id=row.get("load_id"),
            )
            for row in rows
        ]