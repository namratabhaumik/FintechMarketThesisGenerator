"""Article ingestion service."""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from core.interfaces.article_source import IArticleSource
from core.models.article import Article

logger = logging.getLogger(__name__)


def article_to_document(
    article: Article,
    themes: Optional[List[str]] = None,
    risks: Optional[List[str]] = None,
    signals: Optional[List[str]] = None,
) -> Document:
    """Convert an Article to a LangChain Document.

    Shared by the request-time pipeline and the Silver layer so both produce
    identical metadata. `published_at` is included as an ISO string so the
    retrieval layer can filter/rank documents on the time axis.

    The Silver layer also passes the article's deterministic tags (themes /
    risks / signals). They ride in the chunk metadata so retrieval can filter by
    theme and the thesis can read grounded tags straight off the retrieved docs -
    no second DB lookup. Callers without tags (left as None) omit those keys.
    """
    # page_content is what gets embedded: title first, then the body, separated
    # by a blank line. metadata rides alongside the vector so retrieval can show
    # the source/title/url and rank or filter on published_at (kept as an ISO
    # string because the vector store stores plain JSON-friendly values).
    metadata: Dict[str, Any] = {
        "source": article.source,
        "title": article.title,
        "url": article.url or "",
        "published_at": article.published_at.isoformat(),
    }
    # Attach each tag dimension only when the caller supplied it, so the
    # request-time path (no tags) keeps its original, smaller metadata shape.
    if themes is not None:
        metadata["themes"] = themes
    if risks is not None:
        metadata["risks"] = risks
    if signals is not None:
        metadata["signals"] = signals
    return Document(page_content=f"{article.title}\n\n{article.text}", metadata=metadata)


class ArticleIngestionService:
    """Service for ingesting articles.

    Single Responsibility: Article fetching and normalization.
    Implements Dependency Inversion: Depends on IArticleSource abstraction.
    """

    def __init__(self, article_source: IArticleSource):
        """Initialize with article source.

        Args:
            article_source: Injected article source implementation.
        """
        self._article_source = article_source

    def fetch_articles(self, query: str, limit: int) -> List[Article]:
        """Fetch articles from configured source.

        Args:
            query: Search query.
            limit: Maximum number of articles to fetch.

        Returns:
            List of normalized Article objects.
        """
        logger.info(f"Fetching articles from {self._article_source.get_source_name()}")
        articles = self._article_source.fetch_articles(query, limit)

        normalized = self._normalize_articles(articles)
        logger.info(f"Ingested {len(normalized)} articles")
        return normalized

    def _normalize_articles(self, articles: List[Article]) -> List[Article]:
        """Validate and normalize articles.

        Args:
            articles: List of articles to normalize.

        Returns:
            List of valid normalized articles.
        """
        # normalized: the articles we keep. The real validation already happened
        # when each Article was constructed (its __post_init__ raises on bad
        # data), so by here they are sound; this loop is the seam where a future
        # extra check could drop a bad one and log it rather than crash.
        normalized = []

        for article in articles:
            try:
                # Article dataclass validates on creation
                normalized.append(article)
            except ValueError as e:
                logger.warning(f"Invalid article skipped: {e}")
                continue

        return normalized

    def convert_to_documents(self, articles: List[Article]) -> List[Document]:
        """Convert articles to LangChain documents.

        Args:
            articles: List of Article objects.

        Returns:
            List of LangChain Document objects.
        """
        docs = [article_to_document(article) for article in articles]
        logger.info(f"Converted {len(docs)} articles to documents")
        return docs
