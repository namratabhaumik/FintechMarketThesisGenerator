"""Article-to-Document conversion shared by Silver and the re-embed script."""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

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
