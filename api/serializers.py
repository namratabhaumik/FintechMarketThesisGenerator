"""Serialization and deserialization helpers for job data.

Handles converting between Supabase JSON rows and Python domain objects
(StructuredThesis, Article, Document) so the job manager and _RowProxy
stay focused on storage and attribute mapping.
"""

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from api.schemas import JobStatus
from core.models.article import Article
from core.models.thesis import StructuredThesis


def rehydrate_thesis(raw: Any) -> Optional[StructuredThesis]:
    """Convert a stored JSON dict to a StructuredThesis, or None."""
    if raw and isinstance(raw, dict):
        return StructuredThesis(**raw)
    return None


def rehydrate_articles(raw: Any) -> List[Article]:
    """Convert stored JSON dicts to Article objects.

    `published_at` is stored as an ISO 8601 string and parsed back into a
    datetime, since Article requires a real datetime on that field.
    """
    if not raw:
        return []
    articles = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        data = dict(a)
        published = data.get("published_at")
        if isinstance(published, str):
            data["published_at"] = datetime.fromisoformat(published)
        articles.append(Article(**data))
    return articles


def rehydrate_docs(raw: Any) -> list:
    """Convert stored JSON dicts to LangChain Document objects."""
    if not raw:
        return []
    result = []
    for d in raw:
        if isinstance(d, dict) and "page_content" in d:
            result.append(
                Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
            )
        else:
            result.append(d)
    return result


def serialise_thesis(thesis: StructuredThesis) -> dict:
    """Convert a StructuredThesis to a JSON-safe dict."""
    return asdict(thesis)


def serialise_articles(articles: List[Article]) -> list:
    """Convert Article objects to JSON-safe dicts.

    `published_at` is a datetime, which is not JSON-serialisable, so it is
    emitted as an ISO 8601 string.
    """
    result = []
    for a in articles:
        data = asdict(a)
        if isinstance(data.get("published_at"), datetime):
            data["published_at"] = data["published_at"].isoformat()
        result.append(data)
    return result


def serialise_docs(docs: list) -> list:
    """Convert LangChain Documents to JSON-safe dicts."""
    result = []
    for d in docs:
        if hasattr(d, "page_content"):
            result.append({"page_content": d.page_content, "metadata": d.metadata})
        else:
            result.append(d)
    return result


def serialise_job_fields(**fields) -> Dict[str, Any]:
    """Serialise arbitrary job fields for Supabase storage."""
    payload: Dict[str, Any] = {}
    for key, value in fields.items():
        if key == "thesis" and value is not None:
            payload[key] = serialise_thesis(value)
        elif key == "articles":
            payload[key] = serialise_articles(value)
        elif key == "status" and isinstance(value, JobStatus):
            payload[key] = value.value
        elif key == "retrieved_docs":
            payload[key] = serialise_docs(value)
        else:
            payload[key] = value
    return payload
