# core/utils.py
import json
import logging

logger = logging.getLogger(__name__)


def normalize_articles(articles):
    """Normalize any article list to have consistent keys."""
    normalized = []
    for a in articles:
        title = a.get("title") or "Untitled"
        text = (
            a.get("text")
            or a.get("content")
            or a.get("description")
            or a.get("summary")
            or ""
        )
        source = a.get("source") or a.get("link") or "unknown"
        url = a.get("link") or a.get("url") or None

        normalized.append({
            "title": title.strip(),
            "text": text.strip(),
            "source": source.strip() if isinstance(source, str) else "unknown",
            "url": url
        })
    logger.info(f"✅ Normalized {len(normalized)} articles.")
    return normalized


def load_sample_articles(path="data/sample_articles.json"):
    """Load articles from local JSON file."""
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample data not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
