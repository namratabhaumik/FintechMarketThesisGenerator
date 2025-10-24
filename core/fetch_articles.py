# core/fetch_articles.py
import feedparser
import logging

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q=fintech+OR+digital+payments+OR+open+banking&hl=en-US&gl=US&ceid=US:en"


def fetch_articles_from_rss(limit: int = 10):
    """
    Fetch fintech-related articles from Google News RSS.
    Returns a list of dicts with title, link, and summary text.
    """
    feed = feedparser.parse(GOOGLE_NEWS_RSS)
    articles = []

    for entry in feed.entries[:limit]:
        title = entry.get("title", "Untitled")
        link = entry.get("link", "")
        summary = entry.get("summary", "")
        if not summary:
            logger.warning(f"Skipping article with no summary: {title}")
            continue

        articles.append({
            "title": title,
            "url": link,
            "text": summary,
            "source": link.split('/')[2] if link else "Unknown"
        })

    logger.info(
        f"📡 Fetched {len(articles)} fintech-related articles from Google News RSS.")
    return articles
