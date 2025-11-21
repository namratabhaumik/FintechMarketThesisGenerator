import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Default RSS feed sources for fintech news
DEFAULT_RSS_FEEDS = [
    {
        "name": "TechCrunch Fintech",
        "url": "https://techcrunch.com/category/fintech/feed/",
        "enabled": True
    },
    # Additional sources can be added here
    # {
    #     "name": "Financial Times",
    #     "url": "https://www.ft.com/rss/companies/financial-services",
    #     "enabled": False
    # },
]


def fetch_live_articles(query: str = "fintech", limit: int = 5, rss_feeds: list = None):
    """
    Fetches recent fintech-related news articles from RSS feeds.

    Args:
        query: Search term to filter articles (case-insensitive)
        limit: Maximum number of articles to fetch
        rss_feeds: Optional list of RSS feed configs. If None, uses DEFAULT_RSS_FEEDS.
                   Each config should be a dict with 'name', 'url', and 'enabled' keys.

    Returns:
        List of article dicts with keys: title, url, source, text
    """
    if rss_feeds is None:
        rss_feeds = DEFAULT_RSS_FEEDS

    # Filter to only enabled feeds
    enabled_feeds = [feed for feed in rss_feeds if feed.get("enabled", True)]

    try:
        import feedparser

        articles = []
        articles_needed = limit

        for feed_config in enabled_feeds:
            if len(articles) >= limit:
                break

            feed_url = feed_config.get("url")
            feed_name = feed_config.get("name", feed_url)

            try:
                logger.info(f"Fetching from: {feed_name}")
                feed = feedparser.parse(feed_url)

                if not feed.entries:
                    logger.warning(f"No entries found in feed: {feed_name}")
                    continue

                # Process entries directly (no filtering needed - feed is already category-specific)
                for entry in feed.entries[:articles_needed]:
                    title = entry.get("title", "Untitled")
                    url = entry.get("link", "")

                    if not url:
                        continue

                    logger.info(f"Processing: {title[:60]}... | URL: {url}")

                    source = urlparse(url).netloc

                    # Fetch article content
                    article_text = scrape_article_text(url)
                    if not article_text:
                        article_text = entry.get(
                            "summary", "No summary available.")

                    articles.append({
                        "title": title,
                        "url": url,
                        "source": source,
                        "text": article_text[:4000]  # limit length for embedding
                    })
                    logger.info(f"Successfully added article from {source}")

                    articles_needed = limit - len(articles)
                    if articles_needed <= 0:
                        break

            except Exception as e:
                logger.warning(f"Error fetching from {feed_name}: {e}")
                continue

        logger.info(
            f"Fetched and scraped {len(articles)} live articles from RSS feeds.")
        
        return articles

    except Exception as e:
        logger.error(f"Error fetching live articles: {e}")
        return []


def scrape_article_text(url: str) -> str:
    """Scrapes the visible main text content from an article page."""
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts, styles, and nav elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Extract paragraph text
        paragraphs = [p.get_text(strip=True)
                      for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
        logger.debug(f"Found {len(paragraphs)} paragraphs")
        text = " ".join(paragraphs)
        return text.strip()

    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return ""
