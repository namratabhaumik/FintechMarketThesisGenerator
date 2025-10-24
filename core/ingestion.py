"""
core/ingestion.py
-----------------
Fetches live fintech-related news or articles for retrieval context.

- Uses requests + BeautifulSoup
- Scrapes a few article summaries from open sources (Google News RSS)
- Returns them in the same structure as sample_articles.json
"""

import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


def fetch_live_articles(query: str = "fintech", limit: int = 5):
    """
    Fetch recent fintech-related news headlines and summaries from Google News RSS.
    Returns a list of article dicts: {title, source, text}
    """
    try:
        rss_url = f"https://news.google.com/rss/search?q={query}+finance+OR+fintech&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "xml")
        items = soup.find_all("item")[:limit]

        articles = []
        for item in items:
            title = item.title.text
            link = item.link.text
            description = item.description.text if item.description else "No summary available."

            articles.append({
                "title": title,
                "source": link,
                "text": description
            })

        logger.info(f"Fetched {len(articles)} live articles for '{query}'.")
        return articles

    except Exception as e:
        logger.error(f"Error fetching live articles: {e}")
        return []
