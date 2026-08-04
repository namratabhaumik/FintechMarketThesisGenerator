"""BeautifulSoupScraper container selection and its drift warning.

The scraper narrows extraction to an article-body container. When every
selector misses it still returns text - the whole page, chrome included - and
nothing raises. Because Silver verdicts are frozen point-in-time, that noise is
embedded and tagged permanently, so the warning is the only chance to notice a
site changed its template.
"""

import logging
from unittest.mock import Mock, patch

from config.settings import ScraperConfig
from core.implementations.scrapers.beautifulsoup_scraper import BeautifulSoupScraper


def _response(html: str) -> Mock:
    resp = Mock()
    resp.text = html
    resp.raise_for_status = Mock()
    return resp


def _scrape(html: str, url: str = "https://betakit.com/some-post"):
    scraper = BeautifulSoupScraper(ScraperConfig())
    target = "core.implementations.scrapers.beautifulsoup_scraper.requests.get"
    with patch(target, return_value=_response(html)):
        return scraper.scrape(url)


class TestArticleContainerWarning:

    def test_no_warning_when_a_container_matches(self, caplog):
        html = """
            <html><body>
              <div class="wp-block-post-content"><p>The real article body.</p></div>
            </body></html>
        """
        with caplog.at_level(logging.WARNING):
            text = _scrape(html)

        assert "The real article body." in text
        assert "scrape_no_article_container" not in caplog.text

    def test_warns_with_host_and_counts_when_every_selector_misses(self, caplog):
        """No wp-block-post-content/entry-content/article/main anywhere, so root
        stays the whole document and the sidebar reads as article text."""
        html = """
            <html><body>
              <div class="post-body-v2"><p>The real article body.</p></div>
              <div class="sidebar"><p>Most popular this week</p></div>
            </body></html>
        """
        with caplog.at_level(logging.WARNING):
            text = _scrape(html)

        # The failure is silent in the return value: text still comes back, and
        # it now contains the chrome that the container would have excluded.
        assert "The real article body." in text
        assert "Most popular this week" in text

        assert "scrape_no_article_container" in caplog.text
        assert "host=betakit.com" in caplog.text
        assert "selectors_tried=4" in caplog.text
        assert "blocks=2" in caplog.text

    def test_warning_names_the_drifting_host(self, caplog):
        html = "<html><body><div><p>Body text here.</p></div></body></html>"
        with caplog.at_level(logging.WARNING):
            _scrape(html, url="https://www.americanbanker.com/news/x")

        assert "host=www.americanbanker.com" in caplog.text
