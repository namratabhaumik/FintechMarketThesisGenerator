"""Manual ingest: land raw RSS feed entries into the Bronze store.

Collects feed entries verbatim (no classify, no scrape) and appends them to
the Supabase `articles_raw` table, deduped by URL. Run repeatedly over time
(eventually on a schedule) so the corpus accumulates a real history on the
published_at axis. Silver enrichment reads from these rows later.

Prereqs:
    - articles_raw table created (see sql/articles_raw.sql)
    - VECTORSTORE_PROVIDER / SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in .env

Run from the repo root:
    python -m scripts.ingest_bronze            # default limit per feed
    python -m scripts.ingest_bronze 100        # collect up to 100 per feed
"""

import logging
import sys

from dotenv import load_dotenv

from config.settings import AppConfig
from core.implementations.article_sources.rss_source import RSSArticleSource
from dependency_injection.container import ServiceContainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    load_dotenv()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    config = AppConfig.from_env()
    container = ServiceContainer(config)

    # No classifier / scraper: Bronze lands raw entries only.
    source = RSSArticleSource(config.rss_feeds, scraper=None, classifier=None)
    repo = container.get_article_repository()

    before = repo.count()
    raw_articles = source.collect_raw(limit=limit)
    print(f"\nCollected {len(raw_articles)} raw entries from the feeds.")

    inserted = repo.save(raw_articles)
    after = repo.count()
    print(
        f"Inserted {inserted} new (skipped {len(raw_articles) - inserted} "
        f"already-stored). Bronze total: {before} -> {after}."
    )


if __name__ == "__main__":
    main()
