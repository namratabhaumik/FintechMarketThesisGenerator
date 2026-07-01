"""Manual build: enrich Bronze articles into the Silver embedded corpus.

Reads raw entries from articles_raw, keeps the fintech-relevant ones, scrapes
their full text, and embeds them into the Supabase pgvector store with their
publish date in metadata. Already-embedded URLs are skipped, so re-running only
processes genuinely new Bronze articles.

Prereqs:
    - Bronze populated (scripts/ingest_bronze.py)
    - VECTORSTORE_PROVIDER=supabase + SUPABASE_* in .env
    - A relevance classifier reachable (e.g. local Ollama) per CLASSIFIER_* env

Run from the repo root:
    python -m scripts.build_silver
"""

import logging

from dotenv import load_dotenv

from config.settings import AppConfig
from dependency_injection.container import ServiceContainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    load_dotenv()
    config = AppConfig.from_env()
    container = ServiceContainer(config)

    silver = container.get_silver_service()
    processed = silver.build()
    print(
        f"\nSilver build complete: {processed} fintech articles processed this run "
        "(the vector store logs how many were newly embedded)."
    )


if __name__ == "__main__":
    main()
