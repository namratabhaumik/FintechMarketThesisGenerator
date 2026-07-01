"""Manual build: aggregate the fintech corpus into per-theme weekly trends.

Recomputes per-(week, theme) coverage counts from the Silver-accepted articles
and upserts them into the trend_metrics table. Safe to re-run: counts are
recomputed from current data each time.

Prereqs:
    - Silver populated (scripts/build_silver.py)
    - trend_metrics table created (see sql/trend_metrics.sql)
    - SUPABASE_* in .env

Run from the repo root:
    python -m scripts.build_gold
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

    gold = container.get_gold_service()
    written = gold.build()
    print(f"\nGold build complete: {written} (week, theme) trend metrics written.")


if __name__ == "__main__":
    main()
