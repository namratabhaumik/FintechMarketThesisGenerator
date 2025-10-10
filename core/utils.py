# core/utils.py
import json
import logging


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
