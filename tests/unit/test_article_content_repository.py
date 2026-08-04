"""Unit tests for SupabaseArticleContentRepository (validated Silver record)."""

from datetime import datetime, timezone

from core.implementations.repositories.supabase_article_content_repository import (
    SupabaseArticleContentRepository,
)
from core.models.article import Article
from tests.unit.fake_supabase import FakeClient as _FakeClient

PUB = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _article(url, title="Title"):
    return Article(
        title=title, text="full text", source="x.com", url=url, published_at=PUB
    )


def test_save_and_fetch_all_round_trips():
    repo = SupabaseArticleContentRepository(_FakeClient())
    assert repo.save([_article("https://x/1"), _article("https://x/2")]) == 2

    out = {a.url: a for a in repo.fetch_all()}
    assert set(out) == {"https://x/1", "https://x/2"}
    assert out["https://x/1"].text == "full text"
    assert isinstance(out["https://x/1"].published_at, datetime)
    assert out["https://x/1"].published_at == PUB


def test_save_and_fetch_all_round_trips_load_id():
    """Lineage: the Bronze load_id is written and read back on the Article."""
    repo = SupabaseArticleContentRepository(_FakeClient())
    article = Article(
        title="T", text="body", source="x.com", url="https://x/1",
        published_at=PUB, load_id="load-1",
    )
    repo.save([article])
    assert repo.fetch_all()[0].load_id == "load-1"


def test_save_dedupes_by_url():
    repo = SupabaseArticleContentRepository(_FakeClient())
    repo.save([_article("https://x/1")])
    assert repo.save([_article("https://x/1"), _article("https://x/2")]) == 1
    assert len(repo.fetch_all()) == 2


def test_save_empty_is_noop():
    repo = SupabaseArticleContentRepository(_FakeClient())
    assert repo.save([]) == 0
    assert repo.fetch_all() == []
