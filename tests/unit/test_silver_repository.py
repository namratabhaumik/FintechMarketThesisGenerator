"""Unit tests for the Silver SupabaseSilverRepository (verdict dedup)."""

from core.implementations.repositories.supabase_silver_repository import (
    SupabaseSilverRepository,
)
from core.models.silver_record import SilverVerdict
from tests.unit.fake_supabase import FakeClient as _FakeClient


def test_record_and_processed_urls():
    repo = SupabaseSilverRepository(_FakeClient())
    recorded = repo.record(
        [
            SilverVerdict(url="https://x/1", fintech_relevant=True),
            SilverVerdict(url="https://x/2", fintech_relevant=False),
        ]
    )
    assert recorded == 2
    assert repo.processed_urls() == {"https://x/1", "https://x/2"}


def test_fintech_tags_returns_all_dimensions_for_relevant_only():
    repo = SupabaseSilverRepository(_FakeClient())
    repo.record(
        [
            SilverVerdict(
                url="https://x/1",
                fintech_relevant=True,
                themes=["Payments"],
                risks=["Regulatory Risk"],
                signals=["Payment Infrastructure"],
            ),
            SilverVerdict(url="https://x/2", fintech_relevant=False),
        ]
    )
    assert repo.fintech_tags() == {
        "https://x/1": {
            "themes": ["Payments"],
            "risks": ["Regulatory Risk"],
            "signals": ["Payment Infrastructure"],
        }
    }


def test_record_dedupes_by_url():
    repo = SupabaseSilverRepository(_FakeClient())
    repo.record([SilverVerdict(url="https://x/1", fintech_relevant=True)])

    again = repo.record(
        [
            SilverVerdict(url="https://x/1", fintech_relevant=True),
            SilverVerdict(url="https://x/2", fintech_relevant=False),
        ]
    )
    assert again == 1
    assert repo.processed_urls() == {"https://x/1", "https://x/2"}


def test_record_empty_is_noop():
    repo = SupabaseSilverRepository(_FakeClient())
    assert repo.record([]) == 0
    assert repo.processed_urls() == set()
