"""Unit tests for SupabaseQuarantineRepository (URL dedup)."""

from core.implementations.repositories.supabase_quarantine_repository import (
    SupabaseQuarantineRepository,
)
from core.models.quarantine_record import (
    INVALID_ARTICLE,
    SCRAPE_FAILED,
    QuarantineRecord,
)
from tests.unit.fake_supabase import FakeClient as _FakeClient


def test_add_records_and_quarantined_urls():
    repo = SupabaseQuarantineRepository(_FakeClient())
    added = repo.add(
        [
            QuarantineRecord(url="https://x/1", reason=SCRAPE_FAILED),
            QuarantineRecord(url="https://x/2", reason=INVALID_ARTICLE, detail="empty source"),
        ]
    )
    assert added == 2
    assert repo.quarantined_urls() == {"https://x/1", "https://x/2"}


def test_add_dedupes_by_url():
    repo = SupabaseQuarantineRepository(_FakeClient())
    repo.add([QuarantineRecord(url="https://x/1", reason=SCRAPE_FAILED)])

    again = repo.add(
        [
            QuarantineRecord(url="https://x/1", reason=SCRAPE_FAILED),
            QuarantineRecord(url="https://x/2", reason=SCRAPE_FAILED),
        ]
    )
    assert again == 1
    assert repo.quarantined_urls() == {"https://x/1", "https://x/2"}


def test_add_empty_is_noop():
    repo = SupabaseQuarantineRepository(_FakeClient())
    assert repo.add([]) == 0
    assert repo.quarantined_urls() == set()
