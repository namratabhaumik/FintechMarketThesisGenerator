"""Unit tests for SupabaseVectorStoreImpl's build-time URL dedup.

The dedup read is what stops Silver re-embedding an article it has already seen.
It used to scan the whole table, which PostgREST truncates at its max-rows cap
--> a partial "already embedded" set --> duplicate chunks. These tests pin the
read to the batch's own URLs so the answer stays complete at any table size.
"""

import pytest
from langchain_core.documents import Document

from config.settings import VectorStoreConfig
from core.implementations.vectorstores.supabase_vector_store import (
    _dedupe_chunks,
    SupabaseVectorStoreImpl,
)


class _FakeResp:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    """Mimics supabase-py's table builder for the select(...).in_(...) read."""

    def __init__(self, stored_urls, calls, fail=False):
        self._stored = stored_urls
        self._calls = calls
        self._fail = fail
        self._filter = None

    def select(self, *args):
        self._calls.append({"select": args})
        return self

    def in_(self, column, values):
        self._calls[-1]["in_"] = (column, list(values))
        self._filter = set(values)
        return self

    def execute(self):
        if self._fail:
            raise RuntimeError("boom")
        # Only rows matching the filter come back, as the DB would do it.
        matched = self._stored & (self._filter if self._filter is not None else self._stored)
        return _FakeResp([{"url": u} for u in sorted(matched)])


class _FakeClient:
    def __init__(self, stored_urls=(), fail=False):
        self.stored = set(stored_urls)
        self.calls: list = []
        self._fail = fail

    def table(self, name):
        return _FakeTable(self.stored, self.calls, self._fail)


class _FakeEmbeddingModel:
    def get_embeddings(self):
        return object()


def _store(client):
    return SupabaseVectorStoreImpl(VectorStoreConfig(), _FakeEmbeddingModel(), client)


def _doc(url):
    return Document(page_content="body", metadata={"url": url})


class TestFetchExistingUrls:
    def test_asks_only_about_the_batch_urls(self):
        # Table holds an unrelated URL; the read must not ask for it.
        client = _FakeClient(stored_urls={"a", "z"})
        existing = _store(client)._fetch_existing_urls(["a", "b"])

        assert existing == {"a"}
        assert client.calls[0]["in_"] == ("metadata->>url", ["a", "b"])

    def test_large_batch_is_split_across_several_lookups(self):
        """`in_` is serialized into the query string, so one list of every pending
        URL can outgrow the gateway's URI limit on a backlog run. The answer must
        still be complete once the slices are unioned."""
        urls = [f"u{i}" for i in range(250)]
        client = _FakeClient(stored_urls={"u0", "u149", "u249"})

        existing = _store(client)._fetch_existing_urls(urls)

        assert existing == {"u0", "u149", "u249"}
        assert len(client.calls) == 3  # 250 urls at a batch of 100
        assert all(len(c["in_"][1]) <= 100 for c in client.calls)

    def test_empty_batch_skips_the_read_entirely(self):
        client = _FakeClient(stored_urls={"a"})
        assert _store(client)._fetch_existing_urls([]) == set()
        assert client.calls == []

    def test_read_failure_raises_rather_than_returning_empty(self):
        # An empty set here would read as "nothing embedded yet" --> re-embed
        # everything --> duplicate chunks. Failing loudly is the safe direction.
        client = _FakeClient(stored_urls={"a"}, fail=True)
        with pytest.raises(RuntimeError, match="Failed to read existing URLs"):
            _store(client)._fetch_existing_urls(["a"])


class TestChunkDedup:
    """Repeated chunks within one article collide on documents_url_content_uniq,
    which fails the whole insert. They are dropped before embedding."""

    def test_identical_chunks_in_one_article_collapse_to_one(self):
        bio = Document(page_content="Richie is an advisor.", metadata={"url": "a"})
        chunks = [
            Document(page_content="body", metadata={"url": "a"}),
            bio,
            Document(page_content=bio.page_content, metadata={"url": "a"}),
        ]
        kept = _dedupe_chunks(chunks)
        assert [c.page_content for c in kept] == ["body", "Richie is an advisor."]

    def test_same_text_under_different_urls_is_kept(self):
        # Two articles sharing a boilerplate paragraph do not collide: the unique
        # key is (url, content), so each article keeps its own copy.
        chunks = [
            Document(page_content="shared footer", metadata={"url": "a"}),
            Document(page_content="shared footer", metadata={"url": "b"}),
        ]
        assert len(_dedupe_chunks(chunks)) == 2

    def test_no_repeats_passes_everything_through(self):
        chunks = [
            Document(page_content=f"c{i}", metadata={"url": "a"}) for i in range(3)
        ]
        assert _dedupe_chunks(chunks) == chunks


class TestBuildDedup:
    def test_build_looks_up_exactly_the_documents_urls(self, monkeypatch):
        client = _FakeClient(stored_urls={"seen"})
        store = _store(client)
        # build() returns a live handle we do not exercise here.
        monkeypatch.setattr(store, "open", lambda: object())
        embedded: list = []
        monkeypatch.setattr(
            "core.implementations.vectorstores.supabase_vector_store."
            "SupabaseVectorStore.from_documents",
            lambda chunks, *a, **kw: embedded.extend(chunks),
        )

        store.build([_doc("seen"), _doc("fresh")])

        assert client.calls[0]["in_"] == ("metadata->>url", ["seen", "fresh"])
        # Only the unseen article is embedded.
        assert {c.metadata["url"] for c in embedded} == {"fresh"}
