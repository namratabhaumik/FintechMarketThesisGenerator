"""Unit tests for episodic recall (query-similarity ranking of past runs)."""

from types import SimpleNamespace

from core.services.episodic_recall import recall_similar


def _job(jid, emb):
    return SimpleNamespace(id=jid, query_embedding=emb)


class TestRecallSimilar:
    def test_ranks_by_cosine_and_caps_top_n(self):
        q = [1.0, 0.0]
        jobs = [
            _job("a", [1.0, 0.0]),   # identical -> 1.0
            _job("b", [0.0, 1.0]),   # orthogonal -> 0.0
            _job("c", [0.9, 0.1]),   # close
        ]
        out = recall_similar(q, jobs, top_n=2, min_score=0.0)
        assert [j.id for j, _ in out][0] == "a"   # most similar first
        assert len(out) == 2                       # capped at top_n

    def test_min_score_floor_drops_unrelated(self):
        q = [1.0, 0.0]
        jobs = [_job("orth", [0.0, 1.0])]          # cosine 0.0
        assert recall_similar(q, jobs, min_score=0.3) == []

    def test_skips_missing_embedding(self):
        q = [1.0, 0.0]
        jobs = [_job("none", None), _job("empty", []), _job("ok", [1.0, 0.0])]
        out = recall_similar(q, jobs, min_score=0.0)
        assert [j.id for j, _ in out] == ["ok"]

    def test_skips_dimension_mismatch(self):
        q = [1.0, 0.0]
        jobs = [_job("wrong", [1.0, 0.0, 0.0])]
        assert recall_similar(q, jobs) == []

    def test_skips_malformed_embedding(self):
        q = [1.0, 0.0]
        jobs = [_job("bad", ["x", "y"]), _job("ok", [1.0, 0.0])]
        out = recall_similar(q, jobs, min_score=0.0)
        assert [j.id for j, _ in out] == ["ok"]

    def test_empty_or_zero_query_returns_nothing(self):
        assert recall_similar([], [_job("a", [1.0, 0.0])]) == []
        assert recall_similar(None, [_job("a", [1.0, 0.0])]) == []
        assert recall_similar([0.0, 0.0], [_job("a", [1.0, 0.0])]) == []
