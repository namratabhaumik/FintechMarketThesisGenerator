"""Unit tests for SemanticScoringStrategy.

Fully offline: a fake IEmbeddingModel returns canned vectors chosen so the
per-article z-score lands each case on the intended add/veto decision. No model
download, no network.
"""

from finthesis_internal.keyword_scoring_strategy import KeywordCountScoringStrategy
from finthesis_internal.semantic_scoring_strategy import SemanticScoringStrategy
from core.interfaces.embeddings import IEmbeddingModel

# Orthogonal basis vectors the fake maps text onto.
A = [1, 0, 0, 0]     # "capital markets" / trading direction
NEG_A = [-1, 0, 0, 0]  # opposite of A (drives a strongly negative z-score)
B = [0, 1, 0, 0]     # payment direction
C = [0, 0, 1, 0]     # default / unrelated
D = [0, 0, 0, 1]     # neobank direction

CAPITAL_MARKETS = "Capital Markets & Trading Tech"  # an add-eligible category


class FakeEmbeddings(IEmbeddingModel):
    """Maps any text to a vector by first-substring-match, else a default.

    Counts embed calls so tests can assert the strategy skips embedding when it
    should short-circuit.
    """

    def __init__(self, sig_vectors, default=C):
        self._sigs = sig_vectors  # list of (substring, vector)
        self._default = default
        self.calls = 0

    def get_embeddings(self):
        return self

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def _vec(self, text):
        low = text.lower()
        for sub, vec in self._sigs:
            if sub in low:
                return list(vec)
        return list(self._default)

    def get_model_name(self):
        return "fake"


class BrokenEmbeddings(IEmbeddingModel):
    """Raises when asked for embeddings, to exercise the fallback path."""

    def get_embeddings(self):
        raise RuntimeError("embedding model unavailable")

    def get_model_name(self):
        return "broken"


def _tagged(scores):
    return {label for label, count in scores.items() if count > 0}


class TestSemanticAdditions:

    def test_adds_eligible_category_when_semantic_stands_out(self):
        """Capital Markets keyword-missed (base 0) but top by z-score -> added."""
        emb = FakeEmbeddings([
            ("capital markets", A), ("equities", A), ("trading", A),
            ("payment", B), ("neobank", D), ("challenger", D),
        ])
        strategy = SemanticScoringStrategy(emb)
        categories = {
            CAPITAL_MARKETS: ["equities"],       # keyword absent from text
            "Digital Payments": ["payment"],
            "Neobanking": ["challenger"],
        }
        # "trading" drives the article vector to A (Capital Markets direction);
        # none of the keywords appear, so the keyword base scores all zero.
        result = strategy.score("the firm expanded into trading desks", categories)

        assert result[CAPITAL_MARKETS] > 0       # semantic add
        assert result["Digital Payments"] == 0
        assert result["Neobanking"] == 0

    def test_does_not_add_ineligible_category_however_similar(self):
        """A non-add category topping the z-score is still never added."""
        emb = FakeEmbeddings([
            ("neobank", D), ("challenger", D),
            ("capital markets", A), ("equities", A),
        ])
        strategy = SemanticScoringStrategy(emb)
        categories = {
            CAPITAL_MARKETS: ["equities"],   # present only to trigger the semantic path
            "Neobanking": ["challenger"],    # not in add_categories
        }
        result = strategy.score("a brand new neobank app", categories)

        assert result["Neobanking"] == 0     # high similarity, but ineligible
        assert result[CAPITAL_MARKETS] == 0


class TestSemanticVetoes:

    def test_vetoes_single_keyword_tag_the_embedding_rejects(self):
        """Insurtech has one keyword hit but the article vector is opposite -> vetoed."""
        emb = FakeEmbeddings([
            ("trading", A), ("capital markets", A), ("equities", A),
            ("insurtech", NEG_A), ("insurance", NEG_A),
            ("payment", B),
        ])
        strategy = SemanticScoringStrategy(emb)
        categories = {
            CAPITAL_MARKETS: ["equities"],
            "Insurtech": ["insurance"],       # single keyword -> base count 1
            "Digital Payments": ["payment"],
        }
        # "trading" -> article vector A; "insurance" gives Insurtech its lone
        # keyword hit but its prototype is -A, so its z-score is the lowest.
        result = strategy.score(
            "trading and insurance and payment at the firm", categories
        )

        assert result["Insurtech"] == 0       # vetoed
        assert result["Digital Payments"] == 1  # kept (z-score not low enough)

    def test_does_not_veto_multi_keyword_tag(self):
        """Two keyword hits protect the tag even when the embedding disagrees."""
        emb = FakeEmbeddings([
            ("trading", A), ("capital markets", A), ("equities", A),
            ("insurtech", NEG_A), ("insurance", NEG_A), ("underwriting", NEG_A),
            ("payment", B),
        ])
        strategy = SemanticScoringStrategy(emb)
        categories = {
            CAPITAL_MARKETS: ["equities"],
            "Insurtech": ["insurance", "underwriting"],  # two hits -> base count 2
            "Digital Payments": ["payment"],
        }
        result = strategy.score(
            "trading and insurance and underwriting and payment", categories
        )

        assert result["Insurtech"] == 2       # multi-keyword hit is never vetoed


class TestShortCircuitAndFallback:

    def test_dimension_without_add_categories_skips_embedding(self):
        """Risk/signal maps (no add categories) return the keyword base, no embed."""
        emb = FakeEmbeddings([("regulation", A), ("breach", B)])
        strategy = SemanticScoringStrategy(emb)
        risk_like = {"Regulatory Risk": ["regulation"], "Cyber Risk": ["breach"]}

        result = strategy.score("a regulation and a breach occurred", risk_like)

        assert _tagged(result) == {"Regulatory Risk", "Cyber Risk"}
        assert emb.calls == 0                 # never embedded

    def test_broken_embedder_falls_back_to_keyword_base(self):
        """Any embedding failure degrades to the keyword result without raising."""
        strategy = SemanticScoringStrategy(BrokenEmbeddings())
        categories = {CAPITAL_MARKETS: ["equities"], "Digital Payments": ["payment"]}
        base = KeywordCountScoringStrategy().score("a payment at the firm", categories)

        result = strategy.score("a payment at the firm", categories)

        assert result == base                 # identical to keyword base
        assert result[CAPITAL_MARKETS] == 0   # no add happened on the fallback


class TestContract:

    def test_result_keys_match_category_map(self):
        emb = FakeEmbeddings([("equities", A), ("payment", B)])
        strategy = SemanticScoringStrategy(emb)
        categories = {CAPITAL_MARKETS: ["equities"], "Digital Payments": ["payment"]}

        result = strategy.score("some unrelated text", categories)

        assert set(result.keys()) == set(categories.keys())
