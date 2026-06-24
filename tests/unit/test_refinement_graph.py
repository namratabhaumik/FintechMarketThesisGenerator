"""Unit tests for the refinement graph: component resolution, deterministic
scoring/assembly, the assemble node orchestration, and the tool registry shape."""

import json
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from core.agents.refinement_graph import (
    _create_langfuse_handler,
    _make_assemble_node,
    _resolve_components,
    _score_and_build,
)
from core.agents.thesis_tools import create_thesis_tools
from core.models.thesis import StructuredThesis
from core.services.thesis_generator_service import ThesisGeneratorService
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService


SIGNAL_RICH = "Embedded finance, BaaS, BNPL and real-time payment rails scale with AI-driven tools."
RISK_HEAVY = "Regulatory enforcement, compliance bans, data breaches, fraud and recession risk."


@pytest.fixture
def scoring():
    return OpportunityScoringService()


@pytest.fixture
def current_thesis():
    return StructuredThesis(
        key_themes=["Digital Payments"],
        risks=["Regulatory Risk"],
        investment_signals=["Payment Infrastructure"],
        sources=["u1", "u2"],
        raw_output="Original prose about digital payments.",
        opportunity_score=9.9,          # frozen: must be carried forward unchanged on refine
        confidence_level=9.9,           # re-derived on refine (not frozen)
        recommendation="STALE",         # frozen: carried forward with the score
        key_risk_factors=["Regulatory Risk"],
    )


class TestResolveComponents:
    """_resolve_components maps a tool result to a components tuple, or None."""

    def test_refine_thesis_takes_content_from_result(self, current_thesis):
        result = {
            "tool": "refine_thesis", "key_themes": ["A", "B"], "risks": ["R"],
            "investment_signals": ["S"], "sources": ["x"], "raw_output": "new",
        }
        assert _resolve_components("refine_thesis", result, current_thesis) == (
            ["A", "B"], ["R"], ["S"], ["x"], "new"
        )

    def test_refine_thesis_falls_back_to_current_for_missing_keys(self, current_thesis):
        comp = _resolve_components("refine_thesis", {"tool": "refine_thesis"}, current_thesis)
        assert comp == (
            current_thesis.key_themes, current_thesis.risks,
            current_thesis.investment_signals, current_thesis.sources,
            current_thesis.raw_output,
        )

    def test_unknown_tool_returns_none(self, current_thesis):
        assert _resolve_components("bogus", {"tool": "bogus"}, current_thesis) is None

    def test_removed_score_opportunity_tool_is_unknown(self, current_thesis):
        # score_opportunity was removed - it must no longer resolve to components.
        assert _resolve_components("score_opportunity", {}, current_thesis) is None


class TestScoreAndBuild:
    """_score_and_build freezes the score (carried from current) and re-derives
    only confidence + key risks from the refined components."""

    def test_freezes_score_and_recomputes_confidence(self, scoring, current_thesis):
        components = (["Digital Payments"], ["Regulatory Risk"],
                      ["Payment Infrastructure"], ["u1", "u2"], "new prose")
        thesis = _score_and_build(scoring, components, current_thesis)

        assert isinstance(thesis, StructuredThesis)
        assert thesis.key_themes == ["Digital Payments"]
        assert thesis.raw_output == "new prose"
        # Score + recommendation are FROZEN: carried from current, not recomputed.
        assert thesis.opportunity_score == current_thesis.opportunity_score
        assert thesis.recommendation == current_thesis.recommendation
        # Confidence + key risks are re-derived from the refined components.
        expected = scoring.assess_confidence(
            risks=["Regulatory Risk"],
            investment_signals=["Payment Infrastructure"],
            sources=["u1", "u2"],
        )
        assert thesis.confidence_level == expected["confidence_level"]
        assert thesis.key_risk_factors == expected["key_risks"]

    def test_score_is_frozen_regardless_of_content(self, scoring, current_thesis):
        # Different refined prose / tags must NOT move the score - it reflects the
        # retrieved evidence, which a refinement does not change.
        a = _score_and_build(scoring, ([], [], [], [], SIGNAL_RICH), current_thesis)
        b = _score_and_build(
            scoring, (["T"], ["R1", "R2"], ["S"], ["u"], RISK_HEAVY), current_thesis
        )
        assert a.opportunity_score == b.opportunity_score == current_thesis.opportunity_score


class TestAssembleNode:
    """assemble_node: re-score refined content, skip unknown/missing/unparseable."""

    @staticmethod
    def _state(current, tool_msg=None, count=0):
        return {
            "messages": [tool_msg] if tool_msg else [],
            "current_thesis": current,
            "refinement_count": count,
            "execution_log": [],
        }

    def test_refine_thesis_freezes_score_and_logged(self, scoring, current_thesis):
        assemble = _make_assemble_node(scoring)
        msg = ToolMessage(content=json.dumps({
            "tool": "refine_thesis", "key_themes": ["Digital Payments"],
            "risks": ["Regulatory Risk"], "investment_signals": ["Payment Infrastructure"],
            "sources": ["u1", "u2"], "raw_output": SIGNAL_RICH,
        }), tool_call_id="1")

        out = assemble(self._state(current_thesis, msg))
        t = out["current_thesis"]

        # Score + recommendation frozen (carried from current).
        assert t.opportunity_score == current_thesis.opportunity_score
        assert t.recommendation == current_thesis.recommendation
        # Confidence re-derived from the refined components (not frozen).
        expected = scoring.assess_confidence(
            risks=["Regulatory Risk"],
            investment_signals=["Payment Infrastructure"],
            sources=["u1", "u2"],
        )
        assert t.confidence_level == expected["confidence_level"]
        assert t.raw_output == SIGNAL_RICH
        assert out["refinement_count"] == 1
        assert out["execution_log"][-1] == {
            "tool_name": "refine_thesis", "status": "executed", "refinement_number": 1,
        }

    def test_unknown_tool_skips_and_keeps_thesis(self, scoring, current_thesis):
        assemble = _make_assemble_node(scoring)
        msg = ToolMessage(content=json.dumps({"tool": "score_opportunity"}), tool_call_id="2")

        out = assemble(self._state(current_thesis, msg))
        # skip() returns no current_thesis -> graph keeps the unchanged one.
        assert "current_thesis" not in out
        assert out["execution_log"][-1]["status"] == "skipped"
        assert out["refinement_count"] == 1

    def test_no_tool_message_skips(self, scoring, current_thesis):
        assemble = _make_assemble_node(scoring)
        out = assemble(self._state(current_thesis, tool_msg=None))
        assert out["execution_log"][-1]["tool_name"] == "no_tool"
        assert out["execution_log"][-1]["status"] == "skipped"

    def test_unparseable_tool_result_skips(self, scoring, current_thesis):
        assemble = _make_assemble_node(scoring)
        msg = ToolMessage(content="not json", tool_call_id="3")
        out = assemble(self._state(current_thesis, msg))
        assert out["execution_log"][-1]["status"] == "parse_error"


class TestCreateThesisTools:
    """The tool registry is a single tool after score_opportunity removal."""

    def test_returns_only_refine_thesis(self):
        tools = create_thesis_tools(thesis_service=None)  # not invoked at build time
        assert [t.name for t in tools] == ["refine_thesis"]


class TestRefineThesisContentOnly:
    """thesis_service.refine_thesis returns content only - scoring is deferred."""

    def test_refine_returns_content_without_scoring(self):
        llm = Mock()
        llm.refine.return_value = SIGNAL_RICH
        service = ThesisGeneratorService(
            llm=llm,
            scoring_service=OpportunityScoringService(),
        )
        current = StructuredThesis(raw_output="old", opportunity_score=4.0)

        result = service.refine_thesis(
            topic="x",
            documents=[Document(page_content="doc", metadata={"url": "u1"})],
            current_thesis=current,
            feedback_items=["more signals"],
        )

        assert result.raw_output == SIGNAL_RICH
        assert result.sources == ["u1"]
        assert isinstance(result.key_themes, list)
        # Scoring is handled downstream (assemble_node) - left at defaults here.
        assert result.opportunity_score == 0.0
        assert result.recommendation == ""


class TestLangfuseHandler:
    """Cheap branch: no credentials -> tracing disabled (None)."""

    def test_no_credentials_returns_none(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        assert _create_langfuse_handler() is None
