"""Unit tests for the refinement graph: component resolution, frozen-number
assembly, the assemble node orchestration, and the tool registry shape."""

import json
from datetime import date
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from core.agents.refinement_graph import (
    _create_langfuse_handler,
    _diff_thesis,
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
def current_thesis():
    # All numbers are frozen on refine - carried forward unchanged - so these
    # sentinels must survive any re-assembly untouched.
    return StructuredThesis(
        key_themes=["Digital Payments"],
        risks=["Regulatory Risk"],
        investment_signals=["Payment Infrastructure"],
        sources=["u1", "u2"],
        raw_output="Original prose about digital payments.",
        opportunity_score=9.9,
        confidence_level=0.42,
        confidence_as_of=date(2026, 6, 15),
        recommendation="STALE",
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
    """_score_and_build carries every number forward (frozen) and only swaps in
    the refined content; key risks follow the refined risk list."""

    def test_freezes_numbers_and_swaps_content(self, current_thesis):
        components = (["Digital Payments"], ["Regulatory Risk", "Credit Risk"],
                      ["Payment Infrastructure"], ["u1", "u2"], "new prose")
        thesis = _score_and_build(components, current_thesis)

        assert isinstance(thesis, StructuredThesis)
        assert thesis.raw_output == "new prose"
        assert thesis.risks == ["Regulatory Risk", "Credit Risk"]
        # Every number is FROZEN: carried from current.
        assert thesis.opportunity_score == current_thesis.opportunity_score
        assert thesis.confidence_level == current_thesis.confidence_level
        assert thesis.confidence_as_of == current_thesis.confidence_as_of
        assert thesis.recommendation == current_thesis.recommendation
        # key_risk_factors follows the refined (displayed) risks, top 3.
        assert thesis.key_risk_factors == ["Regulatory Risk", "Credit Risk"]

    def test_numbers_frozen_regardless_of_content(self, current_thesis):
        # Different refined prose / tags must NOT move any number - they reflect
        # the retrieved evidence and Gold snapshot, which a refinement leaves alone.
        a = _score_and_build(([], [], [], [], SIGNAL_RICH), current_thesis)
        b = _score_and_build((["T"], ["R1", "R2"], ["S"], ["u"], RISK_HEAVY), current_thesis)
        assert a.opportunity_score == b.opportunity_score == current_thesis.opportunity_score
        assert a.confidence_level == b.confidence_level == current_thesis.confidence_level


class TestDiffThesis:
    """_diff_thesis surfaces the narrative + tag changes and always notes that
    the numbers stayed frozen."""

    def test_reports_narrative_and_tag_changes(self):
        current = StructuredThesis(
            key_themes=["Payments"], risks=["Reg", "Liquidity"],
            investment_signals=["Infra"], raw_output="old",
        )
        new = StructuredThesis(
            key_themes=["Payments", "Crypto"], risks=["Reg"],
            investment_signals=["Infra"], raw_output="new prose",
        )
        changes = _diff_thesis(current, new)
        assert "Narrative rewritten" in changes
        assert "Themes: +Crypto" in changes
        assert "Risks: -Liquidity" in changes
        assert not any(c.startswith("Signals:") for c in changes)  # unchanged dim omitted
        assert "Score, confidence, recommendation unchanged" in changes

    def test_identical_content_reports_only_frozen_numbers(self):
        current = StructuredThesis(key_themes=["A"], raw_output="same")
        new = StructuredThesis(key_themes=["A"], raw_output="same")
        assert _diff_thesis(current, new) == ["Score, confidence, recommendation unchanged"]


class TestAssembleNode:
    """assemble_node: swap refined content with numbers frozen; skip
    unknown/missing/unparseable tool results."""

    @staticmethod
    def _state(current, tool_msg=None, count=0):
        return {
            "messages": [tool_msg] if tool_msg else [],
            "current_thesis": current,
            "refinement_count": count,
            "execution_log": [],
        }

    def test_refine_thesis_freezes_numbers_and_logged(self, current_thesis):
        assemble = _make_assemble_node()
        msg = ToolMessage(content=json.dumps({
            "tool": "refine_thesis", "key_themes": ["Digital Payments"],
            "risks": ["Regulatory Risk"], "investment_signals": ["Payment Infrastructure"],
            "sources": ["u1", "u2"], "raw_output": SIGNAL_RICH,
        }), tool_call_id="1")

        out = assemble(self._state(current_thesis, msg))
        t = out["current_thesis"]

        # All numbers frozen (carried from current); only the content changes.
        assert t.opportunity_score == current_thesis.opportunity_score
        assert t.confidence_level == current_thesis.confidence_level
        assert t.confidence_as_of == current_thesis.confidence_as_of
        assert t.recommendation == current_thesis.recommendation
        assert t.raw_output == SIGNAL_RICH
        assert out["refinement_count"] == 1
        event = out["execution_log"][-1]
        assert event["tool_name"] == "refine_thesis"
        assert event["status"] == "executed"
        assert event["refinement_number"] == 1
        assert isinstance(event["changes"], list) and event["changes"]

    def test_unknown_tool_skips_and_keeps_thesis(self, current_thesis):
        assemble = _make_assemble_node()
        msg = ToolMessage(content=json.dumps({"tool": "score_opportunity"}), tool_call_id="2")

        out = assemble(self._state(current_thesis, msg))
        # skip() returns no current_thesis -> graph keeps the unchanged one.
        assert "current_thesis" not in out
        assert out["execution_log"][-1]["status"] == "skipped"
        assert out["refinement_count"] == 1

    def test_no_tool_message_skips(self, current_thesis):
        assemble = _make_assemble_node()
        out = assemble(self._state(current_thesis, tool_msg=None))
        assert out["execution_log"][-1]["tool_name"] == "no_tool"
        assert out["execution_log"][-1]["status"] == "skipped"

    def test_unparseable_tool_result_skips(self, current_thesis):
        assemble = _make_assemble_node()
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
            trend_repository=Mock(),  # refine_thesis does not read Gold
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
