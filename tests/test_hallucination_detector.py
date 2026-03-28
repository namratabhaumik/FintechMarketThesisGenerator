"""Tests for hallucination detection on structured tool calls."""

from langchain_core.messages import AIMessage, HumanMessage

from core.agents.hallucination_detector import HallucinationDetector


def test_no_tool_calls():
    """No tool_calls in messages → no hallucinations."""
    detector = HallucinationDetector()
    messages = [HumanMessage(content="hello"), AIMessage(content="hi")]
    result = detector.analyze(messages)
    assert result["tool_calls"] == []
    assert not result["has_hallucinations"]


def test_valid_tool_calls_only():
    """All tool_calls are in the registry → no hallucinations."""
    detector = HallucinationDetector()
    ai_msg = AIMessage(content="", tool_calls=[
        {"name": "refine_thesis", "args": {"feedback_focus": "risks"}, "id": "1"},
    ])
    result = detector.analyze([ai_msg])
    assert result["valid_tools"] == ["refine_thesis"]
    assert result["invalid_tools"] == []
    assert not result["has_hallucinations"]


def test_hallucinated_tool_calls():
    """Tool calls not in registry → hallucinations detected."""
    detector = HallucinationDetector()
    ai_msg = AIMessage(content="", tool_calls=[
        {"name": "fetch_market_data", "args": {}, "id": "1"},
        {"name": "validate_sources", "args": {}, "id": "2"},
    ])
    result = detector.analyze([ai_msg])
    assert result["valid_tools"] == []
    assert set(result["invalid_tools"]) == {"fetch_market_data", "validate_sources"}
    assert result["has_hallucinations"]


def test_mixed_valid_and_hallucinated():
    """Mix of valid and invalid tool calls."""
    detector = HallucinationDetector()
    ai_msg = AIMessage(content="", tool_calls=[
        {"name": "refine_thesis", "args": {"feedback_focus": "risks"}, "id": "1"},
        {"name": "nonexistent_tool", "args": {}, "id": "2"},
    ])
    result = detector.analyze([ai_msg])
    assert result["valid_tools"] == ["refine_thesis"]
    assert result["invalid_tools"] == ["nonexistent_tool"]
    assert result["has_hallucinations"]
