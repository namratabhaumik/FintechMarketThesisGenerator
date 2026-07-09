"""Offline eval for the refinement agent's planner decision.

The refinement graph has a SINGLE tool (refine_thesis), so there is no tool
routing to score. The planner's real decision is the three integer deltas it
passes - theme_delta / risk_delta / signal_delta, each in [-1, 0, +1] - which
say how many grounded tags to add/trim per dimension in response to feedback
(+1 surfaces one more concrete grounded tag, -1 trims; see _apply_cap_deltas).

This eval checks: for each fixed feedback reason, does the planner move the
expected dimension(s) in the expected direction? The expected moves are the
intended policy (EXPECTED below); some reasons expect multiple dimensions, in
which case ALL must match. Unspecified dimensions are "don't care".

    dataset  -> the 6 fixed FEEDBACK_OPTIONS, each with expected deltas
    task     -> run the refinement graph once; return the planner's deltas AND
                the rewritten summary
    scorers  -> 1. delta_correct (deterministic): did the planner's deltas match?
                2. rewrite_changed (deterministic): is the rewritten prose non-empty
                   and actually different from the original? (catches no-op rewrites)



Run from the repo root:
    python -m evals.refinement_eval

"""

import asyncio
from datetime import date
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langfuse import Evaluation, Langfuse
from langfuse.experiment import LocalExperimentItem

from config.settings import FEEDBACK_OPTIONS, AppConfig
from core.models.thesis import StructuredThesis
from dependency_injection.container import ServiceContainer

# Each fixed feedback reason (app.py FEEDBACK_OPTIONS) and the planner moves it
# should drive. Keys are tool-arg names; only the dimensions expected to change
# are listed (others are don't-care). Rows with >1 key require ALL to match.
# Expected planner deltas (policy) per feedback reason, keyed by the strings in
# config.FEEDBACK_OPTIONS. The dataset is built from that list, so a changed or
# removed option there surfaces here immediately as a KeyError instead of drift.
EXPECTED_DELTAS = {
    "Too many risks, not enough opportunities": {"risk_delta": -1, "signal_delta": 1},
    "Missing recent market trends": {"theme_delta": 1},
    "Investment signals are too vague": {"signal_delta": 1},
    "Opportunity score seems too low": {"signal_delta": 1},
    "Analysis is too broad, be more specific": {"theme_delta": -1, "signal_delta": 1},
    "Need stronger evidence for key themes": {
        "theme_delta": 0, "risk_delta": 0, "signal_delta": 0
    },
}

TOPIC = "embedded finance in Europe"


def _fixture_documents() -> List[Document]:
    """A few retrieved chunks carrying Silver tags so refine_thesis runs cleanly.

    The planner only reads the thesis score/recommendation + feedback, but the
    tool executes afterwards and derives tags from these documents.
    """
    common = {
        "themes": ["Digital Payments", "Embedded Finance"],
        "risks": ["Regulatory Risk"],
        "signals": ["Payment Infrastructure"],
    }
    return [
        Document(
            page_content=(
                "Embedded finance and BaaS let non-banks offer payments and "
                "lending. BNPL and real-time rails are scaling across the EU."
            ),
            metadata={"url": "https://example.com/embedded-finance", **common},
        ),
        Document(
            page_content=(
                "Regulators tighten oversight of BNPL and BaaS partnerships; "
                "compliance costs rise for fintech infrastructure providers."
            ),
            metadata={"url": "https://example.com/regulation", **common},
        ),
    ]


def _fixture_thesis() -> StructuredThesis:
    """A modest, mid-confidence thesis to refine."""
    return StructuredThesis(
        key_themes=["Digital Payments", "Embedded Finance"],
        risks=["Regulatory Risk"],
        investment_signals=["Payment Infrastructure"],
        sources=["https://example.com/embedded-finance"],
        raw_output="Embedded finance is growing in Europe, led by payments and BNPL.",
        opportunity_score=2.0,
        confidence_level=0.4,
        confidence_as_of=date(2026, 6, 1),
        recommendation="Investigate",
        key_risk_factors=["Regulatory Risk"],
    )


def _planner_deltas(messages: List[BaseMessage]) -> dict:
    """Pull the planner's chosen deltas + tool name from the AIMessage tool call."""
    for msg in messages:
        calls = getattr(msg, "tool_calls", None)
        if calls:
            args = calls[0].get("args", {})
            return {
                "theme_delta": args.get("theme_delta", 0),
                "risk_delta": args.get("risk_delta", 0),
                "signal_delta": args.get("signal_delta", 0),
                "tool": calls[0].get("name"),
            }
    return {}


def build_task(graph, handler):
    """Build the task: run the graph for one feedback reason, return deltas + summary."""

    def task(*, item, **kwargs) -> dict:
        state = {
            "topic": TOPIC,
            "documents": _fixture_documents(),
            "current_thesis": _fixture_thesis(),
            "feedback_history": [[item["input"]["feedback"]]],
            "refinement_count": 0,
            "status": "refining",
            "execution_log": [],
            "messages": [],
        }
        config = {"callbacks": [handler]} if handler else {}
        result = asyncio.run(graph.ainvoke(state, config=config))
        out = _planner_deltas(result.get("messages", []))
        refined = result.get("current_thesis")
        out["summary"] = getattr(refined, "raw_output", None)
        out["original"] = _fixture_thesis().raw_output
        return out

    return task


def delta_correct(*, input, output, expected_output=None, **kwargs) -> Evaluation:
    """Did the planner move ALL expected dimensions in the expected direction?"""
    want = expected_output["deltas"]
    checks = {k: output.get(k) == v for k, v in want.items()}
    ok = all(checks.values())
    got = {k: output.get(k) for k in ("theme_delta", "risk_delta", "signal_delta")}
    return Evaluation(
        name="delta_correct",
        value=ok,
        data_type="BOOLEAN",
        comment=f"want={want} got={got} match={checks}",
    )


def accuracy(*, item_results, **kwargs) -> Evaluation:
    vals = [
        e.value
        for r in item_results
        for e in r.evaluations
        if e.name == "delta_correct"
    ]
    hits = sum(bool(v) for v in vals)
    return Evaluation(
        name="accuracy",
        value=hits / len(vals) if vals else 0.0,
        data_type="NUMERIC",
        comment=f"{hits}/{len(vals)} correct",
    )


def rewrite_changed(*, input, output, expected_output=None, **kwargs) -> Evaluation:
    """Deterministic: is the rewritten prose non-empty and != the original?

    """
    revised = (output.get("summary") or "").strip()
    original = (output.get("original") or "").strip()
    changed = bool(revised) and revised != original
    return Evaluation(
        name="rewrite_changed",
        value=changed,
        data_type="BOOLEAN",
        comment="rewrite differs from original" if changed else "no-op or empty",
    )


def rewrite_changed_rate(*, item_results, **kwargs) -> Evaluation:
    vals = [
        e.value
        for r in item_results
        for e in r.evaluations
        if e.name == "rewrite_changed"
    ]
    hits = sum(bool(v) for v in vals)
    return Evaluation(
        name="rewrite_changed_rate",
        value=hits / len(vals) if vals else 0.0,
        data_type="NUMERIC",
        comment=f"{hits}/{len(vals)} changed",
    )


def _print_summary(result) -> None:
    print("\nPer-item results:")
    for r in result.item_results:
        fb = r.item["input"]["feedback"]
        by = {e.name: e for e in r.evaluations}
        d = by.get("delta_correct")
        c = by.get("rewrite_changed")
        dmark = "ok " if d and d.value else "XX "
        print(f"  [{dmark}] {fb[:44]:<44} delta | {d.comment if d else ''}")
        if c:
            cmark = "ok " if c.value else "XX "
            print(f"        [{cmark}] rewrite_changed | {c.comment}")
    for ev in result.run_evaluations:
        print(f"\n  {ev.name}: {ev.value:.3f}  ({ev.comment})")
    url = getattr(result, "dataset_run_url", None)
    if url:
        print(f"\nLangfuse: {url}")


def main() -> None:
    load_dotenv()
    config = AppConfig.from_env()
    container = ServiceContainer(config)
    graph, handler = container.get_refinement_graph()
    print(f"Planner LLM: {config.llm.model_name}")

    data: List[LocalExperimentItem] = [
        {"input": {"feedback": fb},
         "expected_output": {"deltas": EXPECTED_DELTAS[fb]}}
        for fb in FEEDBACK_OPTIONS
    ]

    langfuse = Langfuse()
    result = langfuse.run_experiment(
        name="refinement-planner-deltas",
        description=f"Planner delta selection per feedback reason ({config.llm.model_name})",
        data=data,
        task=build_task(graph, handler),
        evaluators=[delta_correct, rewrite_changed],
        run_evaluators=[accuracy, rewrite_changed_rate],
    )
    langfuse.flush()
    _print_summary(result)


if __name__ == "__main__":
    main()