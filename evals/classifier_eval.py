"""Offline eval for the fintech relevance classifier (binary YES/NO).

The shortest of the FinThesis evals: a deterministic surface with one correct
answer per row.

    dataset  -> evals/data/classifier_seed.json (hand-labelled YES/NO)
    task     -> classifier.is_relevant(title, description) -> "YES"/"NO"
    scorers  -> per-item exact match (BOOLEAN)
                run-level accuracy / precision / recall / F1 (NUMERIC),
                treating "YES" (fintech) as the positive class

Results are traced to Langfuse (uses the same LANGFUSE_* env keys already wired
for tracing) and a summary is printed locally so the run is useful either way.

Run from the repo root:
    python -m evals.classifier_eval
"""

import csv
import json
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langfuse import Evaluation, Langfuse
from langfuse.experiment import LocalExperimentItem

from config.settings import AppConfig
from dependency_injection.container import ServiceContainer

DATA_DIR = Path(__file__).parent / "data"
SYNTHETIC_PATH = DATA_DIR / "classifier_synthetic.json"  # hand-written fallback
REAL_CSV = DATA_DIR / "classifier_real.csv"  # human-reviewed real articles
POSITIVE = "YES"  # the fintech-relevant class


def load_dataset() -> Tuple[List[LocalExperimentItem], str]:
    """Load the eval dataset, preferring the human-reviewed real-article CSV.

    Falls back to the synthetic JSON dataset when the CSV is absent. Returns the
    items plus a short source label for the run description.
    """
    if REAL_CSV.exists():
        return _load_real_csv(), f"real:{REAL_CSV.name}"
    items: List[LocalExperimentItem] = json.loads(SYNTHETIC_PATH.read_text())
    return items, f"synthetic:{SYNTHETIC_PATH.name}"


def _load_real_csv() -> List[LocalExperimentItem]:
    """Read the reviewed real-article CSV into experiment items.

    `label` is the human-verified ground truth (expected_output); the article's
    `summary` is the classifier's description input.
    """
    items: List[LocalExperimentItem] = []
    with REAL_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            items.append(
                {
                    "input": {
                        "title": row["title"],
                        "description": row["summary"],
                    },
                    "expected_output": row["label"].strip().upper(),
                    "metadata": {
                        "url": row["url"],
                        "model_verdict": row["model_verdict"],
                    },
                }
            )
    return items


def build_task(classifier):
    """Build the task: run one article through the classifier -> YES/NO."""

    def task(*, item, **kwargs) -> str:
        article = item["input"]
        is_fintech = classifier.is_relevant(article["title"], article["description"])
        return POSITIVE if is_fintech else "NO"

    return task


def exact_match(*, input, output, expected_output=None, **kwargs) -> Evaluation:
    """Per-item scorer: did the classifier's YES/NO match the label?"""
    correct = expected_output is not None and output == expected_output
    return Evaluation(
        name="correct",
        value=correct,
        data_type="BOOLEAN",
        comment=f"predicted {output!r}, expected {expected_output!r}",
    )


def classification_metrics(*, item_results, **kwargs) -> List[Evaluation]:
    """Run-level scorer: accuracy + precision/recall/F1 for the YES class."""
    tp = fp = tn = fn = 0
    for r in item_results:
        pred = r.output
        exp = r.item.get("expected_output")
        if pred == POSITIVE and exp == POSITIVE:
            tp += 1
        elif pred == POSITIVE and exp != POSITIVE:
            fp += 1
        elif pred != POSITIVE and exp == POSITIVE:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    def metric(name: str, value: float, comment: str) -> Evaluation:
        return Evaluation(
            name=name, value=value, data_type="NUMERIC", comment=comment
        )

    return [
        metric("accuracy", accuracy, f"{tp + tn}/{total} correct"),
        metric("precision", precision, f"tp={tp}, fp={fp}"),
        metric("recall", recall, f"tp={tp}, fn={fn}"),
        metric("f1", f1, "harmonic mean of precision and recall"),
    ]


def _print_summary(result) -> None:
    """Print a local table so the run is readable without opening Langfuse."""
    print("\nPer-item results:")
    for r in result.item_results:
        title = r.item["input"]["title"]
        exp = r.item.get("expected_output")
        mark = "ok " if r.output == exp else "XX "
        print(f"  [{mark}] pred={r.output:<3} exp={exp:<3} | {title[:60]}")

    print("\nRun metrics:")
    for ev in result.run_evaluations:
        print(f"  {ev.name:<10} {ev.value:.3f}  ({ev.comment})")

    url = getattr(result, "dataset_run_url", None)
    if url:
        print(f"\nLangfuse: {url}")


def main() -> None:
    load_dotenv()
    config = AppConfig.from_env()
    provider = config.classifier.provider
    print(f"Classifier provider: {provider} | model: {config.classifier.model}")

    classifier = ServiceContainer(config).get_relevance_classifier()
    data, source = load_dataset()
    print(f"Loaded {len(data)} labelled items ({source})")

    langfuse = Langfuse()
    result = langfuse.run_experiment(
        name="classifier-fintech-relevance",
        description=f"Binary fintech YES/NO via {provider}:{config.classifier.model}",
        data=data,
        task=build_task(classifier),
        evaluators=[exact_match],
        run_evaluators=[classification_metrics],
        metadata={"provider": provider, "model": config.classifier.model},
    )
    langfuse.flush()
    _print_summary(result)


if __name__ == "__main__":
    main()