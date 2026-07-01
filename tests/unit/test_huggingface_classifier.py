"""Unit tests for the Hugging Face fintech relevance classifier."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config.settings import ClassifierConfig
from core.implementations.classifiers.base_chat_classifier import SYSTEM_PROMPT
from core.implementations.classifiers.huggingface_classifier import (
    HuggingFaceFintechClassifier,
)


def _make_classifier(monkeypatch, content, model="Qwen/Qwen2.5-7B-Instruct"):
    """Build a classifier whose InferenceClient returns `content`.

    Returns (classifier, fake_client, captured_ctor_kwargs).
    """
    fake_client = Mock()
    fake_client.chat_completion.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    captured = {}

    def fake_ctor(**kwargs):
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(
        "core.implementations.classifiers.huggingface_classifier.InferenceClient",
        fake_ctor,
    )
    classifier = HuggingFaceFintechClassifier(ClassifierConfig(api_key="t", model=model))
    return classifier, fake_client, captured


def test_uses_configured_model_and_prompt(monkeypatch):
    """The classifier sends the system prompt and the configured model."""
    classifier, fake_client, captured = _make_classifier(
        monkeypatch, "YES", model="Qwen/Qwen2.5-7B-Instruct"
    )

    assert classifier.is_relevant("Stripe launches new API", "digital payments") is True

    assert captured["model"] == "Qwen/Qwen2.5-7B-Instruct"
    messages = fake_client.chat_completion.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Stripe launches new API" in messages[1]["content"]
    assert "digital payments" in messages[1]["content"]


def test_model_is_configurable(monkeypatch):
    """A different HF repo id is passed straight through to the client."""
    _, _, captured = _make_classifier(monkeypatch, "YES", model="meta-llama/Llama-3.1-8B-Instruct")
    assert captured["model"] == "meta-llama/Llama-3.1-8B-Instruct"


def test_no_is_not_relevant(monkeypatch):
    """A NO answer marks the entry as not fintech."""
    classifier, _, _ = _make_classifier(monkeypatch, "NO")
    assert classifier.is_relevant("SpaceX IPO", "space launch") is False


def test_answer_is_case_and_whitespace_insensitive(monkeypatch):
    """Leading/trailing whitespace and lowercase still parse correctly."""
    classifier, _, _ = _make_classifier(monkeypatch, "  yes\n")
    assert classifier.is_relevant("Robinhood", "trading app") is True


def test_raises_on_api_error(monkeypatch):
    """On any API error the error propagates (no fail-open); the caller decides."""
    fake_client = Mock()
    fake_client.chat_completion.side_effect = RuntimeError("HF down")
    monkeypatch.setattr(
        "core.implementations.classifiers.huggingface_classifier.InferenceClient",
        lambda **kwargs: fake_client,
    )
    classifier = HuggingFaceFintechClassifier(ClassifierConfig(api_key="t", model="m"))

    with pytest.raises(RuntimeError):
        classifier.is_relevant("anything", "anything")
