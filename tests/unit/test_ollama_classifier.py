"""Unit tests for the Ollama fintech relevance classifier."""

from unittest.mock import Mock

import pytest

from config.settings import ClassifierConfig
from core.implementations.classifiers.base_chat_classifier import SYSTEM_PROMPT
from core.implementations.classifiers.ollama_classifier import OllamaFintechClassifier


def _patch_post(monkeypatch, content=None, exc=None):
    """Patch requests.post used by the Ollama classifier."""
    mock_post = Mock()
    if exc is not None:
        mock_post.side_effect = exc
    else:
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        mock_post.return_value = resp
    monkeypatch.setattr(
        "core.implementations.classifiers.ollama_classifier.requests.post", mock_post
    )
    return mock_post


def test_uses_configured_model_and_prompt(monkeypatch):
    """The classifier targets the local endpoint with the configured model."""
    mock_post = _patch_post(monkeypatch, content="YES")
    classifier = OllamaFintechClassifier(
        ClassifierConfig(provider="ollama", model="qwen2.5:7b", base_url="http://localhost:11434")
    )

    assert classifier.is_relevant("Stripe payments", "digital payments") is True

    url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs["url"]
    assert url == "http://localhost:11434/v1/chat/completions"
    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "qwen2.5:7b"
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Stripe payments" in body["messages"][1]["content"]


def test_model_is_configurable(monkeypatch):
    """A different Ollama tag is sent straight through in the request body."""
    mock_post = _patch_post(monkeypatch, content="YES")
    classifier = OllamaFintechClassifier(ClassifierConfig(provider="ollama", model="llama3.2:3b"))
    classifier.is_relevant("x", "y")
    assert mock_post.call_args.kwargs["json"]["model"] == "llama3.2:3b"


def test_no_is_not_relevant(monkeypatch):
    """A NO answer marks the entry as not fintech."""
    _patch_post(monkeypatch, content="NO")
    classifier = OllamaFintechClassifier(ClassifierConfig(provider="ollama", model="qwen2.5:7b"))
    assert classifier.is_relevant("SpaceX IPO", "space") is False


def test_base_url_trailing_slash_is_normalized(monkeypatch):
    """A trailing slash on base_url does not produce a double slash."""
    mock_post = _patch_post(monkeypatch, content="YES")
    classifier = OllamaFintechClassifier(
        ClassifierConfig(provider="ollama", model="qwen2.5:7b", base_url="http://localhost:11434/")
    )
    classifier.is_relevant("x", "y")

    url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs["url"]
    assert url == "http://localhost:11434/v1/chat/completions"


def test_sends_bearer_header_when_api_key_set(monkeypatch):
    """A hosted endpoint (e.g. Ollama Cloud) gets an Authorization header."""
    mock_post = _patch_post(monkeypatch, content="YES")
    classifier = OllamaFintechClassifier(
        ClassifierConfig(
            provider="ollama", model="gpt-oss:20b",
            api_key="sk-test", base_url="https://ollama.com",
        )
    )
    classifier.is_relevant("x", "y")
    assert mock_post.call_args.kwargs["headers"] == {"Authorization": "Bearer sk-test"}


def test_no_auth_header_for_local_server(monkeypatch):
    """A local server (no api_key) sends no Authorization header."""
    mock_post = _patch_post(monkeypatch, content="YES")
    classifier = OllamaFintechClassifier(ClassifierConfig(provider="ollama", model="qwen2.5:7b"))
    classifier.is_relevant("x", "y")
    assert mock_post.call_args.kwargs["headers"] == {}


def test_strips_reasoning_block_before_parsing(monkeypatch):
    """An inline <think> block is stripped so the YES/NO check sees the answer."""
    _patch_post(monkeypatch, content="<think>Stripe is payments, so fintech.</think>YES")
    classifier = OllamaFintechClassifier(ClassifierConfig(provider="ollama", model="deepseek-r1"))
    assert classifier.is_relevant("Stripe payments", "digital payments") is True


def test_raises_when_ollama_unreachable(monkeypatch):
    """If Ollama is down the error propagates; Silver skips and retries 
    rather than freezing a guessed verdict."""
    _patch_post(monkeypatch, exc=ConnectionError("connection refused"))
    classifier = OllamaFintechClassifier(ClassifierConfig(provider="ollama", model="qwen2.5:7b"))
    with pytest.raises(ConnectionError):
        classifier.is_relevant("anything", "anything")
