"""Ollama-backed fintech relevance classifier.

Runs a configurable, locally-served chat model (set via CLASSIFIER_MODEL;
default qwen2.5:7b) through Ollama's OpenAI-compatible chat completions
endpoint. No API key or network egress required.
"""

from typing import List

import requests

from config.settings import ClassifierConfig
from core.implementations.classifiers.base_chat_classifier import BaseChatClassifier


class OllamaFintechClassifier(BaseChatClassifier):
    """Classifies article relevance via a local Ollama chat model."""

    def __init__(self, config: ClassifierConfig):
        """Initialize with classifier configuration (model, base URL, timeout)."""
        self._url = f"{config.base_url.rstrip('/')}/v1/chat/completions"
        self._model = config.model
        self._timeout = config.timeout

    def _chat(self, messages: List[dict]) -> str:
        resp = requests.post(
            self._url,
            json={
                "model": self._model,
                "messages": messages,
                "max_tokens": 3,
                "temperature": 0.0,
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
