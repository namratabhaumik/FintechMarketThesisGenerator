"""Hugging Face-backed fintech relevance classifier.

Runs a configurable chat model (set via CLASSIFIER_MODEL; default
Qwen/Qwen2.5-7B-Instruct) via the official huggingface_hub InferenceClient.
"""

from typing import List

from huggingface_hub import InferenceClient

from config.settings import ClassifierConfig
from core.implementations.classifiers.base_chat_classifier import BaseChatClassifier


class HuggingFaceFintechClassifier(BaseChatClassifier):
    """Classifies article relevance via a hosted chat model on Hugging Face."""

    def __init__(self, config: ClassifierConfig):
        """Initialize with classifier configuration (model, HF token, timeout)."""
        self._client = InferenceClient(
            model=config.model,
            token=config.api_key,
            timeout=config.timeout,
        )

    def _chat(self, messages: List[dict]) -> str:
        completion = self._client.chat_completion(
            messages=messages,
            max_tokens=3,
            temperature=0.0,
        )
        return completion.choices[0].message.content
