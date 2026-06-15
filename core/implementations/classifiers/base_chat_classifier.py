"""Shared base for chat-model fintech relevance classifiers.

Holds the prompt, the YES/NO parsing, and the fail-open behavior. Subclasses
only implement `_chat(messages)` to call their specific backend (Hugging Face,
Ollama, ...).
"""

import logging
from abc import abstractmethod
from typing import List

from core.interfaces.relevance_classifier import IRelevanceClassifier

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a strict financial technology (fintech) classifier. Fintech includes "
    "digital payments, crypto, banking infrastructure, trading apps, insurtech, and "
    "personal finance software. It DOES NOT include general AI, hardware, space, or "
    "standard e-commerce.\n"
    "Read this title and description. Is it related to fintech? Reply ONLY with the "
    "exact word YES or NO."
)


class BaseChatClassifier(IRelevanceClassifier):
    """Classifies fintech relevance via a chat model returning YES/NO."""

    def is_relevant(self, title: str, description: str) -> bool:
        """Return True if the title/description is classified as fintech.

        Fails open: on any API/parse error this returns True so transient
        failures never silently drop articles. Errors are logged.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {title}\nDescription: {description}"},
        ]
        try:
            answer = (self._chat(messages) or "").strip().upper()
            is_fintech = answer.startswith("YES")
            logger.debug(f"Classifier '{title[:60]}' -> {answer!r} ({is_fintech})")
            return is_fintech
        except Exception as e:
            logger.warning(f"Classification failed for '{title[:60]}', keeping it: {e}")
            return True

    @abstractmethod
    def _chat(self, messages: List[dict]) -> str:
        """Send chat `messages` to the backend and return the reply text."""
