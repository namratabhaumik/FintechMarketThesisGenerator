"""Shared base for chat-model fintech relevance classifiers.

Holds the prompt, the YES/NO parsing, and the fail-open behavior. Subclasses
only implement `_chat(messages)` to call their specific backend (Hugging Face,
Ollama, ...).
"""

import logging
from abc import abstractmethod
from typing import List

from core.interfaces.relevance_classifier import IRelevanceClassifier
from core.utils.text_utils import wrap_untrusted

logger = logging.getLogger(__name__)

# The instruction we send to the chat model. It pins down what counts as
# fintech and demands a one-word YES/NO answer.
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

        Returns the model's real answer: YES -> True, anything else -> False.

        build a system+user chat prompt --> ask the backend --> normalize
        the reply --> True only if it starts with "YES".
        """
        # Two-part chat prompt: the system rules, then the article's title and
        # description as the user turn for the model to judge.
        article_block = wrap_untrusted(
            f"Title: {title}\nDescription: {description}", label="article"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": article_block},
        ]
        # Ask the backend, then normalize: None --> "", trim whitespace,
        # uppercase, so the YES/NO check is robust to spacing and case.
        answer = (self._chat(messages) or "").strip().upper()
        # Treat only a leading "YES" as fintech. Anything else (including a
        # failed/empty reply) --> not relevant.
        is_fintech = answer.startswith("YES")
        logger.debug(f"Classifier '{title[:60]}' -> {answer!r} ({is_fintech})")
        return is_fintech

    @abstractmethod
    def _chat(self, messages: List[dict]) -> str:
        """Send chat `messages` to the backend and return the reply text."""
