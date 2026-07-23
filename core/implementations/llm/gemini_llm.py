"""Gemini LLM implementation."""

import logging
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from config.settings import LLMConfig
from core.interfaces.llm import ILanguageModel
from core.utils.observability import get_callback_handler
from core.utils.text_utils import wrap_untrusted

logger = logging.getLogger(__name__)


class GeminiLanguageModel(ILanguageModel):
    """Gemini LLM implementation."""

    def __init__(self, config: LLMConfig):
        """Initialize with Gemini configuration.

        Args:
            config: LLM configuration with model name and API key.
        """
        self._config = config
        logger.info(f"Initializing Gemini LLM: {config.model_name}")

        # Attach the shared Langfuse handler (None when tracing is disabled) at
        # construction so both summarize and refine calls nest under whatever
        # per-request trace is active.
        handler = get_callback_handler()
        self._llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
            timeout=config.timeout,
            max_output_tokens=config.max_output_tokens,
            callbacks=[handler] if handler else None,
        )

    async def summarize(self, documents: List[Document], topic: str = "") -> str:
        """Generate a topic-focused summary from documents using Gemini.

        Args:
            documents: List of LangChain Document objects.
            topic: The user's query.

        Returns:
            Summarized text.

        Raises:
            Exception: If summarization fails.
        """
        try:
            logger.info(f"Summarizing {len(documents)} documents with Gemini")

            # Combine documents for summarization
            doc_content = "\n\n".join(doc.page_content for doc in documents)

            prompt = f"""You are a fintech market analyst. Using ONLY the source documents below, write a concise summary on: {topic}

Write in paragraphs.

If the source documents do not contain enough information to address the topic, respond with exactly "REFUSED: ".

{wrap_untrusted(doc_content, label="documents")}"""

            result = await self._llm.ainvoke([HumanMessage(content=prompt)])
            return result.content

        except Exception as e:
            logger.error(f"Gemini summarization failed: {e}")
            raise

    def get_model_name(self) -> str:
        """Get model identifier."""
        return self._config.model_name

    async def refine(
        self,
        documents: List[Document],
        current_thesis_text: str,
        feedback_items: List[str],
        prior_feedback: Optional[List[List[str]]] = None,
    ) -> str:
        """Refine thesis based on user feedback using direct chat call.

        Args:
            documents: Source documents for context.
            current_thesis_text: Original thesis to refine.
            feedback_items: This round's feedback constraints from user.
            prior_feedback: Earlier rounds' feedback (oldest first), shown as
                already-satisfied constraints to preserve.

        Returns:
            Refined thesis text.
        """
        # Build feedback constraints as bullet points
        feedback_str = "\n".join(f"- {item}" for item in feedback_items)

        # Earlier rounds, if any: shown as constraints to keep satisfying so a new
        # round does not regress them, and framed as "preserve" so repeated
        # feedback does not compound into over-correction.
        prior_items = [item for rnd in (prior_feedback or []) for item in rnd]
        prior_block = ""
        if prior_items:
            prior_str = "\n".join(f"- {item}" for item in prior_items)
            prior_block = (
                "\nFEEDBACK ALREADY ADDRESSED IN EARLIER ROUNDS (keep the thesis "
                f"consistent with these; do not undo them):\n{prior_str}\n"
            )

        # Concatenate document content for context
        doc_content = "\n\n".join(doc.page_content for doc in documents)

        # the LLM writes the narrative; no tool text is injected here
        prompt = f"""You are a fintech market analyst. Revise the investment thesis below to
address the reader's latest feedback, staying grounded in the source documents.

Write a clear, concise narrative analysis in prose. Do NOT use headings, bullet
lists, JSON, just paragraphs.

If the source documents do not contain enough information to address the
feedback, respond with exactly "REFUSED: ".

CURRENT THESIS:
{current_thesis_text}
{prior_block}
LATEST READER FEEDBACK (address each point):
{feedback_str}

SOURCE DOCUMENTS (for context):
{wrap_untrusted(doc_content, label="documents")}

Revised thesis:"""

        try:
            logger.info("Refining thesis with Gemini based on user feedback")
            result = await self._llm.ainvoke([HumanMessage(content=prompt)])
            logger.info("Thesis refinement complete")
            return result.content
        except Exception as e:
            logger.error(f"Gemini refinement failed: {e}")
            raise
