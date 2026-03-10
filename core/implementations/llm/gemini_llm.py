"""Gemini LLM implementation."""

import logging
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from config.settings import LLMConfig
from core.agents.tool_registry import get_tools_description
from core.interfaces.llm import ILanguageModel

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

        self._llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key
        )

    def summarize(self, documents: List[Document]) -> str:
        """Generate summary from documents using Gemini.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            Summarized text.

        Raises:
            Exception: If summarization fails.
        """
        try:
            logger.info(f"Summarizing {len(documents)} documents with Gemini")

            # Combine documents for summarization
            doc_content = "\n\n".join(doc.page_content for doc in documents)

            prompt = f"""Provide a concise summary of the following documents:

{doc_content}

Summary:"""

            result = self._llm.invoke([HumanMessage(content=prompt)])
            return result.content

        except Exception as e:
            logger.error(f"Gemini summarization failed: {e}")
            raise

    def get_model_name(self) -> str:
        """Get model identifier."""
        return self._config.model_name

    def refine(
        self,
        documents: List[Document],
        current_thesis_text: str,
        feedback_items: List[str],
    ) -> str:
        """Refine thesis based on user feedback using direct chat call.

        Args:
            documents: Source documents for context.
            current_thesis_text: Original thesis to refine.
            feedback_items: Feedback constraints from user.

        Returns:
            Refined thesis text.
        """
        # Build feedback constraints as bullet points
        feedback_str = "\n".join(f"- {item}" for item in feedback_items)

        # Concatenate document content for context
        doc_content = "\n\n".join(doc.page_content for doc in documents)

        # Build the refinement prompt with tool constraints to prevent hallucination
        tools_description = get_tools_description()
        prompt = f"""You are a fintech market analyst. The user has reviewed an investment thesis
and provided specific feedback. Your task is to revise the thesis to directly address their concerns.

AVAILABLE TOOLS (only mention these):
{tools_description}

IMPORTANT: Do NOT mention, reference, or claim to use any tools or functions other than those listed above.
Do NOT invent or hallucinate tool names. Only use the available tools listed.

ORIGINAL THESIS:
{current_thesis_text}

USER FEEDBACK (things to improve):
{feedback_str}

SOURCE DOCUMENTS (for context):
{doc_content}

Please produce a revised thesis that:
1. Directly addresses the user's feedback points
2. Maintains the same structured format as the original
3. Uses evidence from the source documents
4. Provides actionable insights

Refined Thesis:"""

        logger.info("Refining thesis with Gemini based on user feedback")
        result = self._llm.invoke([HumanMessage(content=prompt)])
        logger.info("Thesis refinement complete")
        return result.content
