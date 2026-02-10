"""Gemini LLM implementation."""

import json
import logging
import re
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from config.settings import LLMConfig
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
            chain = load_summarize_chain(self._llm, chain_type="map_reduce")
            result = chain.invoke(documents)

            if isinstance(result, dict) and "output_text" in result:
                return result["output_text"]

            return str(result)

        except Exception as e:
            logger.error(f"Gemini summarization failed: {e}")
            raise

    def generate_structured_output(self, prompt: str) -> Dict[str, Any]:
        """Generate structured JSON output from prompt.

        Args:
            prompt: Prompt for the LLM.

        Returns:
            Dictionary with 'raw' (original response) and 'json' (parsed JSON or None).
        """
        try:
            logger.info("Generating structured output with Gemini")
            result = self._llm.invoke(prompt)
            raw = getattr(result, "content", str(result))

            # Clean markdown code blocks
            cleaned = re.sub(
                r"^```json|```$",
                "",
                raw.strip(),
                flags=re.MULTILINE
            ).strip()

            parsed = json.loads(cleaned)
            logger.info("Successfully parsed structured output")
            return {"raw": raw, "json": parsed}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Gemini output as JSON: {e}")
            return {"raw": raw, "json": None}
        except Exception as e:
            logger.error(f"Error generating structured output: {e}")
            raise

    def get_model_name(self) -> str:
        """Get model identifier."""
        return self._config.model_name
