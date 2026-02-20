"""Gemini LLM implementation."""

import logging
from typing import List

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

    def get_model_name(self) -> str:
        """Get model identifier."""
        return self._config.model_name
