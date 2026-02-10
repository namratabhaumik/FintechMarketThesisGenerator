"""HuggingFace embeddings implementation."""

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from config.settings import EmbeddingConfig
from core.interfaces.embeddings import IEmbeddingModel

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingModel(IEmbeddingModel):
    """HuggingFace embeddings implementation."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize with embedding configuration.

        Args:
            config: Embedding configuration with model name.
        """
        self._config = config
        logger.info(f"Loading HuggingFace embeddings model: {config.model_name}")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=config.model_name
        )

    def get_embeddings(self) -> Embeddings:
        """Get LangChain embeddings instance."""
        return self._embeddings

    def get_model_name(self) -> str:
        """Get model identifier."""
        return self._config.model_name
