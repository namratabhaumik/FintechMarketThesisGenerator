"""FastEmbed embeddings implementation (ONNX-based, no PyTorch dependency)."""

import logging
import os

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings

from config.settings import EmbeddingConfig
from core.interfaces.embeddings import IEmbeddingModel

logger = logging.getLogger(__name__)


class FastEmbedEmbeddingModel(IEmbeddingModel):
    """FastEmbed embeddings implementation using ONNX runtime."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize with embedding configuration.

        Args:
            config: Embedding configuration with model name.
        """
        self._config = config
        # Persistent cache_dir so the ONNX model survives OS temp-dir purges
        # (the library default caches under the temp dir).
        cache_dir = os.path.expanduser(config.cache_dir)
        logger.info(f"Loading FastEmbed model: {config.model_name} (cache: {cache_dir})")
        self._embeddings = FastEmbedEmbeddings(
            model_name=config.model_name,
            cache_dir=cache_dir,
        )

    def get_embeddings(self) -> Embeddings:
        """Get LangChain embeddings instance."""
        return self._embeddings

    def get_model_name(self) -> str:
        """Get model identifier."""
        return self._config.model_name
