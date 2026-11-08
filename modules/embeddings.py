"""
Sentence Transformers Embedding Engine.
Loads all-MiniLM-L6-v2 to generate dense semantic vector representations.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME
from utils.logger import setup_logger

logger = setup_logger("EmbeddingEngine")

class EmbeddingEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
            logger.info(f"Initializing SentenceTransformer: {EMBEDDING_MODEL_NAME}")
            cls._instance.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return cls._instance

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for an array of strings."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generates dense vector embedding for a single user query."""
        embedding = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()