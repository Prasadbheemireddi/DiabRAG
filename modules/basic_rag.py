"""
Pipeline A: Basic RAG.
Single dense vector similarity retrieval pipeline suited for Easy questions.
"""

from typing import Any, Dict, List
from modules.vector_store import VectorStore
from config import TOP_K_BASIC
from utils.logger import setup_logger

logger = setup_logger("BasicRAG")

class BasicRAG:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = TOP_K_BASIC) -> List[Dict[str, Any]]:
        logger.info(f"[Basic RAG] Executing dense search for query: '{query}'")
        results = self.vector_store.search(query=query, top_k=top_k)
        for r in results:
            r["retrieval_strategy"] = "Dense Vector Search"
        return results