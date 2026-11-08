"""
Pipeline B: Hybrid RAG.
Dense Vector Search + BM25 Sparse Search combined via Reciprocal Rank Fusion (RRF).
Targeted for Medium complexity questions.
"""

from typing import Any, Dict, List
from modules.vector_store import VectorStore
from modules.bm25_retriever import BM25Retriever
from utils.helpers import reciprocal_rank_fusion
from config import TOP_K_HYBRID, RRF_K_CONSTANT
from utils.logger import setup_logger

logger = setup_logger("HybridRAG")

class HybridRAG:
    def __init__(self, vector_store: VectorStore, bm25_retriever: BM25Retriever):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str, top_k: int = TOP_K_HYBRID) -> List[Dict[str, Any]]:
        logger.info(f"[Hybrid RAG] Executing multi-modal retrieval for: '{query}'")
        dense_results = self.vector_store.search(query=query, top_k=top_k)
        bm25_results = self.bm25_retriever.search(query=query, top_k=top_k)

        # Merge ranked lists via Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            ranked_lists=[dense_results, bm25_results],
            k=RRF_K_CONSTANT
        )
        final_results = fused[:top_k]
        for r in final_results:
            r["retrieval_strategy"] = "Hybrid (Dense + BM25 RRF)"
        return final_results