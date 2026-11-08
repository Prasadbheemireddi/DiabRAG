"""
Pipeline C: Multi-Query RAG.
Synthesizes multiple diverse search queries via LLM, queries the vector space,
and deduplicates results. Targeted for Difficult, multi-perspective questions.
"""

from typing import Any, Dict, List
from google import genai
from modules.vector_store import VectorStore
from utils.helpers import safe_parse_json
from config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K_MULTI_QUERY
from utils.logger import setup_logger

logger = setup_logger("MultiQueryRAG")

class MultiQueryRAG:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def generate_sub_queries(self, query: str) -> List[str]:
        """Generates 2-3 focused diabetes search queries to cover all facets."""
        prompt = (
            f"You are an AI specialized in medical information retrieval.\n"
            f"Given the user's diabetes-related question, generate exactly 3 diverse, specific "
            f"search queries that unpack different aspects (e.g., etiology, diagnostics, clinical presentation).\n"
            f"Question: \"{query}\"\n\n"
            f"Return ONLY valid JSON matching this structure:\n"
            f'{{"sub_queries": ["query 1", "query 2", "query 3"]}}'
        )
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            parsed = safe_parse_json(response.text, default={"sub_queries": [query]})
            return parsed.get("sub_queries", [query])
        except Exception as e:
            logger.error(f"Multi-query generation failed: {e}. Falling back to original query.")
            return [query]

    def retrieve(self, query: str, top_k: int = TOP_K_MULTI_QUERY) -> List[Dict[str, Any]]:
        sub_queries = self.generate_sub_queries(query)
        logger.info(f"[Multi-Query RAG] Generated variants: {sub_queries}")

        aggregated_docs: Dict[str, Dict[str, Any]] = {}
        for sub_q in sub_queries:
            results = self.vector_store.search(query=sub_q, top_k=top_k)
            for r in results:
                doc_id = r["id"]
                if doc_id not in aggregated_docs:
                    r["retrieval_strategy"] = "Multi-Query Dense Variant Search"
                    aggregated_docs[doc_id] = r

        deduplicated = list(aggregated_docs.values())
        return deduplicated[:top_k]