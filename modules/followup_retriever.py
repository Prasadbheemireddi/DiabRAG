"""
Follow-Up Retrieval Engine.
Activated when evidence sufficiency evaluator reports gaps ('insufficient').
Generates a focused gap-closing query and executes Round 2 retrieval.
"""

from typing import Any, Dict, List
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from modules.vector_store import VectorStore
from utils.helpers import safe_parse_json
from utils.logger import setup_logger

logger = setup_logger("FollowUpRetriever")

class FollowUpRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def formulate_followup_query(self, query: str, missing_points: List[str]) -> str:
        """Generates a targeted query to bridge identified evidence gaps."""
        prompt = (
            f"The user asked: \"{query}\"\n"
            f"During retrieval, the following critical diabetes points were missing:\n"
            f"{missing_points}\n\n"
            f"Formulate a single, concise, laser-focused search query to retrieve this missing medical evidence.\n"
            f"Return JSON ONLY:\n"
            f'{{"targeted_query": "focused search query"}}'
        )
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            parsed = safe_parse_json(response.text, default={"targeted_query": " ".join(missing_points)})
            return parsed.get("targeted_query", " ".join(missing_points))
        except Exception:
            return " ".join(missing_points)

    def retrieve_additional(
        self, 
        query: str, 
        missing_points: List[str], 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Executes Round 2 retrieval for missing evidence points."""
        targeted_q = self.formulate_followup_query(query, missing_points)
        logger.info(f"[Follow-up Retrieval Round 2] Query: '{targeted_q}'")
        additional_docs = self.vector_store.search(query=targeted_q, top_k=top_k)
        for d in additional_docs:
            d["retrieval_strategy"] = "Round 2 Follow-Up Dense Search"
        return additional_docs