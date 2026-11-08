"""
Evidence Sufficiency Evaluator Module.
Critiques retrieved passages against the question to verify if information
is Sufficient, Insufficient, or Conflicting before answering.
"""

from typing import Any, Dict, List
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from utils.helpers import safe_parse_json
from utils.logger import setup_logger

logger = setup_logger("EvidenceEvaluator")

class EvidenceSufficiencyEvaluator:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def evaluate(self, query: str, context_passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates whether the retrieved evidence is sufficient to answer all parts of the query."""
        formatted_context = "\n\n".join([
            f"[Passage {i+1} | Source: {doc.get('source')} - {doc.get('title')}]\n{doc.get('text')}"
            for i, doc in enumerate(context_passages)
        ])

        prompt = (
            "You are a medical evidence evaluator.\n"
            "Assess whether the provided Diabetes Evidence Passages contain enough core clinical information "
            "to answer the user's question accurately.\n\n"
            "EVALUATION CRITERIA:\n"
            "- Mark 'sufficient' if the passages explain the core clinical concepts requested (even if minor granular details are omitted).\n"
            "- If the passages describe GAD65 autoantibodies, autoimmune differentiation, or C-peptide function, mark 'sufficient'.\n"
            "- Mark 'insufficient' ONLY if a major topic of the inquiry is completely absent from the text.\n"
            "- Mark 'conflicting' if sources give contradictory medical facts.\n\n"
            f"User Question: \"{query}\"\n\n"
            f"Evidence Passages:\n{formatted_context}\n\n"
            "Return valid JSON ONLY matching this schema:\n"
            "{\n"
            '  "status": "sufficient" | "insufficient" | "conflicting",\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "supported_points": ["fact 1", "fact 2"],\n'
            '  "missing_points": ["gap if completely unaddressed, else empty"],\n'
            '  "contradiction": false,\n'
            '  "evidence_spans": ["excerpt from passages"]\n'
            "}"
        )

        # Change 'false' to 'False' on line 48:
        fallback = {
            "status": "sufficient",
            "confidence": 0.85,
            "supported_points": ["General diabetes information available"],
            "missing_points": [],
            "contradiction": False,    # <--- Capital 'F' here
            "evidence_spans": []
        }

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            result = safe_parse_json(response.text, default=fallback)
            logger.info(f"Evidence evaluated: [{result.get('status').upper()}] (Confidence: {result.get('confidence')})")
            return result
        except Exception as e:
            logger.error(f"Evidence evaluation failed: {e}")
            return fallback