"""
Dual Query Classifier: Complexity & Clinical Intent Classification.
Categorizes queries by:
1. IR Complexity: 'easy' | 'medium' | 'difficult'
2. Clinical Intent: 'diagnostic_criteria' | 'symptom_triage' | 'lab_interpretation' |
                    'pharmacotherapy' | 'pathophysiology' | 'lifestyle_prevention' | 'acute_emergency'
"""

from typing import Any, Dict, List
import re
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from utils.helpers import safe_parse_json
from utils.logger import setup_logger

logger = setup_logger("QueryClassifier")

CLINICAL_INTENTS = [
    "diagnostic_criteria",
    "symptom_triage",
    "lab_interpretation",
    "pharmacotherapy",
    "pathophysiology",
    "lifestyle_prevention",
    "acute_emergency"
]

class QueryComplexityClassifier:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    @staticmethod
    def extract_clinical_entities(query: str) -> List[str]:
        """Extracts recognizable medical entities and biomarkers from text."""
        entities = []
        q_lower = query.lower()
        patterns = {
            "triglycerides": r"\btriglycerides?\b",
            "fasting_glucose": r"\b(?:fasting\s+glucose|fpg|fbs)\b",
            "hba1c": r"\b(?:hba1c|a1c|glycated\s+hemoglobin)\b",
            "c_peptide": r"\bc[- ]?peptide\b",
            "gad65": r"\b(?:gad65|gada|anti[- ]?gad)\b",
            "ketones": r"\bketones?\b",
            "sglt2_inhibitors": r"\b(?:sglt2|empagliflozin|dapagliflozin)\b",
            "glp1_agonists": r"\b(?:glp[- ]?1|semaglutide|dulaglutide)\b",
            "metformin": r"\bmetformin\b",
            "polydipsia_thirst": r"\b(?:thirst|thirsty|polydipsia)\b",
            "polyuria_urination": r"\b(?:urination|peeing|polyuria|nocturia)\b",
            "dka": r"\b(?:dka|ketoacidosis)\b",
            "pancreatitis": r"\bpancreatitis\b",
            "lada": r"\blada\b",
            "mody": r"\bmody\b"
        }
        for entity_name, pattern in patterns.items():
            if re.search(pattern, q_lower):
                entities.append(entity_name)
        return entities

    @staticmethod
    def classify_intent_heuristically(query: str) -> str:
        """Rule-based clinical intent classifier for fast path."""
        q_lower = query.lower()

        if any(w in q_lower for w in ["dka", "ketoacidosis", "severe pain", "pancreatitis", "vomiting", "emergency", "unconscious"]):
            return "acute_emergency"
        if any(w in q_lower for w in ["report", "lab", "mg/dl", "mmol/l", "test result", "triglyceride", "hba1c", "glucose:"]):
            return "lab_interpretation"
        if any(w in q_lower for w in ["diagnos", "criteria", "threshold", "cutoff", "ogtt", "fasting range", "how do you know"]):
            return "diagnostic_criteria"
        if any(w in q_lower for w in ["drug", "medication", "metformin", "sglt2", "glp-1", "insulin dose", "prescrib"]):
            return "pharmacotherapy"
        if any(w in q_lower for w in ["thirsty", "peeing", "symptom", "sign", "feeling", "tired", "vision", "weight loss"]):
            return "symptom_triage"
        if any(w in q_lower for w in ["pathophysiology", "mechanism", "cause", "autoimmune", "beta cell", "genetics"]):
            return "pathophysiology"
        if any(w in q_lower for w in ["diet", "food", "exercise", "lifestyle", "prevent", "weight loss tips"]):
            return "lifestyle_prevention"

        return "diagnostic_criteria"

    def classify(self, query: str) -> Dict[str, Any]:
        """Evaluates both IR complexity and Clinical Intent."""
        q_lower = query.lower().strip()
        entities = self.extract_clinical_entities(query)
        heuristic_intent = self.classify_intent_heuristically(query)

        # 1. DIFFICULT COMPLEXITY HEURISTIC
        difficult_triggers = ["compare", "difference between", "versus", " vs ", "pathophysiology", "comprehensive", "differentiate"]
        if any(trigger in q_lower for trigger in difficult_triggers) or len(entities) >= 3:
            return {
                "complexity": "difficult",
                "intent": heuristic_intent,
                "entities": entities,
                "estimated_hops": 3,
                "reason": "Comparative or multi-entity medical inquiry requires multi-query expansion."
            }

        # 2. MEDIUM COMPLEXITY HEURISTIC
        aspects = ["symptom", "risk", "cause", "treatment", "prevent", "diagnos", "complication", "medication", "report", "lab"]
        matched_aspects = [a for a in aspects if a in q_lower]
        if len(matched_aspects) >= 2 or len(entities) == 2 or (" and " in q_lower and len(matched_aspects) >= 1):
            return {
                "complexity": "medium",
                "intent": heuristic_intent,
                "entities": entities,
                "estimated_hops": 2,
                "reason": f"Multi-aspect clinical inquiry ({', '.join(matched_aspects)}) requiring Hybrid RAG."
            }

        # 3. EASY COMPLEXITY HEURISTIC
        easy_prefixes = ["what is", "define", "meaning of", "explain "]
        if any(q_lower.startswith(p) for p in easy_prefixes) and len(matched_aspects) <= 1 and len(entities) <= 1:
            return {
                "complexity": "easy",
                "intent": heuristic_intent,
                "entities": entities,
                "estimated_hops": 1,
                "reason": "Direct single-concept definitional lookup suited for Basic Dense RAG."
            }

        # 4. LLM FALLBACK FOR NUANCED QUESTIONS
        system_prompt = (
            "You are an expert medical search query and clinical intent analyzer.\n"
            "Analyze the diabetes-related question across two dimensions:\n"
            "1. Complexity: 'easy' (1-hop lookup), 'medium' (2-3 facts/aspects), or 'difficult' (multi-entity comparison).\n"
            f"2. Clinical Intent: choose exactly one from {CLINICAL_INTENTS}.\n\n"
            "Return valid JSON ONLY:\n"
            '{"complexity": "easy"|"medium"|"difficult", "intent": "intent_name", "estimated_hops": 1|2|3, "reason": "string"}'
        )
        user_prompt = f"User Question: \"{query}\""
        fallback = {
            "complexity": "medium",
            "intent": heuristic_intent,
            "entities": entities,
            "estimated_hops": 2,
            "reason": "Defaulted to Hybrid RAG via heuristic fallback."
        }

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[system_prompt, user_prompt]
            )
            parsed = safe_parse_json(response.text, default=fallback)
            if parsed.get("complexity") not in ["easy", "medium", "difficult"]:
                parsed["complexity"] = "medium"
            if parsed.get("intent") not in CLINICAL_INTENTS:
                parsed["intent"] = heuristic_intent
            parsed["entities"] = entities
            logger.info(f"Classified: [{parsed['complexity'].upper()}] Intent: [{parsed['intent']}] - {parsed.get('reason')}")
            return parsed
        except Exception as e:
            logger.warning(f"Query classification fallback engaged: {e}")
            return fallback