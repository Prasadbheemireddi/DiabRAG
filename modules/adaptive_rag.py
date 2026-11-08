"""
Master Adaptive RAG Orchestrator.
Directs the complete cycle:
Deterministic Triage -> Classify Complexity -> Select Pipeline -> Retrieve Round 1 ->
Evaluate Sufficiency -> Follow-Up Round 2 (if required) -> Generate Grounded Answer.
Supports dual-persona synthesis (Patient vs Medical Student) and multi-turn conversational follow-ups.
"""

from typing import Any, Dict, List
import time
from config import GEMINI_MODEL
from modules.vector_store import VectorStore
from modules.bm25_retriever import BM25Retriever
from modules.query_classifier import QueryComplexityClassifier
from modules.basic_rag import BasicRAG
from modules.hybrid_rag import HybridRAG
from modules.multi_query_rag import MultiQueryRAG
from modules.evidence_evaluator import EvidenceSufficiencyEvaluator
from modules.followup_retriever import FollowUpRetriever
from modules.answer_generator import AnswerGenerator
from modules.triage_engine import ClinicalTriageEngine
from utils.logger import setup_logger

logger = setup_logger("AdaptiveRAGOrchestrator")

class DiabetesAdaptiveRAG:
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_retriever = BM25Retriever()
        
        # Retrieval Pipelines
        self.basic_rag = BasicRAG(self.vector_store)
        self.hybrid_rag = HybridRAG(self.vector_store, self.bm25_retriever)
        self.multi_query_rag = MultiQueryRAG(self.vector_store)
        
        # Reasoning & Evaluation Modules
        self.classifier = QueryComplexityClassifier()
        self.evaluator = EvidenceSufficiencyEvaluator()
        self.followup = FollowUpRetriever(self.vector_store)
        self.generator = AnswerGenerator()

    def contextualize_query(self, query: str, chat_history: List[Dict[str, Any]]) -> str:
        """Reformulates pronoun-heavy follow-up questions into standalone queries using dialogue history."""
        if not chat_history:
            return query

        q_lower = query.lower()
        pronouns = [" it", " this", " that", " these", " those", "its ", "explain more", "what about", "and why", "how about"]
        if not any(p in q_lower for p in pronouns) and len(query.split()) > 6:
            return query

        last_turn = chat_history[-1]
        prev_q = last_turn.get("display_query") or last_turn.get("query", "")

        prompt = (
            f"Given the previous diabetes inquiry: \"{prev_q}\"\n"
            f"And the user's follow-up inquiry: \"{query}\"\n"
            "Rephrase the follow-up question into a single, standalone diabetes search query that resolves all pronouns and includes missing context. "
            "Output ONLY the rephrased query without commentary."
        )
        try:
            response = self.generator.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            rephrased = response.text.strip().replace('"', '')
            logger.info(f"Contextualized Follow-up: '{query}' -> '{rephrased}'")
            return rephrased
        except Exception as e:
            logger.warning(f"Query contextualization fallback: {e}")
            return query

    def run(
        self, 
        query: str, 
        user_mode: str = "patient", 
        chat_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Executes adaptive QA with deterministic triage, reflection, and conversational support."""
        start_time = time.time()
        
        # --- STEP 0: DETERMINISTIC CLINICAL TRIAGE LAYER (Executed First) ---
        triage_override = ClinicalTriageEngine.evaluate_triage(query)
        if triage_override:
            logger.info(f"Clinical Safety Override Engaged: {triage_override['triage_banner']}")

        # Resolve contextual query if this is a follow-up turn
        search_query = self.contextualize_query(query, chat_history or [])

        # If a critical acute emergency is detected, enrich search query so relevant guidelines are retrieved
        retrieval_query = search_query
        if triage_override and triage_override.get("pancreatitis_alert"):
            retrieval_query += " severe hypertriglyceridemia acute pancreatitis ACC AHA guidelines"
        elif triage_override and triage_override.get("dka_alert"):
            retrieval_query += " diabetic ketoacidosis DKA symptoms emergency management"

        workflow_trace: Dict[str, Any] = {
            "query": query,
            "search_query": search_query,
            "user_mode": user_mode,
            "triage_info": triage_override,  # Preserved for UI and telemetry
            "complexity_info": {},
            "pipeline_selected": "",
            "retrieval_round_1_count": 0,
            "retrieval_round_2_triggered": False,
            "retrieval_round_2_count": 0,
            "total_retrieval_rounds": 1,
            "evaluation_round_1": {},
            "evaluation_round_2": None,
            "passages": [],
            "answer": "",
            "latency_seconds": 0.0
        }

        # --- STEP 1: COMPLEXITY CLASSIFICATION & SAFETY OVERRIDE ---
        complexity_res = self.classifier.classify(search_query)
        complexity = complexity_res.get("complexity", "medium")

        # Safety Guardrail: Never run basic single-dense search on critical clinical alerts
        if triage_override and triage_override["urgency_level"] in ["EMERGENCY_REVIEW", "EMERGENCY"]:
            if complexity == "easy":
                complexity = "medium"
                complexity_res["reason"] += " (Promoted to Hybrid RAG due to critical clinical biomarker alert)."

        workflow_trace["complexity_info"] = complexity_res

        # --- STEP 2: DYNAMIC PIPELINE ROUTING ---
        passages: List[Dict[str, Any]] = []
        if complexity == "easy":
            workflow_trace["pipeline_selected"] = "Basic RAG (Dense Vector)"
            passages = self.basic_rag.retrieve(retrieval_query)
        elif complexity == "difficult":
            workflow_trace["pipeline_selected"] = "Multi-Query RAG (LLM Variations + Dense)"
            passages = self.multi_query_rag.retrieve(retrieval_query)
        else:
            workflow_trace["pipeline_selected"] = "Hybrid RAG (Dense + BM25 RRF)"
            passages = self.hybrid_rag.retrieve(retrieval_query)

        workflow_trace["retrieval_round_1_count"] = len(passages)

        # --- STEP 3: EVIDENCE SUFFICIENCY EVALUATION (ROUND 1) ---
        eval_res = self.evaluator.evaluate(retrieval_query, passages)
        workflow_trace["evaluation_round_1"] = eval_res

        # --- STEP 4: CONDITIONAL FOLLOW-UP RETRIEVAL (ROUND 2) ---
        final_eval = eval_res
        if eval_res.get("status") == "insufficient" and eval_res.get("missing_points"):
            logger.info("Sufficiency Check: Insufficient. Activating Follow-up Retrieval Round 2.")
            workflow_trace["retrieval_round_2_triggered"] = True
            workflow_trace["total_retrieval_rounds"] = 2
            
            missing_points = eval_res.get("missing_points", [])
            round_2_docs = self.followup.retrieve_additional(retrieval_query, missing_points, top_k=3)
            workflow_trace["retrieval_round_2_count"] = len(round_2_docs)

            existing_ids = {d["id"] for d in passages}
            for d in round_2_docs:
                if d["id"] not in existing_ids:
                    passages.append(d)
                    existing_ids.add(d["id"])

            final_eval = self.evaluator.evaluate(retrieval_query, passages)
            workflow_trace["evaluation_round_2"] = final_eval

        workflow_trace["passages"] = passages

        # --- STEP 5: EVIDENCE-GROUNDED ANSWER GENERATION ---
        final_answer = self.generator.generate_answer(
            query=query, 
            passages=passages, 
            evaluation_result=final_eval, 
            user_mode=user_mode,
            chat_history=chat_history,
            triage_override=triage_override
        )
        workflow_trace["answer"] = final_answer
        workflow_trace["latency_seconds"] = round(time.time() - start_time, 2)

        return workflow_trace