"""
Ablation and Comparison Evaluation Suite for Diabetes RAG.
Executes an empirical comparative study across 6 system architectures:
1. Fixed Basic RAG
2. Fixed Hybrid RAG
3. Fixed Multi-query RAG
4. Complexity Router Only
5. Evidence Sufficiency Evaluator Only
6. Complete Diabetes Adaptive RAG
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from typing import Any, Dict, List
from modules.vector_store import VectorStore
from modules.bm25_retriever import BM25Retriever
from modules.basic_rag import BasicRAG
from modules.hybrid_rag import HybridRAG
from modules.multi_query_rag import MultiQueryRAG
from modules.query_classifier import QueryComplexityClassifier
from modules.evidence_evaluator import EvidenceSufficiencyEvaluator
from modules.followup_retriever import FollowUpRetriever
from modules.answer_generator import AnswerGenerator
from modules.adaptive_rag import DiabetesAdaptiveRAG
from evaluation.metrics import RAGMetrics
from utils.logger import setup_logger

logger = setup_logger("EvaluationSuite")

# Gold-Standard Diabetes Reference Benchmark Dataset
EVALUATION_BENCHMARK: List[Dict[str, Any]] = [
    {
        "id": "q1",
        "question": "What is diabetes?",
        "ground_truth": (
            "Diabetes is a chronic condition that affects how the body turns food into energy. "
            "The body either doesn't make enough insulin or can't use it as well as it should, "
            "leading to high blood sugar levels."
        ),
        "complexity_target": "easy"
    },
    {
        "id": "q2",
        "question": "What are the common symptoms of type 2 diabetes?",
        "ground_truth": (
            "Common symptoms of type 2 diabetes include excessive thirst, frequent urination, "
            "increased hunger, fatigue, blurred vision, slow-healing sores, and unexplained weight loss."
        ),
        "complexity_target": "medium"
    },
    {
        "id": "q3",
        "question": "Compare type 1 and type 2 diabetes regarding causes, insulin dependence, and complications.",
        "ground_truth": (
            "Type 1 diabetes is an autoimmune condition where the immune system destroys insulin-producing "
            "beta cells, requiring lifelong insulin therapy. Type 2 diabetes involves insulin resistance and "
            "relative insulin deficiency, linked to lifestyle, overweight, and genetics. Both can cause long-term "
            "microvascular and macrovascular complications including retinopathy, nephropathy, neuropathy, and heart attacks."
        ),
        "complexity_target": "difficult"
    }
]

class ComparativeEvaluator:
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()
        
        # Modules
        self.basic_rag = BasicRAG(self.vector_store)
        self.hybrid_rag = HybridRAG(self.vector_store, self.bm25)
        self.multi_query_rag = MultiQueryRAG(self.vector_store)
        self.classifier = QueryComplexityClassifier()
        self.evaluator = EvidenceSufficiencyEvaluator()
        self.followup = FollowUpRetriever(self.vector_store)
        self.generator = AnswerGenerator()
        self.complete_rag = DiabetesAdaptiveRAG()

    def run_configuration(self, config_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
        q = item["question"]
        gt = item["ground_truth"]
        t0 = time.time()
        passages: List[Dict[str, Any]] = []
        rounds = 1

        if config_name == "Fixed Basic RAG":
            passages = self.basic_rag.retrieve(q)
            eval_res = {"status": "sufficient"}
            answer = self.generator.generate_answer(q, passages, eval_res)

        elif config_name == "Fixed Hybrid RAG":
            passages = self.hybrid_rag.retrieve(q)
            eval_res = {"status": "sufficient"}
            answer = self.generator.generate_answer(q, passages, eval_res)

        elif config_name == "Fixed Multi-query RAG":
            passages = self.multi_query_rag.retrieve(q)
            eval_res = {"status": "sufficient"}
            answer = self.generator.generate_answer(q, passages, eval_res)

        elif config_name == "Complexity Router Only":
            comp = self.classifier.classify(q).get("complexity", "medium")
            if comp == "easy":
                passages = self.basic_rag.retrieve(q)
            elif comp == "difficult":
                passages = self.multi_query_rag.retrieve(q)
            else:
                passages = self.hybrid_rag.retrieve(q)
            eval_res = {"status": "sufficient"}
            answer = self.generator.generate_answer(q, passages, eval_res)

        elif config_name == "Evaluator Only":
            passages = self.hybrid_rag.retrieve(q)
            eval_res = self.evaluator.evaluate(q, passages)
            if eval_res.get("status") == "insufficient":
                rounds = 2
                supp = self.followup.retrieve_additional(q, eval_res.get("missing_points", []))
                passages.extend(supp)
                eval_res = self.evaluator.evaluate(q, passages)
            answer = self.generator.generate_answer(q, passages, eval_res)

        elif config_name == "Complete Diabetes Adaptive RAG":
            res = self.complete_rag.run(q)
            passages = res["passages"]
            rounds = res["total_retrieval_rounds"]
            answer = res["answer"]

        latency = round(time.time() - t0, 3)
        f1 = RAGMetrics.f1_score(answer, gt)
        rouge = RAGMetrics.rouge_scores(answer, gt)
        
        return {
            "config": config_name,
            "latency": latency,
            "rounds": rounds,
            "f1": f1,
            "rougeL": rouge["rougeL_f1"],
            "passages_count": len(passages)
        }

    def execute_all(self):
        configurations = [
            "Fixed Basic RAG",
            "Fixed Hybrid RAG",
            "Fixed Multi-query RAG",
            "Complexity Router Only",
            "Evaluator Only",
            "Complete Diabetes Adaptive RAG"
        ]

        logger.info("Executing Benchmark Across 6 System Architectures...")
        summary: Dict[str, List[Dict[str, Any]]] = {c: [] for c in configurations}

        for item in EVALUATION_BENCHMARK:
            for c in configurations:
                res = self.run_configuration(c, item)
                summary[c].append(res)

        # Print Aggregated Results Table
        print("\n" + "="*85)
        print(f"{'Configuration':<32} | {'Avg Latency (s)':<15} | {'Avg Rounds':<11} | {'Avg F1':<8} | {'ROUGE-L':<8}")
        print("="*85)

        for c in configurations:
            entries = summary[c]
            avg_lat = round(sum(e["latency"] for e in entries) / len(entries), 2)
            avg_rnd = round(sum(e["rounds"] for e in entries) / len(entries), 2)
            avg_f1 = round(sum(e["f1"] for e in entries) / len(entries), 4)
            avg_rl = round(sum(e["rougeL"] for e in entries) / len(entries), 4)
            print(f"{c:<32} | {avg_lat:<15} | {avg_rnd:<11} | {avg_f1:<8} | {avg_rl:<8}")
        print("="*85 + "\n")

if __name__ == "__main__":
    evaluator = ComparativeEvaluator()
    evaluator.execute_all()