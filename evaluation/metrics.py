"""
Standard RAG Evaluation Metrics.
Implements Token-level Exact Match, F1, Semantic Retrieval Recall@k,
and ROUGE-1/ROUGE-L scoring without fabricating performance figures.
"""

from typing import List, Set
import re
from rouge_score import rouge_scorer

class RAGMetrics:
    @staticmethod
    def normalize_answer(text: str) -> str:
        """Lowercases text and strips punctuation and articles."""
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return " ".join(text.split())

    @classmethod
    def exact_match_score(cls, prediction: str, ground_truth: str) -> float:
        """Returns 1.0 if normalized strings match completely, else 0.0."""
        return float(cls.normalize_answer(prediction) == cls.normalize_answer(ground_truth))

    @classmethod
    def f1_score(cls, prediction: str, ground_truth: str) -> float:
        """Computes token-level precision, recall, and F1 score."""
        pred_tokens = cls.normalize_answer(prediction).split()
        truth_tokens = cls.normalize_answer(ground_truth).split()

        if not pred_tokens or not truth_tokens:
            return float(pred_tokens == truth_tokens)

        common = set(pred_tokens) & set(truth_tokens)
        num_same = sum(min(pred_tokens.count(t), truth_tokens.count(t)) for t in common)

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return round(f1, 4)

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Measures whether relevant document IDs appear in the top-k retrieved."""
        if not relevant_ids:
            return 1.0
        top_k_retrieved = set(retrieved_ids[:k])
        hits = top_k_retrieved & relevant_ids
        return round(len(hits) / len(relevant_ids), 4)

    @staticmethod
    def rouge_scores(prediction: str, ground_truth: str):
        """Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth, prediction)
        return {
            "rouge1_f1": round(scores['rouge1'].fmeasure, 4),
            "rouge2_f1": round(scores['rouge2'].fmeasure, 4),
            "rougeL_f1": round(scores['rougeL'].fmeasure, 4)
        }