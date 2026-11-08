"""
Helper utilities for clean JSON parsing, token estimation, and reciprocal rank fusion.
"""

import json
import re
from typing import Any, Dict, List

def clean_json_text(raw_text: str) -> str:
    """Extracts JSON substring from LLM markdown code blocks."""
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, raw_text)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def safe_parse_json(raw_text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Safely extracts and parses JSON strings with graceful fallbacks."""
    try:
        cleaned = clean_json_text(raw_text)
        return json.loads(cleaned)
    except Exception:
        # Secondary fallback regex attempt for curly braces
        try:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw_text[start:end+1])
        except Exception:
            pass
        return default

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Computes Reciprocal Rank Fusion (RRF) score across multiple ranked retrieval lists.
    Formula: RRF_Score(d) = sum(1.0 / (k + rank_i(d)))
    """
    rrf_scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Dict[str, Any]] = {}

    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs, start=1):
            doc_id = doc["id"]
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = doc
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort documents descending by fusion score
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)
    fused_results = []
    for doc_id in sorted_doc_ids:
        merged = dict(doc_lookup[doc_id])
        merged["rrf_score"] = round(rrf_scores[doc_id], 5)
        fused_results.append(merged)
        
    return fused_results