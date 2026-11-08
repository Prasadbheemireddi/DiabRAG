"""
BM25 Keyword Retrieval Engine using rank-bm25.
Constructs exact-match sparse indices for term-frequency retrieval.
"""

from typing import Any, Dict, List
import json
import re
from rank_bm25 import BM25Okapi
from config import PROCESSED_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("BM25Retriever")

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_path = PROCESSED_DATA_DIR / "bm25_corpus.json"

    def tokenize(self, text: str) -> List[str]:
        """Lowers and extracts alphanumeric tokens."""
        return re.findall(r"\w+", text.lower())

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Constructs and persists the BM25 inverted index."""
        self.chunks = chunks
        tokenized_corpus = [self.tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        logger.info(f"BM25 index built across {len(chunks)} passages.")

    def load_index(self) -> bool:
        """Loads chunks from disk and initializes BM25."""
        if self.corpus_path.exists():
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            tokenized_corpus = [self.tokenize(c["text"]) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 corpus loaded successfully from disk.")
            return True
        return False

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries sparse index and returns top-k ranked documents."""
        if not self.bm25 or not self.chunks:
            if not self.load_index():
                return []
        
        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0.0:
                doc_copy = dict(self.chunks[idx])
                doc_copy["bm25_score"] = round(float(doc_scores[idx]), 4)
                results.append(doc_copy)
        return results