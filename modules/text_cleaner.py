"""
Text Normalization and Sanitation Module for Medical RAG.
Strips boilerplate, sanitizes control characters, and removes duplicate content.
"""

import re
from typing import Dict, List
from utils.logger import setup_logger

logger = setup_logger("TextCleaner")

class TextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        """Sanitizes text by stripping artifacts and consolidating whitespace."""
        if not text:
            return ""
        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)
        # Remove email patterns
        text = re.sub(r"\S+@\S+", "", text)
        # Remove non-ascii noise while keeping punctuation
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        # Consolidate spaces, tabs, and newlines
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Processes and deduplicates incoming document sections."""
        cleaned_docs = []
        seen_texts = set()

        for doc in documents:
            cleaned = self.clean_text(doc.get("text", ""))
            # Discard excessively short fragments (< 30 words)
            if len(cleaned.split()) < 30:
                continue
            
            # Simple hash-based deduplication
            text_signature = cleaned.lower()[:120]
            if text_signature in seen_texts:
                continue

            seen_texts.add(text_signature)
            cleaned_doc = dict(doc)
            cleaned_doc["text"] = cleaned
            cleaned_doc["title"] = self.clean_text(doc.get("title", "Diabetes Resource"))
            cleaned_docs.append(cleaned_doc)

        logger.info(f"Sanitized documents: {len(cleaned_docs)} high-fidelity passages retained.")
        return cleaned_docs