"""
Semantic and Window-based Text Chunker.
Splits long medical texts into 300-500 token passages with preserved metadata.
"""

from typing import Dict, List
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
from utils.logger import setup_logger

logger = setup_logger("TextChunker")

class TextChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Splits document text into overlapping token windows and preserves metadata."""
        chunked_results: List[Dict[str, str]] = []
        chunk_counter = 0

        for doc in documents:
            text = doc["text"]
            words = text.split()
            
            if len(words) <= self.chunk_size:
                chunk_counter += 1
                chunked_results.append({
                    "id": f"chunk_{chunk_counter}",
                    "doc_id": doc.get("source", "DIAB") + f"_{chunk_counter}",
                    "title": doc.get("title", "Diabetes Medical Knowledge"),
                    "source": doc.get("source", "General"),
                    "url": doc.get("url", ""),
                    "section": doc.get("section", "General Section"),
                    "text": text,
                    "word_count": len(words)
                })
            else:
                start = 0
                while start < len(words):
                    end = min(start + self.chunk_size, len(words))
                    chunk_slice = " ".join(words[start:end])
                    chunk_counter += 1
                    
                    chunked_results.append({
                        "id": f"chunk_{chunk_counter}",
                        "doc_id": doc.get("source", "DIAB") + f"_{chunk_counter}",
                        "title": doc.get("title", "Diabetes Medical Knowledge"),
                        "source": doc.get("source", "General"),
                        "url": doc.get("url", ""),
                        "section": doc.get("section", "General Section"),
                        "text": chunk_slice,
                        "word_count": len(words[start:end])
                    })
                    
                    if end == len(words):
                        break
                    start += (self.chunk_size - self.overlap)

        logger.info(f"Chunking complete: {len(chunked_results)} chunks generated.")
        return chunked_results