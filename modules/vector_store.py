"""
ChromaDB Persistent Vector Store Manager.
Maintains dense vector collections, embeddings, and retrieval indices.
"""

from typing import Any, Dict, List
import chromadb
from chromadb.config import Settings
from config import VECTOR_DB_DIR
from modules.embeddings import EmbeddingEngine
from utils.logger import setup_logger

logger = setup_logger("VectorStore")

class VectorStore:
    def __init__(self, collection_name: str = "diabetes_knowledge"):
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        self.embedding_engine = EmbeddingEngine()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Evidence-grounded diabetes knowledge base"}
        )

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Generates embeddings and persists chunks into ChromaDB."""
        if not chunks:
            return
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{
            "doc_id": c.get("doc_id", ""),
            "title": c.get("title", ""),
            "source": c.get("source", ""),
            "url": c.get("url", ""),
            "section": c.get("section", "")
        } for c in chunks]

        embeddings = self.embedding_engine.embed_texts(texts)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Persisted {len(ids)} chunks to ChromaDB collection.")

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Dense similarity vector search."""
        query_vector = self.embedding_engine.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for i in range(len(docs)):
                formatted_results.append({
                    "id": ids[i],
                    "text": docs[i],
                    "title": metas[i].get("title", ""),
                    "source": metas[i].get("source", ""),
                    "url": metas[i].get("url", ""),
                    "section": metas[i].get("section", ""),
                    "similarity_score": round(1.0 - distances[i], 4) if distances[i] <= 1.0 else round(1.0 / (1.0 + distances[i]), 4)
                })
        return formatted_results