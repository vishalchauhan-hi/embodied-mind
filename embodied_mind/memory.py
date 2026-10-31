"""
Episodic Memory — Stores agent experiences and retrieves relevant ones
via text embedding similarity for in-context learning.
"""

import os
import json
import logging
import hashlib
import re
from typing import Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Vector-similarity based episodic memory for storing and retrieving
    past agent experiences. Supports multiple embedding backends.
    
    Each memory entry contains:
        - task: str (the goal being pursued)
        - observation: str (scene description at decision point)
        - plan: str (plan that was executed)
        - actions: list[str] (action sequence taken)
        - outcome: str (success/failure description)
        - reward: float (cumulative reward)
        - lesson: str (post-episode reflection)
        - embedding: np.ndarray (text embedding for similarity search)
    """

    def __init__(
        self,
        embed_model: str = "text-embedding-3-small",
        persist_path: Optional[str] = None,
        max_memories: int = 500,
    ):
        self.embed_model = embed_model
        self.persist_path = Path(persist_path) if persist_path else None
        self.max_memories = max_memories

        self.memories: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None  # (N, D) matrix

        self._init_embedder()
        self._load_from_disk()

        logger.info(
            f"EpisodicMemory initialized | model={embed_model} "
            f"loaded={len(self.memories)} max={max_memories}"
        )

    def _init_embedder(self):
        """Initialize embedding client."""
        if "text-embedding" in self.embed_model and os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            self.embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embed_api = "openai"
            self.embed_dim = 1536
        else:
            # Fallback: use simple TF-IDF style hashing for offline use
            self.embed_client = None
            self.embed_api = "hash"
            self.embed_dim = 256
            logger.info("Using hash-based embeddings (offline mode)")

    @property
    def size(self) -> int:
        return len(self.memories)

    def store(self, experience: dict):
        """Store a new experience in memory."""
        # Generate text for embedding
        text = self._experience_to_text(experience)
        embedding = self._embed(text)

        entry = {**experience, "embedding": embedding}
        self.memories.append(entry)

        # Update embedding matrix
        emb_array = np.array(embedding).reshape(1, -1)
        if self.embeddings is None:
            self.embeddings = emb_array
        else:
            self.embeddings = np.vstack([self.embeddings, emb_array])

        # Evict oldest if over capacity
        if len(self.memories) > self.max_memories:
            self.memories.pop(0)
            self.embeddings = self.embeddings[1:]

        self._save_to_disk()
        logger.debug(f"Stored memory | total={self.size}")

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve the top-k most relevant memories for the given query."""
        if not self.memories:
            return []

        query_emb = np.array(self._embed(query))
        similarities = self._cosine_similarity(query_emb, self.embeddings)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum relevance threshold
                mem = {k: v for k, v in self.memories[idx].items() if k != "embedding"}
                mem["similarity"] = float(similarities[idx])
                results.append(mem)

        logger.debug(f"Recalled {len(results)} memories for query: {query[:50]}...")
        return results

    def _experience_to_text(self, exp: dict) -> str:
        """Convert experience dict to a text string for embedding."""
        parts = [
            f"Task: {exp.get('task', '')}",
            f"Observation: {exp.get('observation', '')[:200]}",
            f"Plan: {exp.get('plan', '')}",
            f"Outcome: {exp.get('outcome', '')}",
            f"Lesson: {exp.get('lesson', '')}",
        ]
        return " | ".join(parts)

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if self.embed_api == "openai":
            return self._embed_openai(text)
        else:
            return self._embed_hash(text)

    def _embed_openai(self, text: str) -> list[float]:
        response = self.embed_client.embeddings.create(
            model=self.embed_model,
            input=text[:8000],  # Truncate to model limit
        )
        return response.data[0].embedding

    def _embed_hash(self, text: str) -> list[float]:
        """
        Offline bag-of-words embedding using the hashing trick: each token is
        hashed to a bucket with a sign. Texts that share words get similar
        vectors, so recall works without an API key (weaker than learned
        embeddings, but meaningful).
        """
        vec = np.zeros(self.embed_dim)
        for tok in re.findall(r"[a-z0-9_]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.embed_dim] += 1.0 if (h >> 1) % 2 == 0 else -1.0
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all rows in matrix."""
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return matrix_norms @ query_norm

    def _save_to_disk(self):
        """Persist memories to JSON file."""
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for m in self.memories:
            entry = {k: v for k, v in m.items() if k != "embedding"}
            entry["embedding"] = m["embedding"] if isinstance(m["embedding"], list) else m["embedding"].tolist()
            serializable.append(entry)
        self.persist_path.write_text(json.dumps(serializable, indent=2))

    def _load_from_disk(self):
        """Load memories from disk if available."""
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text())
            for entry in data:
                emb = entry.pop("embedding", None)
                if emb:
                    entry["embedding"] = emb
                    self.memories.append(entry)

            if self.memories:
                self.embeddings = np.array([m["embedding"] for m in self.memories])
            logger.info(f"Loaded {len(self.memories)} memories from {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to load memories: {e}")

    def clear(self):
        """Clear all memories."""
        self.memories.clear()
        self.embeddings = None
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()
        logger.info("Memory cleared")

    def summary_stats(self) -> dict:
        """Return summary statistics about stored memories."""
        if not self.memories:
            return {"total": 0}
        rewards = [m.get("reward", 0) for m in self.memories]
        tasks = set(m.get("task", "") for m in self.memories)
        return {
            "total": len(self.memories),
            "unique_tasks": len(tasks),
            "avg_reward": float(np.mean(rewards)),
            "success_rate": sum(1 for r in rewards if r > 0) / len(rewards),
        }
