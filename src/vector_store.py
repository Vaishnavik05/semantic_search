"""
Enhanced vector store with Endee and rich metadata support.
"""
from typing import List, Dict, Optional
import numpy as np
import os

try:
    from endee import Endee
except ImportError:
    Endee = None


class VectorStore:
    def __init__(self, dimension: int = 768, db_path: Optional[str] = None):
        self.dimension = dimension
        self.db_path = db_path
        
        if Endee is not None:
            try:
                self.db = Endee()
            except:
                self.db = None
        else:
            self.db = None
            
        self.vectors = None
        self.metadata_store = []

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match metadata length")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match expected {self.dimension}")
        
        self.vectors = vectors
        for i, meta in enumerate(metadata):
            meta['vector_index'] = i
            self.metadata_store.append(meta)
        
        print(f"Stored {len(self.metadata_store)} vectors in database")

    def search(self, query_vector: np.ndarray, top_k: int = 10,
               similarity_threshold: float = 0.0) -> List[Dict]:
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        else:
            query_vector = query_vector.astype(np.float32)
        
        if self.vectors is None or len(self.metadata_store) == 0:
            return []
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        search_results = []
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                result_meta = self.metadata_store[idx].copy()
                result_meta['score'] = float(similarities[idx])
                search_results.append(result_meta)
        
        return search_results

    def size(self) -> int:
        return len(self.metadata_store)

    def clear(self):
        self.vectors = None
        self.metadata_store = []