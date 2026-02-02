"""
Enhanced vector store with Endee and rich metadata support.
"""
from typing import List, Dict, Optional
import numpy as np

try:
    from endee import Endee
except ImportError:
    Endee = None


class VectorStore:
    def __init__(self, dimension: int = 768, db_path: Optional[str] = None):
        if Endee is None:
            raise ImportError("Endee is required. Install with: pip install endee")
        self.dimension = dimension
        self.db_path = db_path
        self.db = Endee()
        self.metadata_store = []

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """Add vectors and metadata to the store."""
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match metadata length")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match expected {self.dimension}")
        
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            meta['vector_index'] = i
            self.metadata_store.append(meta)
        
        self.vectors = vectors
        print(f"Stored {len(self.metadata_store)} metadata entries")

    def search(self, query_vector: np.ndarray, top_k: int = 10,
               similarity_threshold: float = 0.0) -> List[Dict]:
        """Search using cosine similarity."""
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        else:
            query_vector = query_vector.astype(np.float32)
        
        # Ensure vectors exist
        if not hasattr(self, 'vectors') or len(self.metadata_store) == 0:
            return []
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        search_results = []
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                result = self.metadata_store[idx].copy()
                result['score'] = float(similarities[idx])
                search_results.append(result)
        
        return search_results

    def size(self) -> int:
        return len(self.metadata_store)

    def clear(self):
        self.db = Endee()
        self.metadata_store = []