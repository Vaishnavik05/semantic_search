from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingService:
    def __init__(self, model_name: str = "TF-IDF", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.vectorizer = TfidfVectorizer(max_features=768, ngram_range=(1, 2))
        self.dimension = 768
        self.fitted = False

    def embed_text(self, text: str) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Vectorizer not fitted. Index papers first.")
        embedding = self.vectorizer.transform([text]).toarray()[0]
        if len(embedding) < 768:
            embedding = np.pad(embedding, (0, 768 - len(embedding)))
        else:
            embedding = embedding[:768]
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = 16, show_progress: bool = True) -> np.ndarray:
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
        embeddings = self.vectorizer.transform(texts).toarray()
        if embeddings.shape[1] < 768:
            padding = np.zeros((embeddings.shape[0], 768 - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
        else:
            embeddings = embeddings[:, :768]
        return embeddings.astype(np.float32)

    def get_dimension(self) -> int:
        return self.dimension