import pytest
import numpy as np

from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.document_processor import DocumentProcessor
from src.metadata_extractor import MetadataExtractor


@pytest.fixture
def embedding_service():
    return EmbeddingService()


@pytest.fixture
def vector_store():
    return VectorStore(dimension=768)


def test_embedding_service(embedding_service):
    texts = ["This is a research paper about machine learning", 
             "Transformers for NLP", 
             "Deep learning basics"]
    
    embeddings = embedding_service.embed_batch(texts, show_progress=False)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == embedding_service.get_dimension()
    
    embedding = embedding_service.embed_text(texts[0])
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == embedding_service.get_dimension()


def test_vector_store():
    store = VectorStore(dimension=768)
    
    vectors = np.random.rand(10, 768).astype(np.float32)
    metadata = [
        {
            'content': f'Paper {i}',
            'paper_title': f'Research Paper {i}',
            'authors': ['Author A', 'Author B'],
            'year': 2020 + i,
            'section': 'abstract'
        }
        for i in range(10)
    ]
    
    store.add_vectors(vectors, metadata)
    assert store.size() == 10
    
    query = np.random.rand(768).astype(np.float32)
    results = store.search(query, top_k=5)
    
    assert len(results) <= 5
    assert all('score' in r for r in results)


def test_metadata_extractor():
    extractor = MetadataExtractor()
    
    parsed_pdf = {
        'metadata': {
            'title': 'Attention Is All You Need',
            'authors': ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
            'year': 2017,
            'abstract': 'The Transformer model...'
        },
        'text': 'Full text with 2017',
        'num_pages': 15,
        'citations': []
    }
    
    metadata = extractor.extract(parsed_pdf, 'paper.pdf')
    
    assert metadata['year'] == 2017
    assert len(metadata['authors']) > 0
    assert metadata['filename'] == 'paper.pdf'


def test_document_chunking():
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = processor.chunk_text(text, preserve_sentences=False)
    
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])