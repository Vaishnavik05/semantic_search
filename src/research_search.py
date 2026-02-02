"""
Research paper semantic search engine.
"""
import sys
import os
import yaml
import traceback
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

try:
    from pdf_parser import ResearchPaperPDFParser
except ImportError as e:
    ResearchPaperPDFParser = None

try:
    from metadata_extractor import MetadataExtractor
except ImportError as e:
    MetadataExtractor = None

try:
    from citation_extractor import CitationExtractor
except ImportError as e:
    CitationExtractor = None

try:
    from embedding_service import EmbeddingService
except ImportError as e:
    EmbeddingService = None

try:
    from vector_store import VectorStore
except ImportError as e:
    VectorStore = None

from document_processor import DocumentProcessor


class ResearchPaperSearch:
    
    def __init__(self, config_path: str = "config/config.yaml"):
        config_path = Path(config_path)
        if not config_path.exists():
            self.config = {
                'embedding': {'model_name': 'TF-IDF', 'device': 'cpu'},
                'processing': {'chunk_size': 200, 'chunk_overlap': 50},
                'endee': {'db_path': 'vector_store/papers.db'},
                'search': {'top_k': 10, 'similarity_threshold': 0.1}
            }
        else:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        
        self.pdf_parser = ResearchPaperPDFParser() if ResearchPaperPDFParser else None
        self.metadata_extractor = MetadataExtractor() if MetadataExtractor else None
        self.citation_extractor = CitationExtractor() if CitationExtractor else None
        
        self.embedding_service = EmbeddingService(
            model_name=self.config['embedding']['model_name'],
            device=self.config['embedding']['device']
        )
        
        self.vector_store = VectorStore(
            dimension=self.embedding_service.get_dimension(),
            db_path=self.config['endee'].get('db_path')
        )
        
        self.papers = []
        self.document_processor = DocumentProcessor(
            chunk_size=self.config['processing'].get('chunk_size', 200),
            chunk_overlap=self.config['processing'].get('chunk_overlap', 50)
        )
    
    def index_papers(self, directory: str):
        pdf_files = list(Path(directory).rglob("*.pdf")) + list(Path(directory).rglob("*.txt"))
        
        if not pdf_files:
            return
        
        chunks = []
        metadata_list = []
        papers_indexed = []
        
        for pdf_path in tqdm(pdf_files, desc="Processing papers"):
            try:
                parsed = self.pdf_parser.parse_pdf(str(pdf_path))
                if not parsed:
                    continue
                
                filename = pdf_path.name
                paper_metadata = parsed.get('metadata', {})
                paper_metadata['filename'] = filename
                
                sections = parsed.get('sections', {})
                
                if not sections:
                    full_text = parsed.get('text', '')
                    if full_text.strip():
                        sections = {'introduction': full_text}
                
                for section_name, section_text in sections.items():
                    if not section_text or len(section_text.strip()) < 50:
                        continue
                    
                    section_name_clean = section_name.lower().strip()
                    
                    section_chunks = self.document_processor.chunk_text(
                        section_text,
                        preserve_sentences=True
                    )
                    
                    for chunk_text in section_chunks:
                        if len(chunk_text.strip()) < 20:
                            continue
                        
                        chunks.append(chunk_text)
                        
                        chunk_meta = {
                            'paper_title': paper_metadata.get('title', filename),
                            'authors': paper_metadata.get('authors', ['Unknown']),
                            'year': paper_metadata.get('year'),
                            'section': section_name_clean,
                            'content': chunk_text[:500],
                            'filename': filename
                        }
                        metadata_list.append(chunk_meta)
                
                papers_indexed.append(paper_metadata)
                
            except Exception as e:
                continue
        
        if not chunks:
            return
        
        embeddings = self.embedding_service.embed_batch(chunks, show_progress=True)
        self.vector_store.add_vectors(embeddings, metadata_list)
        self.papers = papers_indexed
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict]:
        if similarity_threshold is None:
            similarity_threshold = 0.0
        
        if self.vector_store.size() == 0:
            return []
        
        query_embedding = self.embedding_service.embed_text(query)
        
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k * 3,
            similarity_threshold=similarity_threshold
        )
        
        if filters:
            results = self._apply_filters(results, filters)
        
        results = self._apply_boosting(results)
        
        return results[:top_k]
    
    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        filtered = results
        
        if 'section' in filters and filters['section'] and filters['section'] != 'All Sections':
            section_filter = filters['section'].lower().strip()
            filtered = [r for r in filtered if r.get('section', '').lower() == section_filter]
        
        if 'year_min' in filters and filters['year_min']:
            filtered = [r for r in filtered if r.get('year') and r['year'] >= filters['year_min']]
        
        if 'year_max' in filters and filters['year_max']:
            filtered = [r for r in filtered if r.get('year') and r['year'] <= filters['year_max']]
        
        if 'authors' in filters and filters['authors']:
            author_query = filters['authors'].lower().strip()
            filtered = [
                r for r in filtered
                if any(author_query in str(author).lower() for author in r.get('authors', []))
            ]
        
        return filtered
    
    def _apply_boosting(self, results: List[Dict]) -> List[Dict]:
        for result in results:
            section = result.get('section', '').lower()
            
            if 'abstract' in section:
                result['score'] = result.get('score', 0) * 1.5
            elif 'conclusion' in section:
                result['score'] = result.get('score', 0) * 1.3
            elif 'introduction' in section:
                result['score'] = result.get('score', 0) * 1.2
        
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return results
    
    def get_stats(self) -> Dict:
        stats = {
            'total_papers': len(self.papers),
            'total_chunks': self.vector_store.size(),
            'model_name': self.config['embedding']['model_name'],
            'sections': self.get_available_sections(),
            'papers_by_year': self._count_by_year()
        }
        return stats
    
    def _count_by_year(self) -> Dict[int, int]:
        year_counts = {}
        seen_papers = set()
        
        for meta in self.vector_store.metadata_store:
            paper_title = meta.get('paper_title', '')
            year = meta.get('year')
            
            if paper_title and year and paper_title not in seen_papers:
                seen_papers.add(paper_title)
                year_counts[year] = year_counts.get(year, 0) + 1
        
        return dict(sorted(year_counts.items()))
    
    def get_available_sections(self) -> List[str]:
        sections = set()
        
        for meta in self.vector_store.metadata_store:
            section = meta.get('section', '').lower().strip()
            if section and section != 'unknown':
                sections.add(section)
        
        return sorted(list(sections))