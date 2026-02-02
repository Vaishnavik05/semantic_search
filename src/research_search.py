import os
import sys
import yaml
import json
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
                'processing': {'chunk_size': 300, 'chunk_overlap': 75},
                'endee': {'db_path': 'vector_store/papers.db'},
                'search': {'top_k': 10, 'similarity_threshold': 0.15},
                'citations': {'build_citation_graph': True}
            }
        else:
            with open(config_path, 'r') as f:
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
            print(f"No PDF or TXT files found in {directory}")
            return
        
        chunks = []
        metadata_list = []
        papers_indexed = []
        
        for pdf_path in tqdm(pdf_files, desc="Processing papers"):
            try:
                if str(pdf_path).endswith('.txt'):
                    with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    parsed = {
                        'text': text,
                        'metadata': {'title': pdf_path.stem},
                        'sections': {'full_text': text},
                        'citations': [],
                        'num_pages': 1
                    }
                else:
                    parsed = self.pdf_parser.parse_pdf(str(pdf_path))
                
                if not parsed or not parsed.get('text'):
                    continue
                
                paper_metadata = self.metadata_extractor.extract(parsed, pdf_path.name)
                
                if self.citation_extractor:
                    citations = self.citation_extractor.extract_citations(
                        parsed.get('text', ''),
                        pdf_path.name
                    )
                    paper_metadata['citations'] = citations
                    paper_metadata['num_citations'] = len(citations)
                
                papers_indexed.append(paper_metadata)
                
                sections = parsed.get('sections', {})
                if not sections:
                    sections = {'full_text': parsed.get('text', '')}
                
                for section_name, section_text in sections.items():
                    if not section_text or len(section_text.strip()) < 50:
                        continue
                    
                    section_chunks = self.document_processor.chunk_text(
                        section_text,
                        preserve_sentences=self.config['processing'].get('preserve_sentences', True)
                    )
                    
                    for chunk in section_chunks:
                        chunks.append(chunk)
                        metadata_list.append({
                            'content': chunk,
                            'paper_title': paper_metadata['title'],
                            'authors': paper_metadata['authors'],
                            'year': paper_metadata.get('year'),
                            'section': section_name,
                            'filename': pdf_path.name,
                            'venue': paper_metadata.get('venue'),
                            'num_citations': paper_metadata.get('num_citations', 0)
                        })
            
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")
                continue
        
        if not chunks:
            print("No content extracted from papers")
            return
        
        embeddings = self.embedding_service.embed_batch(chunks, show_progress=True)
        self.vector_store.add_vectors(embeddings, metadata_list)
        self.papers = papers_indexed
        
        if self.citation_extractor and self.config.get('citations', {}).get('build_citation_graph', False):
            self.citation_extractor.build_citation_graph(papers_indexed)
            print(f"Built citation graph with {len(papers_indexed)} papers")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict]:
        if similarity_threshold is None:
            similarity_threshold = self.config['search'].get('similarity_threshold', 0.15)
        
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
        
        if self.citation_extractor:
            for result in results:
                filename = result.get('filename', '')
                result['citation_count'] = self.citation_extractor.get_citation_count(filename)
        
        return results[:top_k]
    
    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        filtered = results
        
        if 'section' in filters and filters['section'] and filters['section'] != 'All Sections':
            filtered = [r for r in filtered if r.get('section') == filters['section']]
        
        if 'year_min' in filters and filters['year_min']:
            filtered = [r for r in filtered if r.get('year') and r['year'] >= filters['year_min']]
        
        if 'year_max' in filters and filters['year_max']:
            filtered = [r for r in filtered if r.get('year') and r['year'] <= filters['year_max']]
        
        if 'authors' in filters and filters['authors']:
            author_filter = [a.lower() for a in filters['authors']]
            filtered = [r for r in filtered 
                       if any(author.lower() in ' '.join(r.get('authors', [])).lower() 
                             for author in author_filter)]
        
        return filtered
    
    def _apply_boosting(self, results: List[Dict]) -> List[Dict]:
        for result in results:
            score = result.get('score', 0)
            
            if result.get('section') == 'abstract':
                score *= self.config['search'].get('boost_abstract', 1.5)
            
            if self.config['search'].get('boost_recent', False):
                year = result.get('year')
                threshold = self.config['search'].get('recent_year_threshold', 2020)
                if year and year >= threshold:
                    score *= 1.2
            
            result['score'] = score
        
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return results
    
    def get_stats(self) -> Dict:
        return {
            'total_papers': len(self.papers),
            'total_chunks': self.vector_store.size(),
            'papers_by_year': self._count_by_year(),
            'citation_stats': self.citation_extractor.get_graph_stats() if self.citation_extractor else {}
        }
    
    def _count_by_year(self) -> Dict[int, int]:
        year_counts = {}
        for paper in self.papers:
            year = paper.get('year')
            if year:
                year_counts[year] = year_counts.get(year, 0) + 1
        return dict(sorted(year_counts.items()))
    
    def get_available_sections(self) -> List[str]:
        sections = set()
        for meta in self.vector_store.metadata_store:
            if meta.get('section'):
                sections.add(meta['section'])
        return sorted(list(sections))
    
    def get_citation_network(self, paper_filename: str) -> Dict:
        if not self.citation_extractor:
            return {}
        
        return {
            'citing_papers': self.citation_extractor.get_citing_papers(paper_filename),
            'cited_papers': self.citation_extractor.get_cited_papers(paper_filename),
            'citation_count': self.citation_extractor.get_citation_count(paper_filename)
        }