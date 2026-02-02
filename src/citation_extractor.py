"""
Extract and analyze citations from research papers.
"""
import re
from typing import List, Dict, Optional
from collections import defaultdict


class CitationExtractor:
    
    def __init__(self):
        self.citation_graph = defaultdict(list)
        self.paper_citations = {}
    
    def extract_citations(self, text: str, paper_id: str) -> List[str]:
        citations = []
        
        ref_match = re.search(
            r'(?:references|bibliography)\s*\n(.*)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if ref_match:
            ref_text = ref_match.group(1)
            citation_lines = ref_text.split('\n')
            
            for line in citation_lines[:100]:
                line = line.strip()
                if len(line) > 20 and not line.startswith('Page'):
                    citations.append(line)
        
        self.paper_citations[paper_id] = citations
        return citations
    
    def build_citation_graph(self, papers: List[Dict]):
        self.citation_graph.clear()
        
        for paper in papers:
            paper_id = paper.get('filename', paper.get('title', ''))
            citations = paper.get('citations', [])
            
            for citation in citations:
                for other_paper in papers:
                    other_id = other_paper.get('filename', other_paper.get('title', ''))
                    other_title = other_paper.get('title', '')
                    
                    if other_id != paper_id and other_title and other_title.lower() in citation.lower():
                        self.citation_graph[paper_id].append(other_id)
    
    def get_citing_papers(self, paper_id: str) -> List[str]:
        citing_papers = []
        for citing_id, cited_ids in self.citation_graph.items():
            if paper_id in cited_ids:
                citing_papers.append(citing_id)
        return citing_papers
    
    def get_cited_papers(self, paper_id: str) -> List[str]:
        return self.citation_graph.get(paper_id, [])
    
    def get_citation_count(self, paper_id: str) -> int:
        return len(self.get_citing_papers(paper_id))
    
    def get_most_cited_papers(self, top_k: int = 10) -> List[tuple]:
        citation_counts = {}
        
        for paper_id in self.paper_citations.keys():
            citation_counts[paper_id] = self.get_citation_count(paper_id)
        
        sorted_papers = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_papers[:top_k]
    
    def get_graph_stats(self) -> Dict:
        return {
            'total_papers': len(self.paper_citations),
            'total_edges': sum(len(cited) for cited in self.citation_graph.values()),
            'avg_citations_per_paper': sum(len(cit) for cit in self.paper_citations.values()) / max(len(self.paper_citations), 1),
            'most_cited': self.get_most_cited_papers(5)
        }