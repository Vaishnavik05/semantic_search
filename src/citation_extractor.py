"""
Extract and process citations from research papers.
"""
import re
from typing import List, Dict


class CitationExtractor:
    """Extract citations and build citation networks."""
    
    def __init__(self):
        pass
    
    def extract_in_text_citations(self, text: str) -> List[str]:
        """Extract in-text citations like [1], [Smith et al. 2020]."""
        citations = []
        
        numeric_citations = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', text)
        for citation in numeric_citations:
            citations.extend(citation.split(','))
        
        author_year = re.findall(r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s+\d{4})\)', text)
        citations.extend(author_year)
        
        return list(set([c.strip() for c in citations]))
    
    def parse_reference(self, reference: str) -> Dict:
        """Parse a single reference into structured format."""
        parsed = {
            'raw': reference,
            'authors': [],
            'title': None,
            'year': None,
            'venue': None
        }
        
        year_match = re.search(r'\b(19|20)\d{2}\b', reference)
        if year_match:
            parsed['year'] = int(year_match.group())
        
        title_match = re.search(r'"([^"]+)"', reference)
        if title_match:
            parsed['title'] = title_match.group(1)
        
        if parsed['year']:
            text_before_year = reference.split(str(parsed['year']))[0]
            potential_authors = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z]\.?)?\b', text_before_year)
            parsed['authors'] = potential_authors[:5]
        
        return parsed
    
    def build_citation_graph(self, papers: List[Dict]) -> Dict:
        """Build citation network from multiple papers."""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        for paper in papers:
            graph['nodes'].append({
                'id': paper.get('title', 'Unknown'),
                'year': paper.get('year'),
                'authors': paper.get('authors', [])
            })
            
            for citation in paper.get('citations', []):
                parsed_citation = self.parse_reference(citation)
                if parsed_citation['title']:
                    graph['edges'].append({
                        'from': paper.get('title'),
                        'to': parsed_citation['title']
                    })
        
        return graph