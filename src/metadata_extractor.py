import re
from typing import Dict, List, Optional


class MetadataExtractor:
    
    def __init__(self):
        self.venue_patterns = [
            r'(?:ICML|NeurIPS|ICLR|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL)',
            r'(?:arXiv preprint|arxiv)',
            r'(?:IEEE|ACM)',
        ]
    
    def extract(self, parsed_pdf: Dict, filename: str) -> Dict:
        metadata = parsed_pdf.get('metadata', {})
        
        enhanced_metadata = {
            'title': self._clean_title(metadata.get('title', filename)),
            'authors': self._parse_authors(metadata.get('authors', [])),
            'year': self._extract_year(metadata, parsed_pdf.get('text', '')),
            'abstract': metadata.get('abstract', ''),
            'venue': self._extract_venue(parsed_pdf.get('text', '')),
            'doi': self._extract_doi(parsed_pdf.get('text', '')),
            'num_pages': parsed_pdf.get('num_pages', 0),
            'num_citations': len(parsed_pdf.get('citations', [])),
            'filename': filename
        }
        
        return enhanced_metadata
    
    def _clean_title(self, title: str) -> str:
        if not title:
            return 'Unknown'
        
        title = re.sub(r'\s+', ' ', title).strip()
        title = re.sub(r'^\d+\s*', '', title)
        
        return title
    
    def _parse_authors(self, authors: List[str]) -> List[str]:
        cleaned_authors = []
        
        exclude_keywords = {
            'university', 'institute', 'department', 'college', 'school',
            'india', 'computer', 'science', 'technology', 'engineering',
            'research', 'center', 'laboratory', 'division', 'faculty',
            'abstract', 'introduction', 'paper', 'algorithm', 'abstract',
            'method', 'results', 'conclusion', 'reference', 'appendix',
            'amaravati'
        }
        
        exclude_patterns = [
            r'^[A-Z]$',
            r'^\d+$',
            r'^et\s+al',
            r'^\s*\*\s*$',
            r'^and$',
            r'^or$',
            r'^the$',
        ]
        
        for author in authors:
            cleaned = author.strip().title()
            
            if not cleaned or len(cleaned) < 5 or len(cleaned) > 40:
                continue
                
            if any(keyword in cleaned.lower() for keyword in exclude_keywords):
                continue
                
            if any(re.match(pattern, cleaned, re.IGNORECASE) for pattern in exclude_patterns):
                continue
                
            if cleaned.isupper() or cleaned.islower():
                continue
                
            word_count = len(cleaned.split())
            if word_count < 2 or word_count > 4:
                continue
                
            if all(len(word) == 1 for word in cleaned.split()):
                continue
                
            if cleaned not in cleaned_authors:
                cleaned_authors.append(cleaned)
        
        return cleaned_authors[:20]
    
    def _extract_year(self, metadata: Dict, text: str) -> Optional[int]:
        if metadata.get('year'):
            return metadata['year']
        
        year_matches = re.findall(r'\b(19|20)\d{2}\b', text[:2000])
        
        if year_matches:
            for year_str in year_matches:
                year = int(year_str)
                if 1950 <= year <= 2100:
                    return year
        
        return None
    
    def _extract_venue(self, text: str) -> Optional[str]:
        text_start = text[:3000]
        
        for pattern in self.venue_patterns:
            match = re.search(pattern, text_start, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_doi(self, text: str) -> Optional[str]:
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        match = re.search(doi_pattern, text[:5000])
        
        if match:
            return match.group(0)
        
        return None


def search(
    extractor: MetadataExtractor,
    query: str,
    top_k: int = 10,
    filters: Optional[Dict] = None,
    similarity_threshold: Optional[float] = None
) -> List[Dict]:
    print(f"Searching for: {query}")

    if not extractor.embedding_service.fitted:
        print("Error: No papers indexed yet")
        return []
    
    semantic_results = []
    metadata_results = []
    
    try:
        query_embedding = extractor.embedding_service.embed_text(query)
        semantic_results = extractor.vector_store.search(
            query_embedding, 
            top_k=top_k, 
            similarity_threshold=similarity_threshold or 0.0
        )
    except Exception as e:
        print(f"Semantic search error: {e}")
    
    metadata_results = extractor._search_metadata(query, top_k)
    
    combined_results = extractor._combine_results(semantic_results, metadata_results, top_k)
    
    if filters:
        combined_results = extractor._apply_filters(combined_results, filters)
    
    combined_results = extractor._apply_boosting(combined_results)
    
    return combined_results

def _search_metadata(self, query: str, top_k: int) -> List[Dict]:
    query_lower = query.lower().strip()
    metadata_matches = []
    
    for meta in self.vector_store.metadata_store:
        score = 0.0
        
        authors = meta.get('authors', [])
        for author in authors:
            if query_lower in author.lower():
                score += 0.95
            elif query_lower.split()[0] in author.lower() if query_lower.split() else False:
                score += 0.7
        
        title = meta.get('paper_title', '').lower()
        if query_lower in title:
            score += 0.85
        
        abstract = meta.get('abstract', '').lower()
        if query_lower in abstract:
            score += 0.6
        
        if score > 0:
            result = meta.copy()
            result['score'] = min(score / 100, 1.0)
            result['search_type'] = 'metadata'
            metadata_matches.append(result)
    
    metadata_matches.sort(key=lambda x: x['score'], reverse=True)
    return metadata_matches[:top_k]

def _combine_results(self, semantic_results: List[Dict], metadata_results: List[Dict], top_k: int) -> List[Dict]:
    seen_indices = set()
    combined = []
    
    for result in metadata_results:
        idx = result.get('vector_index', id(result))
        if idx not in seen_indices:
            combined.append(result)
            seen_indices.add(idx)
    
    for result in semantic_results:
        idx = result.get('vector_index', id(result))
        if idx not in seen_indices:
            combined.append(result)
            seen_indices.add(idx)
    
    combined.sort(key=lambda x: x['score'], reverse=True)
    return combined[:top_k]