"""
Advanced PDF parser specifically for research papers.
"""
import re
import pdfplumber
from typing import Dict, List, Optional
from pathlib import Path


class ResearchPaperPDFParser:
    
    SECTION_KEYWORDS = {
        'abstract': ['abstract'],
        'introduction': ['introduction', 'i. introduction', '1. introduction'],
        'related work': ['related work', 'literature review', 'background', 'prior work'],
        'methods': ['method', 'methodology', 'approach', 'proposed method', 'system design'],
        'experiments': ['experiment', 'experimental', 'setup', 'implementation'],
        'results': ['result', 'findings', 'performance', 'evaluation'],
        'discussion': ['discussion', 'analysis'],
        'conclusion': ['conclusion', 'summary', 'concluding remarks'],
        'future work': ['future work', 'future direction'],
        'references': ['references', 'bibliography']
    }
    
    def __init__(self):
        pass
    
    def parse_pdf(self, filepath: str) -> Dict:
        try:
            with pdfplumber.open(filepath) as pdf:
                full_text = ""
                num_pages = len(pdf.pages)
                
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                
                if not full_text.strip():
                    return None
                
                metadata = self._extract_metadata(full_text, pdf.pages[0] if pdf.pages else None)
                sections = self._detect_sections_aggressive(full_text)
                citations = self._extract_citations(full_text)
                
                return {
                    'text': full_text,
                    'metadata': metadata,
                    'sections': sections,
                    'citations': citations,
                    'num_pages': num_pages
                }
        
        except Exception as e:
            return None
    
    def _extract_metadata(self, text: str, first_page) -> Dict:
        metadata = {
            'title': None,
            'authors': [],
            'year': None,
            'abstract': None,
            'venue': None
        }
        
        lines = text.split('\n')
        if lines:
            potential_title = ' '.join(lines[:3]).strip()
            if len(potential_title) < 200:
                metadata['title'] = potential_title
        
        year_match = re.search(r'\b(19|20)\d{2}\b', text[:2000])
        if year_match:
            metadata['year'] = int(year_match.group())
        
        abstract_match = re.search(
            r'abstract\s*[:\-]?\s*\n(.*?)(?=\n\s*\n|\n\s*\d+\.?\s*introduction)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            metadata['abstract'] = abstract_match.group(1).strip()
        
        metadata['authors'] = self._extract_authors_advanced(text)
        
        return metadata
    
    def _extract_authors_advanced(self, text: str) -> List[str]:
        authors = []
        lines = text.split('\n')
        
        author_keywords = {
            'university', 'institute', 'department', 'college', 'school',
            'india', 'computer', 'science', 'technology', 'engineering',
            'research', 'center', 'laboratory', 'division', 'faculty',
            'abstract', 'introduction', 'paper', 'algorithm'
        }
        
        first_page_lines = lines[:30]
        
        for i, line in enumerate(first_page_lines):
            line_lower = line.lower().strip()
            
            if not line or len(line) < 3 or len(line) > 100:
                continue
            
            if any(keyword in line_lower for keyword in author_keywords):
                continue
            
            if re.match(r'.*\d{4}.*', line):
                continue
            
            words = line.split()
            if len(words) < 2 or len(words) > 5:
                continue
            
            is_valid_author = True
            for word in words:
                if not word[0].isupper() or len(word) < 2:
                    is_valid_author = False
                    break
            
            if is_valid_author:
                authors.append(line.strip())
        
        return authors[:10]
    
    def _detect_sections_aggressive(self, text: str) -> Dict[str, str]:
        """Aggressively detect all sections - split entire PDF into sections."""
        sections = {}
        lines = text.split('\n')
        
        section_markers = []
        
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            
            if len(line_lower) < 3 or len(line_lower) > 100:
                continue
            
            for section_name, keywords in self.SECTION_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in line_lower and len(line_lower) < 50:
                        section_markers.append((i, section_name, line))
                        break
        
        if not section_markers:
            total_lines = len(lines)
            quarter = total_lines // 4
            
            sections['introduction'] = '\n'.join(lines[:quarter]).strip()
            sections['methods'] = '\n'.join(lines[quarter:quarter*2]).strip()
            sections['results'] = '\n'.join(lines[quarter*2:quarter*3]).strip()
            sections['conclusion'] = '\n'.join(lines[quarter*3:]).strip()
            
            return {k: v for k, v in sections.items() if len(v) > 100}
        
        for idx, (line_num, section_name, header) in enumerate(section_markers):
            start = line_num + 1
            
            if idx < len(section_markers) - 1:
                end = section_markers[idx + 1][0]
            else:
                end = len(lines)
            
            section_text = '\n'.join(lines[start:end]).strip()
            
            if section_text and len(section_text) > 50:
                if section_name in sections:
                    sections[section_name] += '\n' + section_text
                else:
                    sections[section_name] = section_text
        
        if not sections:
            sections['introduction'] = text.strip()
        
        return sections
    
    def _extract_citations(self, text: str) -> List[str]:
        citations = []
        
        ref_match = re.search(
            r'(?:references|bibliography)\s*\n(.*)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if ref_match:
            ref_text = ref_match.group(1)
            citation_lines = ref_text.split('\n')
            citations = [line.strip() for line in citation_lines if line.strip()]
        
        return citations[:100]