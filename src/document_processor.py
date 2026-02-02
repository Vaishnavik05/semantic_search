"""
Enhanced document processor for research papers.
"""
import os
import re
from typing import List, Dict
from pathlib import Path


class DocumentProcessor:
    """Process and chunk research papers."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, preserve_sentences: bool = True) -> List[str]:
        """Split text into overlapping chunks while preserving sentence boundaries."""
        if preserve_sentences:
            return self._chunk_by_sentences(text)
        else:
            return self._chunk_by_words(text)
    
    def _chunk_by_words(self, text: str) -> List[str]:
        """Split by words with overlap."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Split by sentences while respecting chunk size."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size <= self.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_size
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Handle overlap
                if len(current_chunk) > 1:
                    # Keep last few sentences for overlap
                    overlap_sentences = current_chunk[-(self.chunk_overlap // 50):]
                    current_chunk = overlap_sentences + [sentence]
                    current_size = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_size = sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [c.strip() for c in chunks if c.strip()]