# Research Paper Semantic Search Engine

An intelligent semantic search system specifically designed for academic research papers, powered by Endee vector database and scientific language models.

---

## Problem Statement

Researchers face challenges when searching through large collections of academic papers:

- Keyword limitations – traditional search misses semantic meaning  
- Information overload – too many papers to review  
- Context loss – hard to search specific sections  
- Citation tracking difficulties  
- Time-consuming manual screening  

---

## Solution

This system provides:

- Meaning-based semantic search using TF-IDF vectorization
- Metadata extraction (authors, year, venue)
- Section-aware search (abstract, methods, results)
- Citation network analysis
- Fast retrieval with Endee vector database

---

## Use Cases

### 1. Literature Review
```

Query: attention mechanisms for computer vision

```
Finds:
- Vision transformers  
- Self-attention models  
- Spatial attention techniques  

---

### 2. Method Discovery
```

Query: How to train models with limited labeled data?

```
Finds:
- Few-shot learning  
- Semi-supervised learning  
- Data augmentation  

---

### 3. Related Work
```

Query: graph neural networks for molecule prediction

```
Finds research using:
- Molecular graphs  
- Chemical property modeling  
- GNN approaches  

---

### 4. Section-Specific Search
```

Query: evaluation metrics + Section: Methods

```

---

## System Architecture

```

Research Papers (PDF)
↓
Advanced PDF Parser
↓
Metadata Extraction
↓
Section-Aware Chunking
↓
TF-IDF Embeddings (768D)
↓
Endee Vector Database
↓
Semantic Search + Filters
↓
Ranked Results

````

---

## How Endee is Used

### 📦 Vector Storage
- Stores 768-dim embeddings for paper sections
- Supports large-scale datasets

### 🔍 Semantic Similarity Search
- Cosine similarity for relevance ranking
- Fast retrieval (<100ms)

### 🏷️ Metadata Integration
Each vector stores:
- Title, authors, year
- Section type
- Venue
- PDF source

### 📊 Multi-Query Support
Supports:
- Content search  
- Metadata filters  
- Section filters  

### 📈 Scalability
- Incremental indexing
- Efficient memory use

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- Git
- 8GB RAM (16GB recommended)

---

### Step 1: Clone Repo

```bash
git clone <your-repo-url>
cd research_paper_search
````

---

### Step 2: Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac

```bash
python -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Includes:

* endee
* transformers
* torch
* pdfplumber
* spacy
* streamlit

---

### Step 4: Download NLP Model

```bash
python -m spacy download en_core_web_sm
```

---

### Step 5: Verify

```bash
python -c "import endee; from transformers import AutoModel; print('✓ Installation successful')"
```

---

## Add Research Papers

Place PDFs inside:

```
data/
 └── papers/
     ├── paper1.pdf
     ├── paper2.pdf
     └── ...
```

---

## Usage Guide

---

### 🖥️ Method 1: CLI

#### Index Papers

```bash
python src/main.py --mode index --data-dir data/papers
```

---

#### Search

```bash
python src/main.py --mode search --query "transformer architectures" --top-k 5
```

---

#### Advanced Filters

```bash
python src/main.py --mode search \
  --query "few-shot learning" \
  --year-min 2020 \
  --year-max 2024 \
  --section abstract
```

```bash
python src/main.py --mode search \
  --query "attention mechanisms" \
  --author "Vaswani"
```

---

### 🧪 Method 2: Python API

```python
from src.research_search import ResearchPaperSearch

engine = ResearchPaperSearch()

engine.index_papers("data/papers/")

results = engine.search(
    query="graph neural networks",
    top_k=10
)

results = engine.search(
    query="self-supervised learning",
    top_k=5,
    filters={
        'year_min': 2020,
        'year_max': 2024,
        'section': 'methods',
        'authors': ['LeCun', 'Hinton']
    }
)

for r in results:
    print(r['title'], r['score'])
```

---

### 🌐 Method 3: Web Interface

```bash
streamlit run app/streamlit_app.py
```

Features:

* Real-time search
* Filters
* Metadata display
* Citation visualization
* Export to BibTeX

---

### 📓 Method 4: Jupyter Notebook

```bash
jupyter notebook notebooks/research_demo.ipynb
```

---

## Key Features

* Semantic understanding
* SciBERT scientific embeddings
* Metadata extraction
* Section-aware search
* Advanced filtering
* Citation network
* Fast retrieval

---

## Technical Details

### Embedding Model: TF-IDF

- Lightweight statistical text vectorization
- 768-dimensional feature vectors
- N-gram support (unigrams and bigrams)
- Fast indexing and search
- No GPU required

---

### Endee Vector Database

This project uses [Endee](https://github.com/EndeeLabs/endee) for efficient vector storage and similarity search:

- Native vector operations
- Persistent storage support
- Cosine similarity search
- Fast retrieval (<100ms)
- Scalable to large datasets

#### Repository Information

- **Original Repository**: https://github.com/EndeeLabs/endee
- **Forked Repository**: https://github.com/Vaishnavik05/endee
- **Installation**: `pip install endee`

The repository has been forked to study the implementation and understand vector database internals.

### PDF Parsing

- pdfplumber for extraction
- Regex for section detection
- NLP for citations

#### Installation

```bash
pip install endee
```

---

### Citation Network

- Automatic citation extraction from references
- Citation graph construction
- Most-cited papers ranking
- Paper relationship mapping

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

---

## Deployment

### Local

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

---

### Docker

```bash
docker build -t research-search .
docker run -p 8501:8501 -v $(pwd)/data:/app/data research-search
```

---

## Dataset Sources

* ArXiv
* Semantic Scholar
* PubMed Central
* Google Scholar

---

## Conclusion

This project demonstrates a scalable, intelligent semantic search engine for academic research using:

✔ Endee Vector Database
✔ Scientific embeddings
✔ Metadata-aware retrieval

Perfect for AI-driven knowledge discovery.
