"""
Streamlit web interface for research paper semantic search.
"""
import streamlit as st
import json
import re
import traceback
from pathlib import Path
import sys
import plotly.graph_objects as go

src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from research_search import ResearchPaperSearch
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error(traceback.format_exc())
    st.stop()

st.set_page_config(
    page_title="Research Paper Search",
    page_icon="📄",
    layout="wide"
)

if 'engine' not in st.session_state:
    st.session_state.engine = ResearchPaperSearch()

if 'indexed' not in st.session_state:
    st.session_state.indexed = False

if 'results' not in st.session_state:
    st.session_state.results = []

if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

def highlight_keywords(text, keywords):
    if not keywords or not text:
        return text
    
    keywords = [k.strip() for k in keywords.split() if k.strip()]
    
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: yellow; padding: 2px;">{keyword}</mark>', text)
    
    return text

st.title("Research Paper Semantic Search")
st.markdown("Powered by TF-IDF Embeddings & Endee Vector Database")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Document Indexing")
    
    data_dir = st.text_input(
        "Papers Directory",
        value="data/papers"
    )
    
    if st.button("Index Papers", use_container_width=True, type="primary"):
        with st.spinner("Indexing papers..."):
            try:
                st.session_state.engine.index_papers(data_dir)
                st.session_state.indexed = True
                st.success("Papers indexed successfully")
                st.rerun()
            except Exception as e:
                st.error(f"Indexing failed: {e}")
    
    st.markdown("---")
    st.subheader("Statistics")
    
    if st.session_state.indexed:
        try:
            stats = st.session_state.engine.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Papers", stats['total_papers'])
            with col2:
                st.metric("Total Chunks", stats['total_chunks'])
            
            if stats.get('papers_by_year'):
                st.subheader("Papers by Year")
                years = list(stats['papers_by_year'].keys())
                counts = list(stats['papers_by_year'].values())
                
                fig = go.Figure(data=[go.Bar(x=years, y=counts)])
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            if stats.get('citation_stats'):
                st.subheader("Citation Network")
                cit_stats = stats['citation_stats']
                st.metric("Total Citation Edges", cit_stats.get('total_edges', 0))
                
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Index papers first to see statistics")
    
    st.markdown("---")
    st.subheader("Search Settings")
    
    top_k = st.slider("Top-K Results", 1, 200, 50)
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.15, 0.05)
    
    st.markdown("---")
    st.subheader("Filters")
    
    use_filters = st.checkbox("Apply Filters")
    
    filters = {}
    if use_filters and st.session_state.indexed:
        available_sections = st.session_state.engine.get_available_sections()
        if available_sections:
            selected_section = st.selectbox(
                "Section",
                options=["All Sections"] + available_sections
            )
            if selected_section != "All Sections":
                filters['section'] = selected_section

if not st.session_state.indexed:
    st.warning("Please index papers first")
    st.info(
        "1. Enter papers directory path in sidebar\n"
        "2. Click 'Index Papers' button\n"
        "3. Wait for indexing to complete\n"
        "4. Start searching"
    )
else:
    st.subheader("Search")
    
    with st.form("search_form"):
        query = st.text_area(
            "Search Query",
            height=100,
            placeholder="Enter search query..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            search_button = st.form_submit_button("Search", use_container_width=True, type="primary")
        with col2:
            clear_button = st.form_submit_button("Clear", use_container_width=True)
    
    if search_button and query:
        with st.spinner("Searching..."):
            try:
                results = st.session_state.engine.search(
                    query,
                    top_k=top_k,
                    filters=filters if filters else None,
                    similarity_threshold=threshold
                )
                st.session_state.results = results
                st.session_state.last_query = query
                st.rerun()
            except Exception as e:
                st.error(f"Search failed: {e}")
    
    if clear_button:
        st.session_state.results = []
        st.session_state.last_query = ""
        st.rerun()
    
    if st.session_state.results:
        st.markdown(f"### Found {len(st.session_state.results)} Results")
        st.markdown("---")
        
        for i, result in enumerate(st.session_state.results, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{i}. {result.get('paper_title', 'Unknown')}**")
                    
                with col2:
                    st.markdown(f"Score: **{result.get('score', 0):.3f}**")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if result.get('authors'):
                        st.markdown(f"Authors: {', '.join(result['authors'][:3])}")
                with col_b:
                    if result.get('year'):
                        st.markdown(f"Year: {result['year']}")
                with col_c:
                    st.markdown(f"Section: {result.get('section', 'N/A')}")
                
                content = result.get('content', '')[:500]
                highlighted = highlight_keywords(content, st.session_state.last_query)
                st.markdown(highlighted, unsafe_allow_html=True)
                
                if result.get('citation_count', 0) > 0:
                    st.caption(f"Citations: {result['citation_count']}")
                
                st.markdown("---")
        
        if st.button("Export Results as JSON"):
            json_str = json.dumps(st.session_state.results, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="search_results.json",
                mime="application/json"
            )
    
    elif query:
        st.info("No results found")