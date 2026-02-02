import argparse
import sys
from pathlib import Path

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

from research_search import ResearchPaperSearch


def main():
    parser = argparse.ArgumentParser(
        description="Research Paper Semantic Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index papers
  python main.py --mode index --data-dir data/papers
  
  # Basic search
  python main.py --mode search --query "transformer models"
  
  # Advanced search with filters
  python main.py --mode search --query "neural networks" \\
    --year-min 2020 --year-max 2024 --section methods
  
  # Search by author
  python main.py --mode search --query "attention" --author "Vaswani"
  
  # Get statistics
  python main.py --mode stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    index_parser = subparsers.add_parser('index', help='Index research papers')
    index_parser.add_argument(
        '--data-dir',
        type=str,
        default='data/papers',
        help='Directory containing PDF papers'
    )
    
    search_parser = subparsers.add_parser('search', help='Search papers')
    search_parser.add_argument('--query', type=str, required=True, help='Search query')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    search_parser.add_argument('--year-min', type=int, help='Minimum publication year')
    search_parser.add_argument('--year-max', type=int, help='Maximum publication year')
    search_parser.add_argument('--author', type=str, help='Filter by author name')
    search_parser.add_argument('--section', type=str, 
                              choices=['abstract', 'introduction', 'methods', 'results', 'conclusion'],
                              help='Search in specific section')
    search_parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold')
    search_parser.add_argument('--export', type=str, choices=['json', 'csv', 'bibtex'],
                              help='Export results format')
    
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    print("\n Research Paper Semantic Search Engine")
    print("=" * 60)
    
    try:
        engine = ResearchPaperSearch()
        
        if args.mode == 'index':
            index_papers(engine, args.data_dir)
        elif args.mode == 'search':
            search_papers(engine, args)
        elif args.mode == 'stats':
            show_stats(engine)
    
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def index_papers(engine: ResearchPaperSearch, data_dir: str):
    """Index papers from directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f" Directory not found: {data_dir}")
        return
    engine.index_papers(str(data_path))


def search_papers(engine: ResearchPaperSearch, args):
    """Search for papers."""
    print(f"\n Query: '{args.query}'")
    print("-" * 60)
    
    filters = {}
    if args.year_min:
        filters['year_min'] = args.year_min
    if args.year_max:
        filters['year_max'] = args.year_max
    if args.section:
        filters['section'] = args.section
    if args.author:
        filters['authors'] = [args.author]
    
    results = engine.search(
        args.query,
        top_k=args.top_k,
        filters=filters if filters else None,
        similarity_threshold=args.threshold
    )
    
    if not results:
        print("\n No results found\n")
        return
    
    print(f"\n Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{'='*60}")
        print(f"Result {i} | Similarity: {result['score']:.4f}")
        print(f"{'='*60}")
        print(f" Paper: {result['paper_title'][:70]}")
        print(f" Authors: {', '.join(result['authors'][:3])}")
        print(f" Year: {result['year']}")
        print(f" Section: {result['section'].upper()}")
        print(f" Content:\n{result['content'][:400]}...\n")
    
    if args.export:
        export_results(results, args.export)


def show_stats(engine: ResearchPaperSearch):
    """Show engine statistics."""
    stats = engine.get_stats()
    
    print("\n Search Engine Statistics")
    print("-" * 60)
    
    if tabulate:
        basic_stats = [
            ["Total Papers", stats['total_papers']],
            ["Total Chunks", stats['total_chunks']],
            ["Embedding Dimension", stats['embedding_dimension']],
            ["Model", stats['model_name']]
        ]
        print(tabulate(basic_stats, headers=['Metric', 'Value'], tablefmt='grid'))
    else:
        print(f"Total Papers: {stats['total_papers']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Embedding Dimension: {stats['embedding_dimension']}")
        print(f"Model: {stats['model_name']}")
    
    if stats.get('papers_by_year'):
        print("\n Papers by Year:")
        print("-" * 60)
        for year, count in stats['papers_by_year'].items():
            print(f"{year}: {count}")


def export_results(results: list, format_type: str):
    """Export results to file."""
    import json
    import csv
    
    if format_type == 'json':
        filename = 'search_results.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif format_type == 'csv':
        filename = 'search_results.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['paper_title', 'authors', 'year', 'section', 'score'])
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'paper_title': result['paper_title'],
                    'authors': '; '.join(result['authors']),
                    'year': result['year'],
                    'section': result['section'],
                    'score': result['score']
                })
    
    print(f"\n Results exported to: {filename}")


if __name__ == '__main__':
    main()