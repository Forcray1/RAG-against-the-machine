import sys
from pathlib import Path

from src.ingestion import Ingestor
from src.BM25 import SearchEngine

DATA_PATH_DEFAULT = "vllm-0.10.1"
INDEX_PATH_DEFAULT = "data/index_vllm"

class RagCLI:
    """
    Command-Line Interface for the RAG-against-the-machine project.
    Contains the 6 mandatory commands required by the subject.
    """

    def index(self, max_chunk_size: int = 2000, data_path: str = DATA_PATH_DEFAULT, index_path: str = INDEX_PATH_DEFAULT):
        """
		Index the repository and create a searchable knowledge base.
		"""
        print(f"--- Starting ingestion of {data_path} ---")
        overlap = 150
        ingestor = Ingestor(max_chunk_size=max_chunk_size, overlap=overlap)
        engine = SearchEngine()
        
        try:
            all_texts, all_sources = ingestor.run(data_path)
            if not all_texts:
                print("No chunks generated. Check your data path.")
                return

            print(f"Ingested {len(all_texts)} chunks. Building BM25 index...")
            engine.build_index(all_texts, all_sources)
            
            print(f"Saving index to {index_path}...")
            engine.save(index_path)
            print("Ingestion complete!")
        except Exception as e:
            print(f"ERROR during ingestion: {e}")
            sys.exit(1)

    def search(self, query: str, k: int = 10, index_path: str = INDEX_PATH_DEFAULT):
        """
		Search for a single query using the indexed documents.
		"""
        engine = SearchEngine()
        if not Path(index_path).exists():
            print(f"ERROR: Index not found at {index_path}. Please run the 'index' command first.")
            return

        print(f"--- Loading existing index from {index_path} ---")
        engine.load(index_path)
        print(f"SEARCHING FOR: '{query}'")
        
        try:
            results = engine.query(query, top_k=k)
            for i, (source, text, score) in enumerate(results, 1):
                print(f"\n[Result #{i}] - Score: {score:.2f}")
                print(f"File: {source.file_path}")
                print(f"Indices: {source.first_character_index} to {source.last_character_index}")
                print("-" * 20)
                print(text[:300].strip() + "...")
                print("-" * 20)
        except Exception as e:
            print(f"ERROR during query: {e}")

    def search_dataset(self, dataset_path: str, k: int, save_directory: str, index_path: str = INDEX_PATH_DEFAULT):
        """
		Process multiple questions from a JSON dataset and output search results.
		"""
        print(f"[TODO] Implement search_dataset")
        print(f"  Dataset path: {dataset_path}")
        print(f"  k: {k}")
        print(f"  Save directory: {save_directory}")

    def answer(self, query: str, k: int = 10, index_path: str = INDEX_PATH_DEFAULT):
        """
		Answer a single question with context passed to the LLM.
		"""
        print(f"[TODO] Implement answer generation for query: '{query}' with top_{k} context chunks")

    def answer_dataset(self, student_search_results_path: str, save_directory: str):
        """
		Generate answers from search results and output structured JSON.
		"""
        print(f"[TODO] Implement answer_dataset")
        print(f"  Search results path: {student_search_results_path}")
        print(f"  Save directory: {save_directory}")

    def evaluate(self, student_answer_path: str, dataset_path: str, k: int, max_context_length: int = 2000):
        """
		Evaluate search results quality against ground truth (Recall@k).
		"""
        print(f"[TODO] Implement evaluate (Recall@k overlapping characters)")
        print(f"  Student answers: {student_answer_path}")
        print(f"  Ground truth: {dataset_path}")
        print(f"  k: {k}, max length: {max_context_length}")
