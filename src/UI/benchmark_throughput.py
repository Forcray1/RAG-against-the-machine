import time
from pathlib import Path
from src.BM25 import SearchEngine
from src.models import RagDataset


def test_warm_retrieval(dataset_path: str,
                        index_path: str,
                        k: int = 10) -> None:
    # Load the index and dataset (this is the "cold" overhead)
    print("Loading index and dataset...")
    engine = SearchEngine()
    try:
        engine.load(index_path)
    except FileNotFoundError:
        print("ERROR: Index file not found")
        return
    try:
        dataset_content = Path(dataset_path).read_text(encoding="utf-8")
        dataset = RagDataset.model_validate_json(dataset_content)
    except FileNotFoundError:
        print("ERROR: Dataset file not found")
        return
    except Exception as e:
        print(f"ERROR: Invalid dataset format: {e}")
        return

    questions = [q.question for q in dataset.rag_questions]

    # Perform a "cold start" warm-up query
    print("Performing warm-up query...")
    _ = engine.query("What is vLLM?", top_k=k)

    # If the dataset is smaller than 1000, multiply it strictly for benchmark
    if len(questions) < 1000:
        questions = (questions * (1000 // len(questions) + 1))[:1000]
    else:
        questions = questions[:1000]

    # Time the warm retrieval for exactly 1000 questions
    print(f"Benchmarking exactly {len(questions)} queries...")
    start_time = time.perf_counter()

    for q in questions:
        _ = engine.query(q, top_k=k)

    end_time = time.perf_counter()

    total_time = end_time - start_time
    print("-" * 40)
    print(f"Total time for 1000 questions: {total_time:.2f} seconds")
    print(f"Throughput: {1000 / total_time:.2f} queries / second")

    print(f"{total_time:.2f}sec for the warm retrieval throughput")


if __name__ == "__main__":
    DATASET = "datasets_public/public/UnansweredQuestions/"\
              "dataset_docs_public.json"
    INDEX = "data/processed/bm25_index"
    test_warm_retrieval(DATASET, INDEX)
