import sys
from pathlib import Path

from src.ingestion import Ingestor
from src.BM25 import SearchEngine

def main():
    DATA_PATH = "vllm-0.10.1" 
    INDEX_PATH = "data/index_vllm"
    CHUNK_SIZE = 1500
    OVERLAP = 150

    engine = SearchEngine()

    if Path(INDEX_PATH).exists():
        print(f"--- Loading existing index from {INDEX_PATH} ---")
        engine.load(INDEX_PATH)
    else:
        print(f"--- Index not found. Starting ingestion of {DATA_PATH} ---")
        ingestor = Ingestor(max_chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        
        try:
            all_texts, all_sources = ingestor.run(DATA_PATH)
        except Exception as e:
            print(f"ERROR during ingestion : {e}")
            sys.exit(1)

        if not all_texts:
            print("No chunks generated. Check your DATA_PATH.")
            return

        print(f"Ingested {len(all_texts)} chunks. Building BM25 index...")
        engine.build_index(all_texts, all_sources)
        
        print(f"Saving index to {INDEX_PATH}...")
        engine.save(INDEX_PATH)

    print("\n" + "="*50)
    query = "How does PagedAttention implement the KV cache?"
    print(f"SEARCHING FOR: '{query}'")
    print("="*50)

    try:
        results = engine.query(query, top_k=3)

        for i, (source, text, score) in enumerate(results, 1):
            print(f"\n[Result #{i}] - Score: {score:.2f}")
            print(f"File: {source.file_path}")
            print(f"Lines: {source.first_character_index} to {source.last_character_index}")
            print("-" * 20)
            print(text[:300].strip() + "...")
            print("-" * 20)

    except Exception as e:
        print(f"ERROR during query: {e}")

if __name__ == "__main__":
    main()