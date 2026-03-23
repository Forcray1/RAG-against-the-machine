import sys
from src.ingestion import Ingestor


def main():
    DATA_PATH = "vllm-0.10.1" 
    CHUNK_SIZE = 1500
    OVERLAP = 150

    ingestor = Ingestor(max_chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    try:
        all_texts, all_sources = ingestor.run(DATA_PATH)
    except Exception as e:
        print(f"ERROR While ingestion : {e}")
        sys.exit(1)

    if not all_texts:
        print("No chunk generated, check the data path")
        return

    print(f"--- Ingestion done ---")
    print(f"Total numbers of chunk : {len(all_texts)}")

    print("\nExemple of first extracted chunk :")
    print(f"Source : {all_sources[0].file_path} (characters {all_sources[0].first_character_index}-{all_sources[0].last_character_index})")
    print(f"Extract : {all_texts[0][:100]}...")


if __name__ == "__main__":
    main()
