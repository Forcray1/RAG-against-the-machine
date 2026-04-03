*This project has been created as part of the 42 curriculum by mlorenzo.*

# RAG Against the Machine: Will You Answer My Questions?

## Description
This project implements a complete **Retrieval-Augmented Generation (RAG)** system entirely from scratch to interrogate and interact with the **vLLM codebase**. The goal is to ingest raw Python code and Markdown documentation, organize it into a searchable index, retrieve the most scientifically relevant context snippets for a given question, and generate human-readable answers using a localized Large Language Model (Qwen/Qwen3-0.6B).

## Instructions

### Prerequisites
Make sure you have **Python 3.10** (or later) and **uv** installed on your machine.

### Installation
Clone the repository and install all dependencies strictly typed within `pyproject.toml` via the `Makefile`:
```bash
make install
```

### Useful Makefile Commands
- `make run`: Runs the core script mapping.
- `make lint`: Verifies static types and PEP8 normative formatting (using `flake8` & `mypy`).
- `make clean`: Removes caches, `.venv`, `.pytest_cache`, and temporary files.

## System Architecture
The application pipeline acts dynamically across 4 main components:
1. **Knowledge Base Ingestion**: Reads the target repository recursively, filtering ignored folders (`.git`, `__pycache__`, etc.).
2. **Intelligent Chunking Strategy**: Dissects big documents into smaller contexts (max: 2000 characters parameterizable limit) preserving semantic structures.
3. **Retrieving System (BM25)**: Transforms text into arrays of tokens, populates an inverted index and queries the top-k textual matching blocks on demand.
4. **Answer Generation System**: Prompts `Qwen/Qwen3-0.6B` feeding it retrieved code-snippets to accurately formulate reliable technical responses.

## Chunking Strategy
To avoid splitting important functional contexts in the middle, two custom chunkers were written:
- **`PythonChunker`**: Uses Python's internal `ast` (Abstract Syntax Tree) logic to walk through the files and chunk them elegantly by `ClassDef` and `FunctionDef`. Falls back to an overlapping size-based split for massive methods exceeding the length limit.
- **`MdChunker`**: Uses `re` module rules to detect structural boundaries inside Markdown documents (headers: `#`, `##`...) and cuts chunks coherently.

Both chunkers respect a parameterizable `max_chunk_size` (default: 2000 characters) and logic `overlap` to not lose variables cut midway. 

## Retrieval Method
We are utilizing **BM25**, a powerful evolution of the TF-IDF statistical method, operated efficiently via the `bm25s` library. BM25 evaluates the relevancy of files by mapping the occurrences of key terms in queries against their frequencies globally inside the corpus, with special logarithmic care avoiding "keyword-stuffing" bias from long files.

## Performance Analysis
*(Note: To be filled once `evaluate` is complete)*
- **Indexing time**: `[To be recorded]` (< 5 minutes target)
- **Recall@5 (Docs Questions)**: `[To be recorded]` (Target > 55%)
- **Recall@5 (Code Questions)**: `[To be recorded]` (Target > 45%)

## Design Decisions
- **Pydantic**: Heavily utilized for data-validation, effectively avoiding silent runtime typing errors by strictly converting Search Results (`MinimalSource`, `StudentSearchResults`).
- **Python Fire**: Chosen to auto-generate a comprehensive CLI mapping Python methods into callable terminal syntax without `argparse` overhead.
- **UV Package Manager**: Chosen to dramatically speed-up dependencies installations scaling above `pip` restrictions.

## Challenges Faced
- Preserving source location index bounds (`first_character_index`, `last_character_index`) flawlessly post-chunking without off-by-one errors.
- Handling empty, poorly formatted markdown components crashing regex captures.
- Prompting logic implementation to bridge retrieved textual information sequentially inside the strict LLM max-token bounds.

## Resources & AI Usage
- **Documentation**: 
- [Python AST Docs](https://docs.python.org/3/library/ast.html)
- [bm25s Repository](https://github.com/xhluca/bm25s)
- vLLM Architecture overviews.

## Example Usage
You can use the RAG system directly through the CLI:

### 1. Ingestion / Indexing
```bash
uv run python -m src index --max_chunk_size 2000
```
### 2. Live Search Output
```bash
uv run python -m src search "How does PagedAttention implement the KV cache?" --k 5
```
### 3. Evaluating a generated dataset
```bash
uv run python -m src evaluate --student_answer_path data/results.json --dataset_path data/ground_truth.json --k 10
```