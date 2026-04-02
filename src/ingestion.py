import pathlib
from typing import List, Tuple

from src.chunkers import PythonChunker, MdChunker
from src.models import MinimalSource

class Ingestor:
    """
    Handles file ingestion and dispatching to correct chunkers based on extension.
    It builds a combined and flattened list of all textual snippets and source data.
    """
    def __init__(self, max_chunk_size: int = 2000, overlap: int = 200):
        self.py_chunker = PythonChunker(max_chunk_size, overlap)
        self.md_chunker = MdChunker(max_chunk_size, overlap)
        self.exclude_dirs = {".git", "__pycache__", "build", "dist", "venv", ".venv", "tests"}

    def is_ignored(self, path: pathlib.Path) -> bool:
        """
        Determines if the current file path overlaps with an ignored directory.
        """
        return any(part in self.exclude_dirs for part in path.parts)

    def run(self, root_dir: str) -> Tuple[List[str], List["MinimalSource"]]:
        """
        Loops over a directory structure, delegating to the precise chunkers.
        Iterates files recursively unless ignored.
        """
        all_texts = []
        all_sources = []
        root = pathlib.Path(root_dir)

        if not root.exists():
            print(f"ERROR: Directory {root_dir} does not exist.")
            return [], []

        for file_path in root.rglob("*"):

            if not file_path.is_file() or self.is_ignored(file_path):
                continue

            display_path = str(file_path.relative_to(root))
            chunks = []
            try:
                if file_path.suffix == ".py":
                    content = file_path.read_text(encoding="utf-8")
                    chunks = self.py_chunker.chunk(display_path, content)
                    
                elif file_path.suffix == ".md":
                    content = file_path.read_text(encoding="utf-8")
                    chunks = self.md_chunker.chunk(display_path, content)

            except Exception as e:
                print(f"ERROR while reading {file_path}: {e}")
                continue

            for source, text in chunks:
                if text.strip():
                    all_texts.append(text)
                    all_sources.append(source)
        
        return all_texts, all_sources