import bm25s
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

from src.models import MinimalSource

class SearchEngine:
    """
    BM25 implementation handling tokenization, document indexing, saving, and retrieval.
    This links raw string segments with their Pydantic source definitions.
    """
    def __init__(self):
        self.retriever: Optional[bm25s.BM25] = None
        self.sources: List[MinimalSource] = []

    def _tokenize(self, texts: List[str]):
        """
        Cleans and splits texts recursively for building the 
        """
        return bm25s.tokenize(
            texts, 
            lower=True,
            show_progress=True
        )

    def build_index(self, texts: List[str], sources: List[MinimalSource]) -> None:
        """
        Constructs the search index with the provided string documents mapping to 
        metadata schemas.
        """
        self.sources = sources
        corpus_tokens = self._tokenize(texts)

        self.retriever = bm25s.BM25(corpus=texts)
        self.retriever.index(corpus_tokens)
        print(f"Indexation terminée : {len(texts)} chunks indexés.")

    def save(self, path: str) -> None:
        """
        Dumps the built BM25 index and properties to a specified local directory.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.retriever:
            # bm25s creates its own files in this dir
            self.retriever.save(save_dir)

        # Save list of MinimalSource objects
        with open(save_dir / "sources.pkl", "wb") as f:
            pickle.dump(self.sources, f)
        print(f"Index et sources sauvegardés dans {path}")

    def load(self, path: str) -> None:
        """
        Reloads index matrices and minimal sources from serialized directory storage.
        """
        load_dir = Path(path)
        if not load_dir.exists():
            raise FileNotFoundError(f"Le dossier d'index {path} n'existe pas.")

        # Load BM25 index
        self.retriever = bm25s.BM25.load(load_dir, load_corpus=True)

        # Load sources
        with open(load_dir / "sources.pkl", "rb") as f:
            self.sources = pickle.load(f)
        print(f"Index chargé : {len(self.sources)} chunks prêts.")

    def query(self, user_query: str, top_k: int = 5) -> List[Tuple[MinimalSource, str, float]]:
        """
        Queries the engine for matches retrieving both the text document, origin,
        and its BM25 matching coefficient.
        """
        if not self.retriever or not self.sources:
            raise ValueError("L'index n'est pas initialisé ou chargé.")

        # 1. Tokenisation de la requête
        query_tokens = self._tokenize([user_query])

        # 2. Récupération des résultats (format tuple ou objet)
        results = self.retriever.retrieve(query_tokens, k=top_k)

        # 3. Extraction des matrices
        if isinstance(results, tuple):
            indices_matrice, scores_matrice = results
        else:
            indices_matrice = getattr(results, 'indices', results.get('indices'))
            scores_matrice = getattr(results, 'scores', results.get('scores'))

        indices_top = indices_matrice[0]
        scores_top = scores_matrice[0]

        final_results = []
        for i in range(len(indices_top)):
            # Extraction de l'index (gestion du format dictionnaire si besoin)
            val = indices_top[i]
            idx = int(val.get('id', val.get('index', 0))) if isinstance(val, dict) else int(val)
            
            score = float(scores_top[i])
            source = self.sources[idx]

            raw_content = self.retriever.corpus[idx]
            
            # Si le corpus est stocké sous forme de liste de tokens, on les rejoint
            if isinstance(raw_content, list):
                text = " ".join(raw_content)
            else:
                text = str(raw_content)
            
            final_results.append((source, text, score))

        return final_results
