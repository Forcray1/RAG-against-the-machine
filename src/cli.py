import re
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from src.ingestion import Ingestor
from src.BM25 import SearchEngine
from src.models import RagDataset, StudentSearchResults, MinimalSearchResults
from src.models import StudentSearchResultsAndAnswer, MinimalAnswer

DATA_PATH_DEFAULT = "data/raw/vllm-0.10.1"
INDEX_PATH_DEFAULT = "data/processed/bm25_index"


class RagCLI:
    """
    Command-Line Interface for the RAG-against-the-machine project.
    Contains the 6 mandatory commands required by the subject.
    """

    def index(self,
              max_chunk_size: int = 2000,
              data_path: str = DATA_PATH_DEFAULT,
              index_path: str = INDEX_PATH_DEFAULT):
        """
        Index the repository and create a searchable knowledge base.
        """
        print(f"--- Starting ingestion of {data_path} ---")
        ingestor = Ingestor(max_chunk_size=max_chunk_size, overlap=150)
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

    def search(self,
               query: str,
               k: int = 10,
               index_path: str = INDEX_PATH_DEFAULT):
        """
        Search for a single query using the indexed documents.
        """
        engine = SearchEngine()
        if not Path(index_path).exists():
            print(f"ERROR: Index not found at {index_path}. "
                  f"Please run the 'index' command first.")
            return

        print(f"--- Loading existing index from {index_path} ---")
        engine.load(index_path)
        print(f"SEARCHING FOR: '{query}'")

        try:
            results = engine.query(query, top_k=k)
            for i, (source, text, score) in enumerate(results, 1):
                print(f"\n[Result #{i}] - Score: {score:.2f}")
                print(f"File: {source.file_path}")
                print(f"Indices: {source.first_character_index} "
                      f"to {source.last_character_index}")
                print("-" * 20)
                print(text[:300].strip() + "...")
                print("-" * 20)
        except Exception as e:
            print(f"ERROR during query: {e}")

    def search_dataset(self,
                       dataset_path: str,
                       k: int,
                       save_directory: str,
                       index_path: str = INDEX_PATH_DEFAULT):
        """
        Process multiple questions from a JSON dataset and output
        search results.
        """
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            print(f"ERROR: Dataset not found at {dataset_path}")
            return

        engine = SearchEngine()
        if not Path(index_path).exists():
            print(f"ERROR: Index not found at {index_path}. "
                  f"Please run 'index' first.")
            return

        print(f"--- Loading index from {index_path} ---")
        engine.load(index_path)

        print(f"--- Loading dataset from {dataset_path} ---")
        try:
            dataset_content = dataset_file.read_text(encoding="utf-8")
            dataset = RagDataset.model_validate_json(dataset_content)
        except Exception as e:
            print(f"ERROR: Failed to parse dataset JSON: {e}")
            return

        all_results = []

        # Process with a progress bar
        for q in tqdm(dataset.rag_questions, desc="Searching queries"):
            try:
                # Query the engine
                results = engine.query(q.question, top_k=k)

                # Extract only the MinimalSource object from the
                # engine's tuple (source, text, score)
                retrieved_sources = [res[0] for res in results]

                # Format as the subject requires
                minimal_res = MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=retrieved_sources
                )
                all_results.append(minimal_res)

            except Exception as e:
                print(f"\nERROR during query '{q.question_id}': {e}")
                # Append empty results to maintain the alignment if query fails
                all_results.append(MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=[]
                ))

        # Wrap in the main output model
        final_output = StudentSearchResults(
            search_results=all_results,
            k=k
        )

        # Save to directory
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        out_file = save_dir / dataset_file.name
        try:
            # We dump the fully compliant Pydantic object
            out_file.write_text(final_output.model_dump_json(indent=4),
                                encoding="utf-8")
            print(f"\nSaved student_search_results to {out_file}")
        except Exception as e:
            print(f"ERROR: Failed to save results: {e}")

    def answer(self,
               query: str,
               k: int = 10,
               index_path: str = INDEX_PATH_DEFAULT):
        """
        Answer a single question with context passed to the LLM.
        """

        # Load the LLM model and tokenizer
        model_name = "Qwen/Qwen3-0.6B"

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            engine = SearchEngine()
        except Exception as e:
            print(f"ERROR: Failed to initialize models: {e}")
            return

        # Load BM25 index from disk
        if not Path(index_path).exists():
            print(f"ERROR: Index not found at {index_path}. "
                  f"Please run the 'index' command first.")
            return

        print(f"--- Loading existing index from {index_path} ---")
        engine.load(index_path)
        print(f"SEARCHING FOR: '{query}'")

        # Retrieve top-k relevant chunks and build a context string
        try:
            results = engine.query(query, top_k=k)
            retrieved_sources = [res[0] for res in results]
            context_text = "\n\n".join([text for _, text, _ in results])
        except Exception as e:
            print(f"ERROR during query: {e}")
            return

        # Limit context size to avoid bloated prompts
        max_context_chars = 3000
        if len(context_text) > max_context_chars:
            context_text = context_text[:max_context_chars]

        # Build the prompt with retrieved context
        prompt = ("You are a helpful assistant. Answer the question concisely "
                  "in 1-3 sentences based ONLY on the following context.\n\n"
                  f"CONTEXT:\n{context_text}\n\n"
                  f"QUESTION: {query}\n\nANSWER:")

        # Generate the answer using the LLM
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            stop_ids = tokenizer("\nAnswer:",
                                 add_special_tokens=False).input_ids

            class StopOnTokens(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    ids = input_ids[0].tolist()
                    if len(ids) >= len(stop_ids):
                        return ids[-len(stop_ids):] == stop_ids
                    return False

            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                repetition_penalty=1.3,
                stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
            )
            raw_text = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=False
            )
            # Strip Qwen3 thinking block to avoid duplicate answer
            generated_text = re.sub(
                r'<think>.*?</think>', '', raw_text, flags=re.DOTALL
            )
            # Remove remaining special tokens
            generated_text = re.sub(r'<\|[^|]*\|>', '', generated_text)
            # Cut off at any repeated "Answer:" or "Question:" block
            generated_text = re.split(
                r'\n\s*(?:Answer:|Question:)', generated_text
            )[0]
            generated_text = generated_text.replace('\\n', '\n').strip()
        except Exception as e:
            print(f"ERROR during answer generation: {e}")
            return

        # Wrap everything in the structured output model and print
        answer_obj = MinimalAnswer(
            question_id="single_query",
            question=query,
            retrieved_sources=retrieved_sources,
            answer=generated_text
        )

        final_output = StudentSearchResultsAndAnswer(
            search_results=[answer_obj],
            k=k
        )

        print(final_output.model_dump_json(indent=2))

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str,
                       data_path: str = DATA_PATH_DEFAULT,
                       max_context_chars: int = 3000):
        """
        Generate answers from search results and output structured JSON.
        """
        # Vérifier que le fichier student_search_results_path existe
        # Si non -> print erreur et return
        # Lire et parser le JSON en StudentSearchResults
        # (model_validate_json), gérer l'exception si le JSON est invalide
        # Afficher combien de questions ont été chargées (comme le sujet)
        # Charger le tokenizer et le modèle LLM (même modèle que answer)
        # Gérer l'exception si le chargement échoue
        # Préparer le StoppingCriteria (même logique que answer)
        # pour stopper la génération sur "\nAnswer:"
        # Boucle sur chaque item dans student_results.search_results
        # avec tqdm pour la barre de progression
        # Pour chaque item, construire le contexte texte :
        #   - Pour chaque source dans item.retrieved_sources :
        #     * Construire le chemin du fichier (data_path / file_path)
        #     * Lire le fichier et extraire les caractères
        #        [first_character_index : last_character_index]
        #     * Ajouter ce chunk à une liste context_parts
        #   - Joindre les parties avec "\n\n"
        #   - Tronquer à max_context_chars si trop long
        # Construire le prompt (même structure que answer) :
        #   "You are a helpful assistant. Answer concisely..."
        #    + CONTEXT + QUESTION + ANSWER:
        # Tokenizer le prompt, générer avec model.generate()
        #   Décoder, nettoyer le bloc <think>...</think>,
        #   supprimer les tokens spéciaux restants,
        #   couper au premier "\nAnswer:" ou "\nQuestion:"
        #  Construire un objet MinimalAnswer avec :
        #   question_id, question, retrieved_sources, answer
        #   L'ajouter à une liste all_answers
        # Wrapper all_answers dans StudentSearchResultsAndAnswer
        # avec k = student_results.k
        # Créer le répertoire save_directory (mkdir parents=True)
        # Sauvegarder le JSON dans save_directory / nom_du_fichier_source
        # Afficher confirmation : "Saved student_search_results_and_answer to <path>"
        pass

    def evaluate(self,
                 student_answer_path: str,
                 dataset_path: str,
                 k: int,
                 max_context_length: int = 2000):
        """
        Evaluate search results quality against ground truth (Recall@k).
        """
        print("[TODO] Implement evaluate (Recall@k overlapping characters)")
        print(f"  Student answers: {student_answer_path}")
        print(f"  Ground truth: {dataset_path}")
        print(f"  k: {k}, max length: {max_context_length}")
