import sys
from pathlib import Path
from tqdm import tqdm
import time
from llama_cpp import Llama

from src.ingestion import Ingestor
from src.BM25 import SearchEngine
from src.models import RagDataset, StudentSearchResults, MinimalSearchResults
from src.models import StudentSearchResultsAndAnswer, MinimalAnswer
from src.utils import calculate_overlap

DATA_PATH_DEFAULT = "data/raw/vllm-0.10.1"
INDEX_PATH_DEFAULT = "data/processed/bm25_index"


class RagCLI:
    """
    Command-Line Interface for the RAG-against-the-machine project.
    Contains the 6 mandatory commands required by the subject.
    """

    def index(self,
              max_chunk_size: int = 800,
              data_path: str = DATA_PATH_DEFAULT,
              index_path: str = INDEX_PATH_DEFAULT) -> None:
        """
        Index the repository and create a searchable knowledge base.
        """
        print(f"--- Starting ingestion of {data_path} ---")
        t = time.perf_counter()
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
            f = time.perf_counter() - t
            print(f"time: {f:.2f}")
        except Exception as e:
            print(f"ERROR during ingestion: {e}")
            sys.exit(1)

    def search(self,
               query: str,
               k: int = 10,
               index_path: str = INDEX_PATH_DEFAULT) -> None:
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
                       index_path: str = INDEX_PATH_DEFAULT) -> None:
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
                    question_str=q.question,
                    retrieved_sources=retrieved_sources
                )
                all_results.append(minimal_res)

            except Exception as e:
                print(f"\nERROR during query '{q.question_id}': {e}")
                # Append empty results to maintain the alignment if query fails
                all_results.append(MinimalSearchResults(
                    question_id=q.question_id,
                    question_str=q.question,
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
               index_path: str = INDEX_PATH_DEFAULT) -> None:
        """
        Answer a single question with context passed to the LLM.
        """

        # Load quantized GGUF model via llama-cpp (fast CPU inference)
        model_name = "Qwen/Qwen3-0.6B-GGUF"
        filename = "*.gguf"

        try:
            llm = Llama.from_pretrained(
                repo_id=model_name,
                filename=filename,
                n_ctx=1024,
                n_threads=None,  # use all cores
                verbose=False,
            )
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

        t = time.perf_counter()
        # Retrieve top-k relevant chunks and build a context string
        try:
            results = engine.query(query, top_k=k)
            retrieved_sources = [res[0] for res in results]
            context_text = "\n\n".join([text for _, text, _ in results])
        except Exception as e:
            print(f"ERROR during query: {e}")
            return

        all_scores = [score for _, _, score in results]
        top_score = all_scores[0]
        avg_score = sum(all_scores) / len(all_scores)
        dominance_ratio = top_score / avg_score if avg_score > 0 else 0

        # Reject if the average score is too low
        # or if one rare term inflates the top score far above the others
        if avg_score < 5 or dominance_ratio > 1.5:
            source, text, score = results[0]
            print("\nNo relevant informations found, "
                  "Can't answer efficiently on this question")
            print(f"Top score: {top_score:.2f} | "
                  f"Avg score: {avg_score:.2f} | "
                  f"Dominance ratio: {dominance_ratio:.2f}")
            print(f"\n[Result #1] - Score: {score:.2f}")
            print(f"File: {source.file_path}")
            print(f"Indices: {source.first_character_index} "
                  f"to {source.last_character_index}")
            print("-" * 20)
            print(text[:300].strip() + "...")
            print("-" * 20)
            return

        # Limit context size to avoid bloated prompts and slow prefill
        max_context_chars = 700
        if len(context_text) > max_context_chars:
            context_text = context_text[:max_context_chars]

        # Build Qwen3 prompt manually — pre-fill empty <think> block
        # so the model skips thinking and answers directly
        prompt = (
            f"<|im_start|>system\n"
            f"You are a helpful assistant. Answer the question concisely "
            f"in 1-3 sentences based ONLY on the following context.\n\n"
            f"CONTEXT:\n{context_text}\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n"
        )

        # Generate the answer using llama-cpp
        try:
            response = llm.create_completion(
                prompt=prompt,
                max_tokens=48,
                temperature=0.0,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            # Type ignore since Llama-cpp type hints for choices can be tricky
            generated_text = (
                response["choices"][0]["text"].strip()  # type: ignore
            )
        except Exception as e:
            print(f"ERROR during answer generation: {e}")
            f = time.perf_counter() - t
            print(f"time: {f:.2f}")
            return

        # Wrap everything in the structured output model and print
        answer_obj = MinimalAnswer(
            question_id="single_query",
            question_str=query,
            retrieved_sources=retrieved_sources,
            answer=generated_text
        )

        final_output = StudentSearchResultsAndAnswer(
            search_results=[answer_obj],
            k=k
        )

        print(final_output.model_dump_json(indent=2))
        f = time.perf_counter() - t
        print(f"time: {f:.2f}")

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str = "output/answer_dataset",
                       data_path: str = DATA_PATH_DEFAULT,
                       max_context_chars: int = 700) -> None:
        """
        Generate answers from search results and output structured JSON.
        Expects a StudentSearchResults JSON (output of search_dataset).
        """
        results_file = Path(student_search_results_path)
        if not results_file.exists():
            print(f"ERROR: File not found at {student_search_results_path}")
            return

        try:
            student_results = StudentSearchResults.model_validate_json(
                results_file.read_text(encoding="utf-8")
            )
        except Exception as e:
            print(f"ERROR: Failed to parse student search results JSON: {e}")
            return

        print(f"Loaded {len(student_results.search_results)} questions "
              f"from {student_search_results_path}")

        # Load quantized GGUF model via llama-cpp (fast CPU inference)
        try:
            llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen3-0.6B-GGUF",
                filename="*.gguf",
                n_ctx=1024,
                n_threads=None,
                verbose=False,
            )
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            return

        all_answers = []
        t_total = time.perf_counter()

        for item in tqdm(student_results.search_results, desc="Answering"):
            t_q = time.perf_counter()

            # Build context by reading source file slices
            context_parts = []
            for src in item.retrieved_sources:
                src_file = Path(data_path) / src.file_path
                try:
                    file_text = src_file.read_text(encoding="utf-8",
                                                   errors="ignore")
                    chunk = file_text[src.first_character_index:
                                      src.last_character_index]
                    context_parts.append(chunk)
                except Exception:
                    continue
            context_text = "\n\n".join(context_parts)
            if len(context_text) > max_context_chars:
                context_text = context_text[:max_context_chars]

            # Build Qwen3 prompt with pre-filled empty <think> block
            prompt = (
                f"<|im_start|>system\n"
                f"You are a helpful assistant. Answer the question concisely "
                f"in 1-3 sentences based ONLY on the following context.\n\n"
                f"CONTEXT:\n{context_text}\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n{item.question_str}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n"
            )

            try:
                response = llm.create_completion(
                    prompt=prompt,
                    max_tokens=48,
                    temperature=0.0,
                    stop=["<|im_end|>", "<|endoftext|>"],
                )
                answer_text = (
                    response["choices"][0]["text"].strip()  # type: ignore
                )
            except Exception as e:
                print(f"\nERROR generating answer for "
                      f"'{item.question_id}': {e}")
                answer_text = ""

            tqdm.write(f"  [{item.question_id}] "
                       f"{time.perf_counter() - t_q:.2f}s")
            all_answers.append(MinimalAnswer(
                question_id=item.question_id,
                question_str=item.question_str,
                retrieved_sources=item.retrieved_sources,
                answer=answer_text,
            ))

        print(f"\nTotal time: {time.perf_counter() - t_total:.2f}s "
              f"({len(all_answers)} questions)")

        # Wrap and save results
        final_output = StudentSearchResultsAndAnswer(
            search_results=all_answers,
            k=student_results.k,
        )

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / Path(student_search_results_path).name
        try:
            out_file.write_text(final_output.model_dump_json(indent=4),
                                encoding="utf-8")
            print(f"\nSaved student_search_results_and_answer to {out_file}")
        except Exception as e:
            print(f"ERROR: Failed to save results: {e}")

    def evaluate(self,
                 student_answer_path: str,
                 dataset_path: str,
                 k: int,
                 max_context_length: int = 2000) -> None:
        """
        Evaluate search results quality against ground truth (Recall@k).
        """
        results_file = Path(student_answer_path)
        if not results_file.exists():
            print(f"ERROR: File not found at {student_answer_path}")
            return

        valid_file = Path(dataset_path)
        if not valid_file.exists():
            print(f"ERROR: File not found at {dataset_path}")
            return

        try:
            student_results = StudentSearchResults.model_validate_json(
                results_file.read_text(encoding="utf-8")
            )
        except Exception as e:
            print(f"ERROR: Failed to parse student search results JSON: {e}")
            return

        try:
            valid_results = RagDataset.model_validate_json(
                valid_file.read_text(encoding="utf-8")
            )
        except Exception as e:
            print(f"ERROR: Failed to parse answered questions search "
                  f"results JSON: {e}")
            return

        student_results_dict = {res.question_id: res for
                                res in student_results.search_results}

        nb_trouvees_total: float = 0.0
        total_attendu = 0
        for question in valid_results.rag_questions:
            correct = getattr(question, "sources", [])
            student_answer = student_results_dict.get(question.question_id)
            # Ne récupérer que les k premiers résultats demandés
            if student_answer:
                search = student_answer.retrieved_sources[:k]
            else:
                search = []

            nb_trouvees_question = 0

            for G in correct:
                for R in search:
                    if calculate_overlap(R, G) >= 0.05:
                        nb_trouvees_question += 1
                        break

            # Formule pour Recall@k sur une question :
            if len(correct) > 0:
                recall_question = nb_trouvees_question / len(correct)
                nb_trouvees_total += recall_question
                total_attendu += 1

        if total_attendu > 0:
            recall = nb_trouvees_total / total_attendu
            print(f"Recall@{k} global : "
                  f"{recall * 100}%")
        else:
            print("Aucune question trouvée dans le dataset !")
