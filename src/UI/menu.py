from src.cli import RagCLI
from src.UI.benchmark_throughput import test_warm_retrieval
import time


def menu() -> None:
    rag = RagCLI()
    index = False
    while True:
        print("\033[2J\033[H", end="")
        print("\n\nWhat function do you want to test?")
        if not index:
            print("\n1. Make Index")
        else:
            print("\n1. Remake Index")
        print("2. Search")
        print("3. Search_dataset")
        print("4. Answer")
        print("5. Answer_dataset")
        print("6. Evaluate the recall@k")
        print("7. Warm retrieval throughput verification")
        print("8. Quit")
        t = input("Choice: ")
        if t == "1":
            print("Starting the indexing (equivalent of "
                  "'python3 -m src index')")
            rag.index()
            index = True
            time.sleep(2)
        elif t == "2":
            while True:
                print("\033[2J\033[H", end="")
                print("\n\n2. Search")
                print("\n1. Question from the subject")
                print("2. Personnalised question")
                print("3. Back to the main menu")
                b = input("Choice: ")
                if b == "1":
                    rag.search("How to configure OpenAI server?", 5)
                    time.sleep(2)
                    break
                elif b == "2":
                    query = input("What's the question?\n")
                    rag.search(query, 5)
                    time.sleep(2)
                    break
                elif b == "3":
                    break
                else:
                    print(f"{t} isn't a valid option")
                    time.sleep(2)
        elif t == "3":
            while True:
                print("\033[2J\033[H", end="")
                print("\n\n3. Search_dataset")
                print("1. Search the docs dataset")
                print("2. Search the code dataset")
                print("3. Other file")
                print("4. Back to the main menu")
                d = input("Choice: ")
                if d == "1":
                    rag.search_dataset(
                        "datasets_public/public/UnansweredQuestions/"
                        "dataset_docs_public.json",
                        10,
                        "output/search_results"
                        )
                elif d == "2":
                    rag.search_dataset(
                        "datasets_public/public/UnansweredQuestions/"
                        "dataset_code_public.json",
                        10,
                        "output/search_results"
                        )
                elif d == "3":
                    file_dataset = input("Enter the file path: ")
                    rag.search_dataset(
                        file_dataset,
                        10,
                        "output/search_results"
                        )
                elif d == "4":
                    break
                else:
                    print(f"{d} isn't a valid option")
                    time.sleep(2)
        elif t == "4":
            while True:
                print("\033[2J\033[H", end="")
                print("\n\n4. Answer")
                print("\n1. Question from the subject")
                print("2. Personnalised question")
                print("3. Back to the main menu")
                b = input("Choice: ")
                if b == "1":
                    rag.answer("How to configure OpenAI server?", 5)
                    time.sleep(2)
                    break
                elif b == "2":
                    query = input("What's the question?\n")
                    rag.answer(query, 5)
                    time.sleep(2)
                    break
                elif b == "3":
                    break
                else:
                    print(f"{t} isn't a valid option")
                    time.sleep(2)
        elif t == "5":
            print("\033[2J\033[H", end="")
            print("\n\n5. Answer_dataset")
            print("1. Answer the docs dataset")
            print("2. Answer the code dataset")
            print("3. Other file")
            print("4. Back to the main menu")
            d = input("Choice: ")
            if d == "1":
                rag.answer_dataset(
                    "datasets_public/public/UnansweredQuestions/"
                    "dataset_docs_public.json",
                    "output/answers"
                    )
            elif d == "2":
                rag.answer_dataset(
                    "datasets_public/public/UnansweredQuestions/"
                    "dataset_code_public.json",
                    "output/answers"
                    )
            elif d == "3":
                file_dataset = input("Enter the file path: ")
                rag.answer_dataset(
                    file_dataset,
                    "output/answers"
                    )
            elif d == "4":
                break
            else:
                print(f"{d} isn't a valid option")
                time.sleep(2)
        elif t == "6":
            print("\033[2J\033[H", end="")
            print("\n\n6. Evaluate the recall@k")
            print("1. Evaluate docs recall@k")
            print("2. Evaluate code recall@k")
            print("3. Evaluate another dataset recall@k")
            print("4. Go back to the main menu")
            e = input("Choice: ")
            if e == "1":
                rag.evaluate(
                    "output/search_results/dataset_code_public.json",
                    "datasets_public/public/AnsweredQuestions"
                    "/dataset_code_public.json",
                    5)
                time.sleep(3)
            elif e == "2":
                rag.evaluate(
                    "output/search_results/dataset_code_public.json",
                    "datasets_public/public/AnsweredQuestions"
                    "/dataset_code_public.json",
                    5)
                time.sleep(3)
            elif e == "3":
                search_results = input("Enter the search result file path: ")
                answered = input("Enter the answered result file path: ")
                rag.evaluate(
                    search_results,
                    answered,
                    5
                )
            elif e == "4":
                break
            else:
                print(f"{e} isn't a valid option")
                time.sleep(2)
        elif t == "7":
            while True:
                print("\033[2J\033[H", end="")
                print("\n\n7. Warm retrieval throughput verification")
                print("\nDo you want to check docs(1) or code(2)?")
                print("Send 3 if you want to go back to the main menu")
                print("All test are with k = 5 according to the subject")
                a = input("Choice: ")
                if a == "1":
                    path = (
                        "datasets_public/public/AnsweredQuestions/"
                        "dataset_docs_public.json"
                        )
                    test_warm_retrieval(
                        path,
                        "data/processed/bm25_index",
                        5)
                    time.sleep(3)
                elif a == "2":
                    path = (
                        "datasets_public/public/AnsweredQuestions/"
                        "dataset_code_public.json"
                        )
                    test_warm_retrieval(
                        path,
                        "data/processed/bm25_index",
                        5)
                    time.sleep(3)
                elif a == "3":
                    break
                else:
                    print(f"{t} isn't a valid option")
                    time.sleep(2)
        elif t == "8":
            break
        else:
            print(f"{t} isn't a valid option")
            time.sleep(2)


if __name__ == "__main__":
    menu()
