import fire
from src.cli import RagCLI

def main():
    """
    Main entry point for the RAG-against-the-machine CLI.
    """
    fire.Fire(RagCLI)

if __name__ == "__main__":
    main()
