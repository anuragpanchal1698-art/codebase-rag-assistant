"""
CLI entrypoint.

Usage:
    python main.py ingest      # clone repo, chunk, embed, store in Qdrant + BM25
    python main.py chat        # interactive terminal chat loop
"""
import sys


def run_ingest():
    from ingestion.loader import clone_repo, load_files
    from ingestion.chunker import chunk_all_files
    from ingestion.embedder import embed_and_store
    from retrieval.bm25_store import load_bm25

    repo_path = clone_repo()
    files = load_files(repo_path)
    chunks = chunk_all_files(files)
    embed_and_store(chunks)
    load_bm25()
    print("\n✅ Ingestion complete. You can now run: python main.py chat")


def run_chat():
    from agent.graph import ask

    print("Codebase RAG Assistant (Grok + hybrid search). Type 'exit' to quit.\n")
    history = []
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        answer, messages = ask(q, chat_history=history)
        history = messages
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("ingest", "chat"):
        print("Usage: python main.py [ingest|chat]")
        sys.exit(1)

    if sys.argv[1] == "ingest":
        run_ingest()
    else:
        run_chat()