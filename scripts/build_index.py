from pathlib import Path
from app.rag.index import RAG

DEFAULT_DOC = """# ShipSafe AI Docs

This folder is used for the RAG demo.

If you add more `.md` or `.txt` files here, the retrieval quality improves.
Try asking:
- "What is this project?"
- "What endpoints are available?"
- "How does the RAG pipeline work?"
"""

def ensure_docs_folder(docs_dir: str = "docs") -> None:
    d = Path(docs_dir)
    d.mkdir(parents=True, exist_ok=True)

    has_docs = any(p.is_file() and p.suffix.lower() in {".md", ".txt"} for p in d.rglob("*"))
    if not has_docs:
        (d / "README.md").write_text(DEFAULT_DOC, encoding="utf-8")

if __name__ == "__main__":
    ensure_docs_folder("docs")
    RAG.build("docs")
    print("✅ Built FAISS index from ./docs into artifacts/rag/")
