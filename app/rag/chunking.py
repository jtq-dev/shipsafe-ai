from pathlib import Path
from typing import List, Tuple

ALLOWED = {".md", ".txt"}

def load_docs(folder: str) -> List[Tuple[str, str]]:
    root = Path(folder).resolve()
    out: List[Tuple[str, str]] = []

    if not root.exists():
        return out

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED:
            continue

        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            # Ensure empty docs still contribute something
            text = f"# {p.name}\n\n(Empty doc file — add content to improve RAG results.)\n"

        # Store relative path for nicer sources in responses
        rel = str(p.relative_to(root))
        out.append((rel, text))

    return out

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120):
    # Never return zero chunks if text has any characters
    if not text:
        return []

    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)

    while i < len(text):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        i += step

    # Fallback: if everything got stripped, keep a minimal chunk
    if not chunks and text.strip():
        chunks = [text[:min(len(text), chunk_size)]]

    return chunks
