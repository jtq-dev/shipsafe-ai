from pathlib import Path
from typing import List, Tuple

def load_docs(folder: str) -> List[Tuple[str, str]]:
    out = []
    for p in Path(folder).rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}:
            out.append((str(p), p.read_text(encoding="utf-8", errors="replace")))
    return out

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size].strip())
        i += max(1, chunk_size - overlap)
    return [c for c in chunks if c]
