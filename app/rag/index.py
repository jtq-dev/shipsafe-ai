import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.rag.chunking import load_docs, chunk_text

EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR = os.getenv("INDEX_DIR", "artifacts/rag")

class RagIndex:
    def __init__(self):
        self.model = SentenceTransformer(EMB_MODEL)
        self.index = None
        self.meta = []

    def build(self, docs_dir="docs"):
        os.makedirs(INDEX_DIR, exist_ok=True)
        docs = load_docs(docs_dir)

        meta = []
        texts = []
        for path, text in docs:
            for c in chunk_text(text):
                meta.append({"source": path, "text": c})
                texts.append(c)

        if not texts:
            raise RuntimeError(f"No docs found in {docs_dir} (add .md/.txt files).")

        vecs = self.model.encode(texts, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype="float32")

        d = vecs.shape[1]
        self.index = faiss.IndexFlatIP(d)   # exact cosine when normalized
        self.index.add(vecs)
        self.meta = meta

        faiss.write_index(self.index, os.path.join(INDEX_DIR, "faiss.index"))
        with open(os.path.join(INDEX_DIR, "meta.jsonl"), "w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def load(self):
        idx_path = os.path.join(INDEX_DIR, "faiss.index")
        meta_path = os.path.join(INDEX_DIR, "meta.jsonl")
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise RuntimeError("RAG index not found. Run: python scripts/build_index.py")
        self.index = faiss.read_index(idx_path)
        self.meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))

    def query(self, q: str, top_k: int = 4):
        qv = self.model.encode([q], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")
        scores, ids = self.index.search(qv, top_k)
        out = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0: 
                continue
            m = self.meta[int(i)]
            out.append({"source": m["source"], "score": float(score), "text": m["text"]})
        return out

RAG = RagIndex()
