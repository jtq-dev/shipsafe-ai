from app.rag.index import RAG

if __name__ == "__main__":
    RAG.build("docs")
    print("✅ Built FAISS index from ./docs into artifacts/rag/")
