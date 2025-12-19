import os
from app.rag.index import RAG

def _simple_answer(question: str, chunks):
    # “Understandable for anyone” mode: no LLM needed.
    if not chunks:
        return "I couldn't find anything relevant in the docs."
    bullets = "\n".join([f"- ({c['source']}, score={c['score']:.3f}) {c['text'][:220]}..." for c in chunks])
    return f"Here’s what I found in the docs about: '{question}'\n\n{bullets}"

def _llm_answer(question: str, chunks):
    # Optional: requires OPENAI_API_KEY
    from langchain_openai import ChatOpenAI
    ctx = "\n\n".join([f"SOURCE: {c['source']}\n{c['text']}" for c in chunks])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        "Answer the question using ONLY the provided context.\n"
        "Return a short answer and cite sources by filename.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}"
    )
    return llm.invoke(prompt).content

def answer_question(question: str, top_k: int = 4, use_llm: bool = False):
    # Lazy-load index
    try:
        if RAG.index is None:
            RAG.load()
    except Exception:
        # if index missing, build once
        RAG.build("docs")

    chunks = RAG.query(question, top_k=top_k)

    if use_llm:
        if not os.getenv("OPENAI_API_KEY"):
            return {"answer": "OPENAI_API_KEY not set. Uncheck 'Use LLM' or set the key.", "chunks": chunks}
        ans = _llm_answer(question, chunks)
    else:
        ans = _simple_answer(question, chunks)

    return {"answer": ans, "chunks": chunks}
