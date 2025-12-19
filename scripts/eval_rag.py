import json
import os
from datasets import Dataset

from app.rag.qa import answer_question

THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.70"))

def load_gold(path="gold/gold_qa.jsonl"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def run_ragas(rows):
    # Uses an LLM (set OPENAI_API_KEY in env / GitHub secret)
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from langchain_openai import ChatOpenAI

    data = {"question": [], "answer": [], "contexts": [], "ground_truths": []}
    for r in rows:
        out = answer_question(r["question"], top_k=4, use_llm=True)
        data["question"].append(r["question"])
        data["answer"].append(out["answer"])
        data["contexts"].append([c["text"] for c in out["chunks"]])
        data["ground_truths"].append(r.get("ground_truths", [r.get("answer","")]))

    ds = Dataset.from_dict(data)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision], llm=llm)
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    # simple combined score
    combined = float(sum(scores.values()) / max(1, len(scores)))
    return combined, scores

def run_retrieval_hit_rate(rows):
    # Offline fallback: checks if at least 1 retrieved chunk contains a key phrase
    hits = 0
    for r in rows:
        out = answer_question(r["question"], top_k=4, use_llm=False)
        target = (r.get("answer") or "").lower()
        got = " ".join([c["text"].lower() for c in out["chunks"]])
        if target and target[:25] in got:
            hits += 1
    return hits / max(1, len(rows)), {"hit_rate": hits / max(1, len(rows))}

if __name__ == "__main__":
    rows = load_gold()

    if os.getenv("OPENAI_API_KEY"):
        score, detail = run_ragas(rows)
        print("RAGAS detail:", detail)
        print("RAGAS combined:", score)
    else:
        score, detail = run_retrieval_hit_rate(rows)
        print("Offline detail:", detail)
        print("Offline score:", score)

    if score < THRESHOLD:
        raise SystemExit(f"❌ RAG quality below threshold {THRESHOLD}. Got {score}.")
    print("✅ RAG eval passed")
