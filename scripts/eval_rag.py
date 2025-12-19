import json
import os
from typing import List, Dict, Tuple

from app.rag.qa import answer_question

THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.70"))

def load_gold(path: str = "gold/gold_qa.jsonl") -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No gold rows found in {path}")
    return rows

def score_offline(rows: List[Dict]) -> Tuple[float, Dict]:
    """
    Deterministic retrieval-quality score:
    Hit = any retrieved chunk source matches expected_sources.
    """
    hits = 0
    total = 0
    misses = []

    for r in rows:
        q = r["question"]
        expected = [s.lower() for s in r.get("expected_sources", [])]
        if not expected:
            raise RuntimeError("Each gold row must include expected_sources (list of filenames).")

        out = answer_question(q, top_k=4, use_llm=False)
        got_sources = [c["source"].lower() for c in out["chunks"]]

        total += 1
        ok = any(exp in src for exp in expected for src in got_sources)
        if ok:
            hits += 1
        else:
            misses.append({"question": q, "expected": expected, "got": got_sources})

    score = hits / max(1, total)
    detail = {"hit_rate": score, "hits": hits, "total": total, "misses_preview": misses[:3]}
    return score, detail

def score_ragas(rows: List[Dict]) -> Tuple[float, Dict]:
    """
    If OPENAI_API_KEY is set, run Ragas metrics (LLM-based).
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from langchain_openai import ChatOpenAI

    data = {"question": [], "answer": [], "contexts": [], "ground_truths": []}

    for r in rows:
        q = r["question"]
        gt = r.get("ground_truths") or [r.get("answer", "")]

        out = answer_question(q, top_k=4, use_llm=True)

        data["question"].append(q)
        data["answer"].append(out["answer"])
        data["contexts"].append([c["text"] for c in out["chunks"]])
        data["ground_truths"].append(gt)

    ds = Dataset.from_dict(data)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision], llm=llm)
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    combined = float(sum(scores.values()) / max(1, len(scores)))
    return combined, scores

if __name__ == "__main__":
    rows = load_gold()

    if os.getenv("OPENAI_API_KEY"):
        score, detail = score_ragas(rows)
        print("RAGAS detail:", detail)
        print("RAGAS combined:", score)
    else:
        score, detail = score_offline(rows)
        print("Offline detail:", detail)
        print("Offline score:", score)

    if score < THRESHOLD:
        raise SystemExit(f"❌ RAG quality below threshold {THRESHOLD}. Got {score}. Detail: {detail}")

    print("✅ RAG eval passed")
