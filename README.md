# ShipSafe AI 🚢🛡️  
**MLOps + RAG “demo that recruiters can actually run”**  
PyTorch training → MLflow tracking → FastAPI model service → RAG `/qa` with sources → evaluation gate in CI.

> Goal: showcase a clean, testable ML system with reproducible runs, tracked experiments, an API, and a simple RAG endpoint.

---

## What’s inside

### ✅ Core features
- **Model training (PyTorch)**: train a small baseline classifier/regressor (project template-ready).
- **Experiment tracking (MLflow)**: log params, metrics, and artifacts.
- **Model serving (FastAPI)**:
  - `GET /healthz`
  - `POST /predict` (model inference)
- **RAG Q&A endpoint**:
  - `POST /qa` returns **answer + sources**
  - Vector store: FAISS or pgvector (depending on your setup)
- **Evaluation gate**:
  - run eval script to produce a score
  - can be plugged into CI to fail when quality drops

---

## Project structure (example)
> Your folders may differ slightly—adjust the paths below if needed.

