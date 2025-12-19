from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional

from app.ml.infer import predict_digit_from_base64
from app.rag.qa import answer_question

app = FastAPI(title="ShipSafe AI", version="1.0.0")

app.mount("/static", StaticFiles(directory="app/web"), name="static")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/web/index.html", "r", encoding="utf-8") as f:
        return f.read()

class PredictReq(BaseModel):
    image_base64: str = Field(..., description="Base64 PNG/JPG (data URL is OK)")

class PredictRes(BaseModel):
    prediction: int
    probabilities: List[float]

@app.post("/predict", response_model=PredictRes)
def predict(req: PredictReq):
    try:
        pred, probs = predict_digit_from_base64(req.image_base64)
        return {"prediction": pred, "probabilities": probs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class QAReq(BaseModel):
    question: str
    top_k: int = 4
    use_llm: bool = False  # optional (needs OPENAI_API_KEY)

class QAChunk(BaseModel):
    source: str
    score: float
    text: str

class QARes(BaseModel):
    answer: str
    chunks: List[QAChunk]

@app.post("/qa", response_model=QARes)
def qa(req: QAReq):
    out = answer_question(req.question, top_k=req.top_k, use_llm=req.use_llm)
    return out
