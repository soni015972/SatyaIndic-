"""
FastAPI Backend Application
===========================
Serves:
- /predict    : Sequence classification with token importance and latency
- /benchmark  : Precomputed & live benchmark metrics for IndicBERT, MuRIL, XLM-R, BERT
- /models     : Architecture metadata & configuration
- /health     : Health check & compute device status
"""

import os
import sys
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from backend.inference import predict_fake_news, manager
    from backend.benchmark_data import BENCHMARK_DATA, RESEARCH_INSIGHTS
except ImportError:
    from inference import predict_fake_news, manager
    from benchmark_data import BENCHMARK_DATA, RESEARCH_INSIGHTS

app = FastAPI(
    title="Multilingual & Code-Switched Fake News Detection API",
    description="Inference and Benchmarking Engine for IndicBERT, MuRIL, XLM-RoBERTa, and English BERT",
    version="1.0.0"
)

# Enable CORS for Next.js / React frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response Schemas
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=2, description="News text or message to classify.")
    model: Optional[str] = Field(
        default="muril",
        description="Transformer model: 'indic-bert', 'muril', 'xlm-roberta', or 'bert-base-uncased'"
    )
    explain: Optional[bool] = Field(
        default=True,
        description="Whether to return token-level importance scores."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Doston 5G tower se nikalne wali radiation se log behosh ho rahe hain, forward this to all groups!",
                "model": "muril",
                "explain": True
            }
        }


class TokenImportance(BaseModel):
    token: str
    importance: float
    label_driver: str


class PredictionResponse(BaseModel):
    prediction: str
    label_id: int
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    latency_ms: float
    detected_script: str
    detected_language: str
    cleaned_text: str
    token_importance: List[TokenImportance]


@app.get("/")
def root():
    return {
        "title": "Multilingual & Code-Switched Fake News Detection API",
        "status": "online",
        "device": manager.device.upper(),
        "endpoints": ["/predict", "/benchmark", "/models", "/health", "/docs"]
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": manager.device.upper(),
        "loaded_models": list(manager.loaded_models.keys())
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """
    Classifies input text as FAKE or REAL news using the specified Transformer architecture.
    """
    valid_models = ["indic-bert", "muril", "xlm-roberta", "bert-base-uncased"]
    selected_model = req.model.lower().strip()
    if selected_model not in valid_models:
        selected_model = "muril"

    result = predict_fake_news(
        text=req.text,
        model_key=selected_model,
        explain=req.explain
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/benchmark")
def get_benchmarks():
    """
    Returns comparative evaluation metrics across IndicBERT, MuRIL, XLM-RoBERTa, and English BERT.
    """
    return {
        "models": BENCHMARK_DATA,
        "research_insights": RESEARCH_INSIGHTS,
        "summary": {
            "top_performer": "muril",
            "fastest_model": "indic-bert",
            "baseline_model": "bert-base-uncased"
        }
    }


@app.get("/models")
def list_models():
    """
    Returns specifications, parameter counts, and supported scripts for each model.
    """
    models_info = []
    for key, data in BENCHMARK_DATA.items():
        models_info.append({
            "key": key,
            "display_name": data["display_name"],
            "architecture": data["architecture"],
            "params_millions": data["params_millions"],
            "pretraining_corpus": data["pretraining_corpus"],
            "supported_scripts": data.get("supported_scripts", ["Devanagari", "Latin"]),
            "accuracy": data["accuracy"],
            "macro_f1": data["macro_f1"],
            "latency_gpu_ms": data["latency_gpu_ms"]
        })
    return {"models": models_info}


# Serve built frontend if exists
frontend_dist = os.path.join(BASE_DIR, "frontend", "dist")
frontend_out = os.path.join(BASE_DIR, "frontend", "out")

if os.path.exists(frontend_dist):
    app.mount("/dashboard", StaticFiles(directory=frontend_dist, html=True), name="frontend_dist")
elif os.path.exists(frontend_out):
    app.mount("/dashboard", StaticFiles(directory=frontend_out, html=True), name="frontend_out")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
