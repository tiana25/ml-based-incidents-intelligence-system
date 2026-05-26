import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ClusterReport,
    CorrelateRequest,
    CorrelateResponse,
    HealthResponse,
    PrioritizeRequest,
    PrioritizeResponse,
    SimilarMatch,
)
from src.pipeline.classify import _embed, classify_incident
from src.pipeline.prioritize import score_priority
from src.pipeline.similarity import build_fused_reports, find_similar, run_temporal_dbscan

EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_incidents.csv"

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not EMBEDDINGS_PATH.exists() or not RAW_DATA_PATH.exists():
        raise RuntimeError(
            f"Missing data files. Run src/features/embed.py first.\n"
            f"  embeddings: {EMBEDDINGS_PATH}\n"
            f"  dataset:    {RAW_DATA_PATH}"
        )
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    cluster_labels = run_temporal_dbscan(embeddings, df["timestamp"])
    reports = build_fused_reports(df, embeddings, cluster_labels)

    _state["embeddings"] = embeddings
    _state["df"] = df
    _state["cluster_labels"] = cluster_labels
    _state["reports"] = reports

    yield

    _state.clear()


app = FastAPI(
    title="Incident Intelligence API",
    description="ML pipeline for incident classification, priority scoring, and correlation.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    result = classify_incident(request.text)
    return ClassifyResponse(label=result["label"], confidence=result["confidence"])


@app.post("/prioritize", response_model=PrioritizeResponse)
def prioritize(request: PrioritizeRequest) -> PrioritizeResponse:
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    result = score_priority(request.text, request.source_type)
    return PrioritizeResponse(priority=result["priority"], score=result["score"])


@app.post("/correlate", response_model=CorrelateResponse)
def correlate(request: CorrelateRequest) -> CorrelateResponse:
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    embeddings: np.ndarray = _state["embeddings"]
    df: pd.DataFrame = _state["df"]
    cluster_labels: np.ndarray = _state["cluster_labels"]

    embedding = _embed(request.text)
    all_ids = df["id"].tolist()
    matches = find_similar(embedding, embeddings, all_ids, threshold=request.threshold)

    cluster_id: Optional[int] = None
    if matches:
        best_id = max(matches, key=lambda m: m["similarity"])["id"]
        raw = int(cluster_labels[df["id"] == best_id].values[0])
        cluster_id = raw if raw != -1 else None

    return CorrelateResponse(
        related=[SimilarMatch(id=m["id"], similarity=m["similarity"]) for m in matches],
        cluster_id=cluster_id,
    )


@app.get("/clusters", response_model=list[ClusterReport])
def clusters() -> list[ClusterReport]:
    return [ClusterReport(**r) for r in _state["reports"]]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
