import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ClusterReport,
    CorrelateRequest,
    CorrelateResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    PrioritizeRequest,
    PrioritizeResponse,
    SimilarMatch,
)
from src.pipeline.classify import _embed, classify_incident
from src.pipeline.prioritize import score_priority
from src.pipeline.similarity import build_fused_reports, find_similar, run_temporal_dbscan

EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_incidents.csv"

app = FastAPI(
    title="Incident Intelligence API",
    description="ML pipeline for incident classification, priority scoring, and correlation.",
    version="1.0.0",
)


def _recluster() -> None:
    cluster_labels = run_temporal_dbscan(app.state.embeddings, app.state.df["timestamp"])
    app.state.cluster_labels = cluster_labels
    app.state.reports = build_fused_reports(app.state.df, app.state.embeddings, cluster_labels)


@app.on_event("startup")
def startup() -> None:
    try:
        if not EMBEDDINGS_PATH.exists() or not RAW_DATA_PATH.exists():
            raise RuntimeError(
                f"Missing data files. Run src/features/embed.py first.\n"
                f"  embeddings: {EMBEDDINGS_PATH}\n"
                f"  dataset:    {RAW_DATA_PATH}"
            )
        app.state.embeddings = np.load(EMBEDDINGS_PATH)
        app.state.df = pd.read_csv(RAW_DATA_PATH)
        _recluster()
        logger.info("Startup complete: %d signals, %d clusters", len(app.state.df), len(app.state.reports))
    except Exception:
        logger.error("Startup failed:\n%s", traceback.format_exc())
        raise


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        total_signals=len(request.app.state.df),
        total_clusters=len(request.app.state.reports),
    )


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
def correlate(body: CorrelateRequest, request: Request) -> CorrelateResponse:
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    embedding = _embed(body.text)
    matches = find_similar(
        embedding,
        request.app.state.embeddings,
        request.app.state.df["id"].tolist(),
        threshold=body.threshold,
    )

    cluster_id: Optional[int] = None
    if matches:
        best_id = max(matches, key=lambda m: m["similarity"])["id"]
        raw = int(request.app.state.cluster_labels[(request.app.state.df["id"] == best_id).to_numpy()][0])
        cluster_id = raw if raw != -1 else None

    return CorrelateResponse(
        related=[SimilarMatch(id=m["id"], similarity=m["similarity"]) for m in matches],
        cluster_id=cluster_id,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    timestamp = body.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    new_id = int(request.app.state.df["id"].max()) + 1

    classification = classify_incident(body.text)
    priority_result = score_priority(body.text, body.source_type)
    new_embedding = _embed(body.text)

    new_row = pd.DataFrame([{
        "id": new_id,
        "incident_group_id": -1,
        "source_type": body.source_type,
        "text": body.text,
        "label": classification["label"],
        "priority": priority_result["priority"],
        "timestamp": timestamp,
    }])

    request.app.state.df = pd.concat([request.app.state.df, new_row], ignore_index=True)
    request.app.state.embeddings = np.vstack([request.app.state.embeddings, new_embedding])
    _recluster()

    raw_cluster = int(request.app.state.cluster_labels[(request.app.state.df["id"] == new_id).to_numpy()][0])
    cluster_id: Optional[int] = raw_cluster if raw_cluster != -1 else None

    related_ids: list[int] = []
    if cluster_id is not None:
        report = next((r for r in request.app.state.reports if r["cluster_id"] == cluster_id), None)
        if report:
            related_ids = [i for i in report["incident_ids"] if i != new_id]

    return IngestResponse(
        id=new_id,
        label=classification["label"],
        confidence=classification["confidence"],
        priority=priority_result["priority"],
        cluster_id=cluster_id,
        related_ids=related_ids,
    )


@app.get("/clusters", response_model=list[ClusterReport])
def clusters(request: Request) -> list[ClusterReport]:
    return [ClusterReport(**r) for r in request.app.state.reports]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
