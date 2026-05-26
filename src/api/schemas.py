from typing import Optional

from pydantic import BaseModel


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    label: str
    confidence: float


class PrioritizeRequest(BaseModel):
    text: str
    source_type: str


class PrioritizeResponse(BaseModel):
    priority: str
    score: float


class CorrelateRequest(BaseModel):
    text: str
    threshold: float = 0.80


class SimilarMatch(BaseModel):
    id: int
    similarity: float


class CorrelateResponse(BaseModel):
    related: list[SimilarMatch]
    cluster_id: Optional[int]


class ClusterReport(BaseModel):
    cluster_id: int
    type: str
    priority: str
    sources: list[str]
    signal_count: int
    confidence: float
    incident_ids: list[int]


class IngestRequest(BaseModel):
    text: str
    source_type: str
    timestamp: Optional[str] = None


class IngestResponse(BaseModel):
    id: int
    label: str
    confidence: float
    priority: str
    cluster_id: Optional[int]
    related_ids: list[int]


class HealthResponse(BaseModel):
    status: str
    total_signals: int
    total_clusters: int
