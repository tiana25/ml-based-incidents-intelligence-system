from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline.prioritize import escalate_to_high

EMBEDDINGS_PATH = Path(__file__).parents[2] / "data" / "processed" / "embeddings.npy"
RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "synthetic_incidents.csv"

SIMILARITY_THRESHOLD = 0.80
TIME_WINDOW_SECONDS = 3600
DBSCAN_EPS = 0.20
DBSCAN_MIN_SAMPLES = 2


def find_similar(
    embedding: np.ndarray,
    recent_embeddings: np.ndarray,
    recent_ids: list[int],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    if recent_embeddings.shape[0] == 0:
        return []
    scores = cosine_similarity(embedding.reshape(1, -1), recent_embeddings)[0]
    return [
        {"id": recent_ids[i], "similarity": float(scores[i])}
        for i in range(len(scores))
        if scores[i] >= threshold
    ]


def correlate(
    incident_id: int,
    timestamp: pd.Timestamp,
    embedding: np.ndarray,
    all_incidents: pd.DataFrame,
    all_embeddings: np.ndarray,
    cluster_assignments: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
    time_window: int = TIME_WINDOW_SECONDS,
) -> dict:
    idx = all_incidents.index[all_incidents["id"] == incident_id].tolist()
    if not idx:
        return {"cluster_id": None, "related_ids": [], "max_similarity": 0.0}

    within_window = all_incidents[
        (all_incidents["timestamp"].apply(pd.Timestamp) - timestamp).abs()
        <= pd.Timedelta(seconds=time_window)
    ]
    candidate_ids = within_window["id"].tolist()
    candidate_mask = all_incidents["id"].isin(candidate_ids).values
    candidate_embeddings = all_embeddings[candidate_mask]
    candidate_ids_list = within_window["id"].tolist()

    matches = find_similar(embedding, candidate_embeddings, candidate_ids_list, threshold)
    matches = [m for m in matches if m["id"] != incident_id]

    if not matches:
        return {"cluster_id": None, "related_ids": [], "max_similarity": 0.0}

    cluster_id = int(cluster_assignments[all_incidents["id"] == incident_id].values[0])
    if cluster_id == -1:
        cluster_id = None

    return {
        "cluster_id": cluster_id,
        "related_ids": [m["id"] for m in matches],
        "max_similarity": float(max(m["similarity"] for m in matches)),
    }


def run_dbscan(embeddings: np.ndarray) -> np.ndarray:
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine")
    return db.fit_predict(embeddings)


def _temporal_cosine_distance(a: np.ndarray, b: np.ndarray, time_window: float = TIME_WINDOW_SECONDS) -> float:
    """Cosine distance that returns 2.0 (beyond any eps) when timestamps differ by more than time_window."""
    if abs(a[0] - b[0]) > time_window:
        return 2.0
    cos_sim = float(np.dot(a[1:], b[1:]) / (np.linalg.norm(a[1:]) * np.linalg.norm(b[1:]) + 1e-10))
    return 1.0 - cos_sim


def run_temporal_dbscan(
    embeddings: np.ndarray,
    timestamps: pd.Series,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    time_window: float = TIME_WINDOW_SECONDS,
) -> np.ndarray:
    unix_times = pd.to_datetime(timestamps).astype(np.int64) // 10**9
    features = np.hstack([unix_times.to_numpy().reshape(-1, 1).astype(np.float64), embeddings.astype(np.float64)])

    scale = time_window
    features[:, 0] /= scale

    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        if abs(a[0] - b[0]) * scale > time_window:
            return 2.0
        cos_sim = float(np.dot(a[1:], b[1:]) / (np.linalg.norm(a[1:]) * np.linalg.norm(b[1:]) + 1e-10))
        return 1.0 - cos_sim

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=_dist, algorithm="ball_tree")
    return db.fit_predict(features)


def build_fused_reports(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> list[dict]:
    df = df.copy()
    df["cluster_label"] = cluster_labels

    reports = []
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            continue
        mask = df["cluster_label"] == cluster_id
        members = df[mask]
        member_embeddings = embeddings[mask.values]

        n = len(members)
        if n < 2:
            continue

        sims = cosine_similarity(member_embeddings)
        upper = sims[np.triu_indices(n, k=1)]
        max_similarity = float(upper.max()) if len(upper) > 0 else 0.0

        label_counts = members["label"].value_counts()
        majority_label = label_counts.index[0]

        source_types = members["source_type"].unique().tolist()
        incidents = members.to_dict("records")
        escalated = escalate_to_high(incidents)
        priority = escalated[0]["priority"] if escalated else "low"

        reports.append({
            "cluster_id": int(cluster_id),
            "type": majority_label,
            "priority": priority,
            "sources": source_types,
            "signal_count": n,
            "confidence": round(max_similarity, 4),
            "incident_ids": members["id"].tolist(),
        })

    return reports


if __name__ == "__main__":
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(RAW_DATA_PATH)

    print("Running DBSCAN clustering...")
    cluster_labels = run_dbscan(embeddings)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())
    print(f"Clusters found: {n_clusters}  |  Noise (singletons): {n_noise}")

    reports = build_fused_reports(df, embeddings, cluster_labels)
    print(f"\nFused incident reports: {len(reports)}")
    for r in reports[:5]:
        print(
            f"  cluster={r['cluster_id']:3d}  type={r['type']:<25}  "
            f"priority={r['priority']:<6}  sources={r['sources']}  "
            f"signals={r['signal_count']}  confidence={r['confidence']:.4f}"
        )
