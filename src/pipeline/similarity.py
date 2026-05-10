from pathlib import Path
from turtle import distance

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

# finds who is this incident similar to
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

# finds related incidents, but only within the last hour
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

# group all incidents into clusters in one batch
# DBSCAN is a clustering algorithm
# eps=0.20 is the max distance allowed between two neighbors (distance = 1 - similarity, so 0.20 = similarity of 0.80). Points with no neighbors become noise (-1).
def run_dbscan(embeddings: np.ndarray) -> np.ndarray:
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine")
    # runs the algorithm and returns an array of cluster labels, one per incident
    # embeddings:     [emb_0, emb_1, emb_2, emb_3, emb_4, ...]   # 450 vectors
    # cluster_labels: [   0,     0,     0,     -1,     1,  ...]   # 450 labels
    # 0, 1, 2 ... → cluster IDs (incidents grouped together)
    # -1 -> noise, incident didn't fit into any cluster
    return db.fit_predict(embeddings)


def _temporal_cosine_distance(a: np.ndarray, b: np.ndarray, time_window: float = TIME_WINDOW_SECONDS) -> float:
    """Cosine distance that returns 2.0 (beyond any eps) when timestamps differ by more than time_window."""
    if abs(a[0] - b[0]) > time_window:
        return 2.0
    cos_sim = float(np.dot(a[1:], b[1:]) / (np.linalg.norm(a[1:]) * np.linalg.norm(b[1:]) + 1e-10))
    return 1.0 - cos_sim

# same as run_dbscan but also checks timestamps
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
    # features[:, 0] selects the entire first column — all the timestamp values.
    # /= scale divides them all by 3600 (the time window in seconds).
    # Raw timestamp: 1705286400 (huge)
    # Embedding values: 0.12, 0.45, ... (tiny)
    # Dividing by 3600 makes the timestamp small too, so neither one drowns out the other in the distance calculation.
    features[:, 0] /= scale

    #So each row is now [timestamp, 768 embedding values].
    #The _dist function then uses a[0] for the timestamp check and a[1:] for the cosine similarity calculation.
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        #a[0] and b[0] are the normalized timestamps. Multiply back by scale (3600) to get seconds. If the two incidents are more than 60 min apart -> return 2.0, which is an impossibly large distance (DBSCAN's eps is only 0.20), so they will never be clustered together.
        if abs(a[0] - b[0]) * scale > time_window:
            return 2.0
        cos_sim = float(np.dot(a[1:], b[1:]) / (np.linalg.norm(a[1:]) * np.linalg.norm(b[1:]) + 1e-10))
        # DBSCAN needs a distance (0 = identical, 1 = completely different), but cosine gives similarity (1 = identical). Flipping it with 1 - cos_sim converts one to the other.
        return 1.0 - cos_sim

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=_dist, algorithm="ball_tree")
    return db.fit_predict(features)

# turn each cluster into one incident report
# For each cluster: picks the majority incident type (vote), 
# takes the highest pairwise similarity as confidence,
# escalates priority if ≥ 2 source types are present, 
# and bundles it all into one dict.
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

    print("--- Debug: find_similar ---")
    test_embedding = embeddings[0]
    recent_embeddings = embeddings[1:10]
    recent_ids = df["id"].tolist()[1:10]
    matches = find_similar(test_embedding, recent_embeddings, recent_ids)
    print(f"Query incident id: {df['id'].iloc[0]}  text: {df['text'].iloc[0][:60]}")
    print(f"Matches above threshold {SIMILARITY_THRESHOLD}:")
    for m in matches:
        row = df[df["id"] == m["id"]].iloc[0]
        print(f"  id={m['id']}  similarity={m['similarity']:.4f}  text: {row['text'][:60]}")
    print()

    print("--- Debug: run_temporal_dbscan ---")
    temporal_labels = run_temporal_dbscan(embeddings, df["timestamp"])
    n_temporal_clusters = len(set(temporal_labels)) - (1 if -1 in temporal_labels else 0)
    n_temporal_noise = int((temporal_labels == -1).sum())
    print(f"Temporal clusters found: {n_temporal_clusters}  |  Noise (singletons): {n_temporal_noise}")
    df["temporal_cluster"] = temporal_labels
    for cid in sorted(set(temporal_labels))[:3]:
        if cid == -1:
            continue
        members = df[df["temporal_cluster"] == cid]
        print(f"\n  cluster {cid} ({len(members)} incidents):")
        for _, row in members.iterrows():
            print(f"    id={row['id']}  source={row['source_type']}  ts={row['timestamp']}  text: {row['text'][:60]}")
    print()

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
