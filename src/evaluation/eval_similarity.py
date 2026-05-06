from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline.similarity import run_temporal_dbscan, build_fused_reports

EMBEDDINGS_PATH = Path(__file__).parents[2] / "data" / "processed" / "embeddings.npy"
RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "synthetic_incidents.csv"


def within_group_similarity(embeddings: np.ndarray, group_ids: np.ndarray) -> float:
    scores = []
    for gid in np.unique(group_ids):
        mask = group_ids == gid
        group_embs = embeddings[mask]
        if group_embs.shape[0] < 2:
            continue
        sims = cosine_similarity(group_embs)
        n = sims.shape[0]
        upper = sims[np.triu_indices(n, k=1)]
        scores.append(float(upper.mean()))
    return float(np.mean(scores))


def cross_label_similarity(embeddings: np.ndarray, labels: np.ndarray, n_pairs: int = 500) -> float:
    rng = np.random.default_rng(42)
    unique_labels = np.unique(labels)
    scores = []
    for _ in range(n_pairs):
        l1, l2 = rng.choice(unique_labels, size=2, replace=False)
        idx1 = rng.choice(np.where(labels == l1)[0])
        idx2 = rng.choice(np.where(labels == l2)[0])
        sim = float(cosine_similarity(
            embeddings[idx1].reshape(1, -1),
            embeddings[idx2].reshape(1, -1),
        )[0, 0])
        scores.append(sim)
    return float(np.mean(scores))


def evaluate_clusters(cluster_labels: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(normalized_mutual_info_score(ground_truth, cluster_labels))


def check_three_source_escalation(df: pd.DataFrame, reports: list[dict]) -> dict:
    three_source = [r for r in reports if len(set(r["sources"])) == 3]
    escalated = [r for r in three_source if r["priority"] == "high"]
    return {
        "total_three_source_clusters": len(three_source),
        "escalated_to_high": len(escalated),
        "all_escalated": len(three_source) == len(escalated),
    }


if __name__ == "__main__":
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    group_ids = df["incident_group_id"].to_numpy()
    labels = df["label"].to_numpy()

    print("Running temporal DBSCAN (cosine similarity + 1-hour time window)...")
    cluster_labels = run_temporal_dbscan(embeddings, df["timestamp"])
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())
    print(f"Clusters: {n_clusters}  |  Noise (singletons): {n_noise}\n")

    within = within_group_similarity(embeddings, group_ids)
    cross = cross_label_similarity(embeddings, labels)
    nmi = evaluate_clusters(cluster_labels, group_ids)

    print(f"Within-group mean cosine similarity  : {within:.4f}  (target >= 0.85)")
    print(f"Cross-label mean cosine similarity   : {cross:.4f}  (target < within-group)")
    print(f"NMI (temporal DBSCAN vs group_id)    : {nmi:.4f}  (target >= 0.80)")

    reports = build_fused_reports(df, embeddings, cluster_labels)
    escalation = check_three_source_escalation(df, reports)
    print(f"\n3-source clusters  : {escalation['total_three_source_clusters']}")
    print(f"Escalated to high  : {escalation['escalated_to_high']}")
    print(f"All escalated      : {escalation['all_escalated']}")

    print("\n--- Summary ---")
    checks = [
        ("Within-group >= 0.85", within >= 0.85),
        ("Cross-label < within-group (same-group pairs are tighter)", cross < within),
        ("NMI          >= 0.80", nmi >= 0.80),
        ("All 3-source clusters escalated", escalation["all_escalated"]),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
