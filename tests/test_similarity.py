import numpy as np
import pandas as pd
import pytest

from src.pipeline.similarity import build_fused_reports, find_similar, run_temporal_dbscan


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _timestamps(base: str, offsets_minutes: list[int]) -> pd.Series:
    t0 = pd.Timestamp(base)
    return pd.Series([str(t0 + pd.Timedelta(minutes=m)) for m in offsets_minutes])


# --- find_similar ---

def test_find_similar_empty_candidates_returns_empty():
    embedding = np.ones((1, 4), dtype=np.float32)
    result = find_similar(embedding, np.empty((0, 4), dtype=np.float32), [])
    assert result == []


def test_find_similar_match_above_threshold():
    base = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    similar = _unit(np.array([[0.99, 0.1, 0.0, 0.0]], dtype=np.float32))
    results = find_similar(base, similar, [42], threshold=0.80)
    assert len(results) == 1
    assert results[0]["id"] == 42
    assert results[0]["similarity"] > 0.80


def test_find_similar_orthogonal_not_returned():
    base = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    orthogonal = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    results = find_similar(base, orthogonal, [99], threshold=0.80)
    assert results == []


def test_find_similar_result_has_id_and_similarity_keys():
    base = np.array([[1.0, 0.0]], dtype=np.float32)
    same = np.array([[1.0, 0.0]], dtype=np.float32)
    results = find_similar(base, same, [7], threshold=0.50)
    assert "id" in results[0]
    assert "similarity" in results[0]


# --- run_temporal_dbscan ---

def test_temporal_dbscan_clusters_similar_nearby_signals():
    rng = np.random.default_rng(42)
    base = _unit(rng.random(768).astype(np.float32))
    embs = np.vstack([
        _unit(base + rng.random(768).astype(np.float32) * 0.01),
        _unit(base + rng.random(768).astype(np.float32) * 0.01),
    ])
    labels = run_temporal_dbscan(embs, _timestamps("2024-01-15 10:00", [0, 5]))
    assert labels[0] == labels[1]
    assert labels[0] != -1


def test_temporal_dbscan_separates_signals_beyond_time_window():
    rng = np.random.default_rng(42)
    base = _unit(rng.random(768).astype(np.float32))
    embs = np.vstack([base.copy(), base.copy()])
    # 3 hours apart — well beyond the 60-minute window
    labels = run_temporal_dbscan(embs, _timestamps("2024-01-15 10:00", [0, 180]))
    assert all(label == -1 for label in labels)


def test_temporal_dbscan_returns_array_same_length_as_input():
    rng = np.random.default_rng(0)
    embs = rng.random((10, 768)).astype(np.float32)
    for i in range(len(embs)):
        embs[i] = _unit(embs[i])
    labels = run_temporal_dbscan(embs, _timestamps("2024-01-15 10:00", list(range(10))))
    assert len(labels) == 10


# --- build_fused_reports ---

def _make_df(labels: list[str], source_types: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "id": list(range(1, len(labels) + 1)),
        "label": labels,
        "source_type": source_types,
        "priority": ["low"] * len(labels),
    })


def _rand_embs(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embs = rng.random((n, dim)).astype(np.float32)
    for i in range(n):
        embs[i] = _unit(embs[i])
    return embs


def test_build_fused_reports_excludes_noise_cluster():
    df = _make_df(["authentication_failure"] * 3, ["ticket", "log", "alert"])
    reports = build_fused_reports(df, _rand_embs(3), np.array([-1, -1, -1]))
    assert reports == []


def test_build_fused_reports_escalates_multi_source_cluster():
    df = _make_df(["authentication_failure"] * 3, ["ticket", "log", "alert"])
    reports = build_fused_reports(df, _rand_embs(3), np.array([0, 0, 0]))
    assert len(reports) == 1
    assert reports[0]["priority"] == "high"
    assert reports[0]["signal_count"] == 3


def test_build_fused_reports_majority_label_selected():
    df = _make_df(
        ["authentication_failure", "authentication_failure", "network_issue"],
        ["ticket", "log", "alert"],
    )
    reports = build_fused_reports(df, _rand_embs(3), np.array([0, 0, 0]))
    assert reports[0]["type"] == "authentication_failure"


def test_build_fused_reports_report_contains_expected_keys():
    df = _make_df(["network_issue"] * 2, ["ticket", "log"])
    reports = build_fused_reports(df, _rand_embs(2), np.array([0, 0]))
    report = reports[0]
    for key in ("cluster_id", "type", "priority", "sources", "signal_count", "confidence", "incident_ids"):
        assert key in report


def test_build_fused_reports_confidence_in_unit_range():
    df = _make_df(["deployment_issue"] * 2, ["ticket", "alert"])
    reports = build_fused_reports(df, _rand_embs(2), np.array([0, 0]))
    assert 0.0 <= reports[0]["confidence"] <= 1.0
