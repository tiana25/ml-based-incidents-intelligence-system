from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline.classify import classify_incident

LABEL_CLASSES = ["authentication_failure", "deployment_issue", "network_issue"]


def _make_clf(pred_idx: int, confidence: float = 0.95) -> MagicMock:
    probs = np.zeros(len(LABEL_CLASSES))
    probs[pred_idx] = confidence
    remaining = (1.0 - confidence) / (len(LABEL_CLASSES) - 1)
    for i in range(len(LABEL_CLASSES)):
        if i != pred_idx:
            probs[i] = remaining
    clf = MagicMock()
    clf.predict_proba.return_value = np.array([probs])
    return clf


def _fake_embed(*_args, **_kwargs) -> np.ndarray:
    return np.random.default_rng(42).random((1, 768)).astype(np.float32)


@patch("src.pipeline.classify._embed", side_effect=_fake_embed)
@patch("src.pipeline.classify._load_models")
def test_result_has_label_and_confidence_keys(mock_load, _mock_embed):
    mock_load.return_value = (None, None, _make_clf(0), LABEL_CLASSES)
    result = classify_incident("Token validation failed for user admin")
    assert "label" in result
    assert "confidence" in result


@patch("src.pipeline.classify._embed", side_effect=_fake_embed)
@patch("src.pipeline.classify._load_models")
def test_label_is_a_known_class(mock_load, _mock_embed):
    mock_load.return_value = (None, None, _make_clf(2), LABEL_CLASSES)
    result = classify_incident("Packet loss on backbone switch")
    assert result["label"] in LABEL_CLASSES


@patch("src.pipeline.classify._embed", side_effect=_fake_embed)
@patch("src.pipeline.classify._load_models")
def test_confidence_is_in_unit_range(mock_load, _mock_embed):
    mock_load.return_value = (None, None, _make_clf(0, 0.91), LABEL_CLASSES)
    result = classify_incident("Token validation failed for user admin")
    assert 0.0 <= result["confidence"] <= 1.0


@patch("src.pipeline.classify._embed", side_effect=_fake_embed)
@patch("src.pipeline.classify._load_models")
def test_predicted_label_matches_highest_probability_class(mock_load, _mock_embed):
    mock_load.return_value = (None, None, _make_clf(1, 0.88), LABEL_CLASSES)
    result = classify_incident("Pod restarting after deployment")
    assert result["label"] == LABEL_CLASSES[1]
    assert result["confidence"] == pytest.approx(0.88, abs=1e-4)


@patch("src.pipeline.classify._embed", side_effect=_fake_embed)
@patch("src.pipeline.classify._load_models")
def test_confidence_reflects_actual_probability(mock_load, _mock_embed):
    mock_load.return_value = (None, None, _make_clf(0, 0.72), LABEL_CLASSES)
    result = classify_incident("Some incident text")
    assert result["confidence"] == pytest.approx(0.72, abs=1e-4)
