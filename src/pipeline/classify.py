from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = Path(__file__).parents[2] / "models" / "classifier.pkl"
LABEL_CLASSES_PATH = Path(__file__).parents[2] / "models" / "label_classes.txt"
DISTILBERT_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


@lru_cache(maxsize=1)
def _load_models() -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_NAME)
    bert = AutoModel.from_pretrained(DISTILBERT_NAME)
    bert.eval()
    clf = joblib.load(MODEL_PATH)
    label_classes = LABEL_CLASSES_PATH.read_text().splitlines()
    return tokenizer, bert, clf, label_classes


def _embed(text: str) -> np.ndarray:
    tokenizer, bert, _, _ = _load_models()
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = bert(**encoded)
    return output.last_hidden_state[:, 0, :].numpy().astype(np.float32)


def classify_incident(text: str) -> dict:
    _, _, clf, label_classes = _load_models()
    embedding = _embed(text)
    probabilities = clf.predict_proba(embedding)[0]
    predicted_idx = int(np.argmax(probabilities))
    return {
        "label": label_classes[predicted_idx],
        "confidence": float(probabilities[predicted_idx]),
    }


if __name__ == "__main__":
    test_cases = [
        "Token validation failed for user admin",
        "Packet loss detected on backbone switch, DNS resolution failing",
        "Pod restarting repeatedly after deployment. OOMKilled in kubelet logs.",
    ]
    for text in test_cases:
        result = classify_incident(text)
        print(f"Text:       {text[:60]}...")
        print(f"Label:      {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print()
