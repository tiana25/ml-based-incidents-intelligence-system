from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "synthetic_incidents.csv"
EMBEDDINGS_PATH = Path(__file__).parents[2] / "data" / "processed" / "embeddings.npy"
LABELS_PATH = Path(__file__).parents[2] / "data" / "processed" / "labels.npy"

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
MAX_LENGTH = 128


def extract_cls_embeddings(
    texts: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**encoded)
        cls_embeddings = output.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(cls_embeddings)
        print(f"  Embedded {min(start + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_embeddings).astype(np.float32)


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, list[str]]:
    label_classes = sorted(labels.unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(label_classes)}
    encoded = labels.map(label_to_idx).to_numpy()
    return encoded, label_classes


if __name__ == "__main__":
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df)} rows from {RAW_DATA_PATH}")

    print("Extracting [CLS] embeddings from DistilBERT...")
    embeddings = extract_cls_embeddings(df["text"].tolist())

    encoded_labels, label_classes = encode_labels(df["label"])

    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, encoded_labels)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {encoded_labels.shape}")
    print(f"Label classes: {label_classes}")
    print(f"Saved embeddings -> {EMBEDDINGS_PATH}")
    print(f"Saved labels -> {LABELS_PATH}")
