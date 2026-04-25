from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

EMBEDDINGS_PATH = Path(__file__).parents[2] / "data" / "processed" / "embeddings.npy"
LABELS_PATH = Path(__file__).parents[2] / "data" / "processed" / "labels.npy"
RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "synthetic_incidents.csv"
MODEL_PATH = Path(__file__).parents[2] / "models" / "classifier.pkl"
LABEL_CLASSES_PATH = Path(__file__).parents[2] / "models" / "label_classes.txt"


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    label_classes = sorted(df["label"].unique().tolist())
    return embeddings, labels, label_classes


def train(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(x_train, y_train)
    return clf, x_test, y_test


if __name__ == "__main__":
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    embeddings, labels, label_classes = load_data()
    print(f"Embeddings: {embeddings.shape}, Labels: {labels.shape}")

    clf, x_test, y_test = train(embeddings, labels)
    print(f"Training complete. Test set size: {len(y_test)}")

    joblib.dump(clf, MODEL_PATH)
    LABEL_CLASSES_PATH.write_text("\n".join(label_classes))

    print(f"Model saved -> {MODEL_PATH}")
    print(f"Label classes: {label_classes}")
    print(f"Test accuracy: {clf.score(x_test, y_test):.4f}")
