import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

EMBEDDINGS_PATH = Path(__file__).parents[2] / "data" / "processed" / "embeddings.npy"
LABELS_PATH = Path(__file__).parents[2] / "data" / "processed" / "labels.npy"
RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "synthetic_incidents.csv"
MODEL_PATH = Path(__file__).parents[2] / "models" / "classifier.pkl"
LABEL_CLASSES_PATH = Path(__file__).parents[2] / "models" / "label_classes.txt"
MLFLOW_EXPERIMENT = "incident-classifier"


def load_data(
    embeddings_path: Path = EMBEDDINGS_PATH,
    data_path: Path = RAW_DATA_PATH,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    embeddings = np.load(embeddings_path)
    df = pd.read_csv(data_path)
    label_classes = sorted(df["label"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(label_classes)}
    labels = df["label"].map(label_to_idx).to_numpy()
    return embeddings, labels, label_classes


def train(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(x_train, y_train)
    return clf, x_train, x_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression classifier on DistilBERT embeddings")
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=None,
        help="Path to .npy embeddings file. Defaults to data/processed/embeddings.npy.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to raw incidents CSV. Defaults to data/raw/synthetic_incidents.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to save the trained classifier .pkl. Defaults to models/classifier.pkl.",
    )
    args = parser.parse_args()

    embeddings_path = args.embeddings_path if args.embeddings_path is not None else EMBEDDINGS_PATH
    data_path = args.data_path if args.data_path is not None else RAW_DATA_PATH
    model_path = args.output_path if args.output_path is not None else MODEL_PATH

    model_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings, labels, label_classes = load_data(embeddings_path, data_path)
    print(f"Embeddings: {embeddings.shape}, Labels: {labels.shape}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="logistic-regression"):
        mlflow.set_tags({
            "model_type": "LogisticRegression",
            "embedding_model": "distilbert-base-uncased",
            "embedding_dim": "768",
        })
        mlflow.log_params({
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "test_size": 0.2,
            "n_samples": len(labels),
            "n_classes": len(set(labels)),
            "embeddings_path": str(embeddings_path),
            "data_path": str(data_path),
        })

        clf, x_train, x_test, y_train, y_test = train(embeddings, labels)

        train_acc = clf.score(x_train, y_train)
        test_acc = clf.score(x_test, y_test)
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        })

        mlflow.sklearn.log_model(clf, name="classifier")

        joblib.dump(clf, model_path)
        label_classes_path = model_path.parent / (model_path.stem + "_label_classes.txt")
        label_classes_path.write_text("\n".join(label_classes))
        mlflow.log_artifact(str(model_path), artifact_path="model_pkl")

        run_id = mlflow.active_run().info.run_id
        print(f"Training complete. Test set size: {len(y_test)}")
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy:  {test_acc:.4f}")
        print(f"Model saved -> {model_path}")
        print(f"Label classes: {label_classes}")
        print(f"MLflow run:    {run_id}")
