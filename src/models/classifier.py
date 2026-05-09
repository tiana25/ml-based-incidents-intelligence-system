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


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    label_classes = sorted(df["label"].unique().tolist())
    return embeddings, labels, label_classes

# x_train - 360 embeddings the model trains on
# y_train - 360 correct labels for those embeddings
# x_test - 90 embeddings the model has never seen
# y_test - 90 correct labels used to check predictions against
def train(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    # It looks at each of the 360 embeddings in x_train alongside the correct answer in y_train and repeatedly adjusts its internal weights to minimise prediction errors.
    # When the adjustments become negligibly small it stops - that's called converging.
    # After this, clf knows which regions of the 768-dim embedding space correspond to which incident type.
    clf.fit(x_train, y_train)
    return clf, x_train, x_test, y_train, y_test


if __name__ == "__main__":
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    embeddings, labels, label_classes = load_data()
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
        })

        clf, x_train, x_test, y_train, y_test = train(embeddings, labels)

        train_acc = clf.score(x_train, y_train)
        test_acc = clf.score(x_test, y_test)
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        })

        mlflow.sklearn.log_model(clf, name="classifier")

        joblib.dump(clf, MODEL_PATH)
        LABEL_CLASSES_PATH.write_text("\n".join(label_classes))
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_pkl")

        run_id = mlflow.active_run().info.run_id
        print(f"Training complete. Test set size: {len(y_test)}")
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy:  {test_acc:.4f}")
        print(f"Model saved -> {MODEL_PATH}")
        print(f"Label classes: {label_classes}")
        print(f"MLflow run:    {run_id}")
