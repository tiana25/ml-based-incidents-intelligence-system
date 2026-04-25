from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

EMBEDDINGS_PATH = Path(__file__).parents[2] / "data" / "processed" / "embeddings.npy"
LABELS_PATH = Path(__file__).parents[2] / "data" / "processed" / "labels.npy"
MODEL_PATH = Path(__file__).parents[2] / "models" / "classifier.pkl"
LABEL_CLASSES_PATH = Path(__file__).parents[2] / "models" / "label_classes.txt"
CONFUSION_MATRIX_PATH = Path(__file__).parents[2] / "data" / "processed" / "confusion_matrix.png"


def evaluate(
    clf,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_classes: list[str],
) -> dict:
    _, x_test, _, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=42
    )

    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=label_classes)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    label_classes: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_classes,
        yticklabels=label_classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Incident Type Classifier")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Confusion matrix saved -> {output_path}")


if __name__ == "__main__":
    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    clf = joblib.load(MODEL_PATH)
    label_classes = LABEL_CLASSES_PATH.read_text().splitlines()

    results = evaluate(clf, embeddings, labels, label_classes)

    print(f"Accuracy:     {results['accuracy']:.4f}")
    print(f"Weighted F1:  {results['weighted_f1']:.4f}")
    print()
    print(results["classification_report"])

    plot_confusion_matrix(results["confusion_matrix"], label_classes, CONFUSION_MATRIX_PATH)
