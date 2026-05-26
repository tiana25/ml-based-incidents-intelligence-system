import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "synthetic_incidents.csv"
EMBEDDINGS_PATH = Path(__file__).parents[2] / "data" / "processed" / "embeddings.npy"
EMBEDDINGS_BASE_PATH = Path(__file__).parents[2] / "data" / "processed" / "embeddings_base.npy"
EMBEDDINGS_FINETUNED_PATH = Path(__file__).parents[2] / "data" / "processed" / "embeddings_finetuned.npy"
LABELS_PATH = Path(__file__).parents[2] / "data" / "processed" / "labels.npy"
FINETUNED_MODEL_PATH = Path(__file__).parents[2] / "models" / "distilbert-finetuned"
FINETUNED_REALISTIC_MODEL_PATH = Path(__file__).parents[2] / "models" / "distilbert-finetuned-realistic"
EMBEDDINGS_REALISTIC_PATH = Path(__file__).parents[2] / "data" / "processed" / "embeddings_realistic_finetuned.npy"
REALISTIC_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "synthetic_incidents_realistic.csv"

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
MAX_LENGTH = 128


def extract_cls_embeddings(
    texts: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    # Tokenizer is always loaded from the base model — fine-tuning doesn't change the vocabulary
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
        encoded.pop("token_type_ids", None)
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
    parser = argparse.ArgumentParser(description="Extract DistilBERT CLS embeddings")
    parser.add_argument(
        "--model",
        choices=["base", "finetuned", "finetuned-realistic"],
        default="base",
        help=(
            "'base' uses distilbert-base-uncased; "
            "'finetuned' loads models/distilbert-finetuned/; "
            "'finetuned-realistic' loads models/distilbert-finetuned-realistic/ and uses the realistic dataset"
        ),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to input CSV. Defaults to data/raw/synthetic_incidents.csv.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to a fine-tuned model directory. Overrides --model when provided.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Filename prefix for the output .npy file saved to data/processed/. "
            "E.g. 'embeddings_realistic_finetuned' -> data/processed/embeddings_realistic_finetuned.npy. "
            "Defaults to 'embeddings_base' or 'embeddings_finetuned' based on --model."
        ),
    )
    args = parser.parse_args()

    if args.model == "finetuned-realistic":
        data_path = args.data_path if args.data_path is not None else REALISTIC_DATA_PATH
        model_dir = args.model_dir if args.model_dir is not None else FINETUNED_REALISTIC_MODEL_PATH
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Fine-tuned realistic model not found at {model_dir}.\n"
                "Run research/finetune_distilbert.ipynb in Colab on the realistic dataset first,\n"
                "then extract the downloaded zip to models/distilbert-finetuned-realistic/."
            )
        model_name = str(model_dir)
        output_path = EMBEDDINGS_REALISTIC_PATH if args.output_prefix is None else EMBEDDINGS_PATH.parent / f"{args.output_prefix}.npy"
    else:
        data_path = args.data_path if args.data_path is not None else RAW_DATA_PATH

        if args.model_dir is not None:
            if not args.model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
            model_name = str(args.model_dir)
        elif args.model == "finetuned":
            if not FINETUNED_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Fine-tuned model not found at {FINETUNED_MODEL_PATH}.\n"
                    "Run research/finetune_distilbert.ipynb in Colab first, then extract\n"
                    "the downloaded zip to models/distilbert-finetuned/ in this project."
                )
            model_name = str(FINETUNED_MODEL_PATH)
        else:
            model_name = MODEL_NAME

        if args.output_prefix is not None:
            output_path = EMBEDDINGS_PATH.parent / f"{args.output_prefix}.npy"
        elif args.model_dir is not None or args.model == "finetuned":
            output_path = EMBEDDINGS_FINETUNED_PATH
        else:
            output_path = EMBEDDINGS_BASE_PATH

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")
    print(f"Model: {model_name}")

    print("Extracting [CLS] embeddings from DistilBERT...")
    embeddings = extract_cls_embeddings(df["text"].tolist(), model_name=model_name)

    encoded_labels, label_classes = encode_labels(df["label"])

    np.save(output_path, embeddings)
    if output_path == EMBEDDINGS_BASE_PATH:
        np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, encoded_labels)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {encoded_labels.shape}")
    print(f"Label classes: {label_classes}")
    print(f"Saved -> {output_path}")
