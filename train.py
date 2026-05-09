import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("Generate synthetic data",       "src/data/generate.py"),
    ("Extract DistilBERT embeddings",  "src/features/embed.py"),
    ("Train classifier",               "src/models/classifier.py"),
    ("Evaluate classification",        "src/evaluation/eval_classifier.py"),
    ("Evaluate similarity/clustering", "src/evaluation/eval_similarity.py"),
]

def run_step(name: str, script: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"  python {script}")
    print(f"{'─' * 60}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd=Path(__file__).parent,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAILED] {script} exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[OK] completed in {elapsed:.1f}s")


if __name__ == "__main__":
    total_start = time.time()
    print("ML-Based Incident Intelligence — Training Pipeline")

    for name, script in STEPS:
        run_step(name, script)

    total = time.time() - total_start
    print(f"\n{'═' * 60}")
    print(f"  All steps completed in {total:.1f}s")
    print(f"{'═' * 60}")
