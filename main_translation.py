"""
main_translation.py — Task 2: English-German Machine Translation
Runs the full pipeline in order.

Usage:
    python main_translation.py

Run from your Homework 3 directory.
"""

import subprocess
import sys


def run(script: str, args: list[str] = []) -> None:
    """Run a script and exit if it fails."""
    cmd = [sys.executable, script] + args
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed. Fix the error above and re-run.")
        sys.exit(1)


def main() -> None:
    # ── Install dependencies ──────────────────────────────────
    subprocess.run([sys.executable, "-m", "pip", "install", "sacrebleu"])  

    # ── Step 1: Download and verify Multi30K ──────────────────────────────────
    run("data_prep_translation.py")

    # ── Step 2: Build vocabularies ────────────────────────────────────────────
    run("vocab_translation.py")

    # ── Step 3: Verify dataset and dataloaders ────────────────────────────────
    run("dataset_translation.py")

    # ── Step 4: Verify models ─────────────────────────────────────────────────
    run("models_translation.py")

    # ── Step 5: Train all 4 combinations ─────────────────────────────────────
    for arch in ["gru", "lstm"]:
        for embedding in ["glove", "onehot"]:
            run("train_translation.py", ["--arch", arch, "--embedding", embedding])

    # ── Step 6: Evaluate all 4 checkpoints (BLEU + qualitative examples) ─────
    run("evaluate_translation.py", ["--all"])

    print(f"\n{'='*60}")
    print(f"  Task 2 complete!")
    print(f"  Checkpoints saved to: checkpoints_translation/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()