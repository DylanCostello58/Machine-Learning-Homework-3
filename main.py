"""
main.py — Task 1: Shakespeare Text Generation
Runs the full pipeline in order.

Usage:
    python main.py

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
    # ── Step 1: Tokenise ──────────────────────────────────────────────────────
    run("tokenise.py")

    # ── Step 2: Verify dataset ────────────────────────────────────────────────
    run("dataset.py")

    # ── Step 3: Verify models ─────────────────────────────────────────────────
    run("models.py")

    # ── Step 4: Verify embeddings ─────────────────────────────────────────────
    run("embeddings.py")

    # ── Step 5: Train all 4 combinations ─────────────────────────────────────
    for arch in ["gru", "lstm"]:
        for embedding in ["glove", "onehot"]:
            run("train.py", ["--arch", arch, "--embedding", embedding])

    # ── Step 6: Evaluate and plot ─────────────────────────────────────────────
    run("evaluate.py")

    # ── Step 7: Generate text from all 4 checkpoints ─────────────────────────
    for arch in ["gru", "lstm"]:
        for embedding in ["glove", "onehot"]:
            run("generate.py", [
                "--checkpoint", f"checkpoints/{arch}_{embedding}_best.pt",
                "--prompt",     "to be or not",
                "--words",      "100",
            ])

    print(f"\n{'='*60}")
    print(f"  Task 1 complete!")
    print(f"  Plots saved to: plots/")
    print(f"  Checkpoints saved to: checkpoints/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()