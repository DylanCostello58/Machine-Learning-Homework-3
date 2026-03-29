"""
Step 8: Evaluation & results comparison.

Loads all 4 training history files, computes a comparison table,
and generates plots comparing architectures and embedding types.

Run this AFTER all 4 training runs are complete:
    python evaluate.py

Expects these files in the checkpoints directory:
    gru_glove_history.json
    lstm_glove_history.json
    gru_onehot_history.json
    lstm_onehot_history.json
"""

import os
import json
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_DIR  = os.path.join("checkpoints")
PLOTS_DIR = os.path.join("plots")

RUNS = [
    {"name": "GRU + GloVe",   "file": "gru_glove_history.json",   "arch": "GRU",  "emb": "GloVe"},
    {"name": "LSTM + GloVe",  "file": "lstm_glove_history.json",  "arch": "LSTM", "emb": "GloVe"},
    {"name": "GRU + One-Hot", "file": "gru_onehot_history.json",  "arch": "GRU",  "emb": "One-Hot"},
    {"name": "LSTM + One-Hot","file": "lstm_onehot_history.json", "arch": "LSTM", "emb": "One-Hot"},
]
# ──────────────────────────────────────────────────────────────────────────────

# Consistent colours and line styles for all plots
STYLES = {
    "GRU + GloVe":    {"color": "#2196F3", "linestyle": "-"},
    "LSTM + GloVe":   {"color": "#F44336", "linestyle": "-"},
    "GRU + One-Hot":  {"color": "#2196F3", "linestyle": "--"},
    "LSTM + One-Hot": {"color": "#F44336", "linestyle": "--"},
}


def load_history(run: dict) -> list[dict] | None:
    path = os.path.join(CKPT_DIR, run["file"])
    if not os.path.exists(path):
        print(f"[WARN] Missing history file: {path} — skipping {run['name']}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def print_results_table(histories: dict) -> None:
    """Print a formatted table of best val loss and perplexity per run."""
    print("\n" + "="*65)
    print(f"  {'Run':<22} {'Best Val Loss':>14} {'Best Val PPL':>13} {'Epochs':>7}")
    print("="*65)
    for run in RUNS:
        history = histories.get(run["name"])
        if history is None:
            print(f"  {run['name']:<22} {'N/A':>14} {'N/A':>13} {'N/A':>7}")
            continue
        best     = min(history, key=lambda e: e["val_loss"])
        print(
            f"  {run['name']:<22} {best['val_loss']:>14.4f} "
            f"{best['val_ppl']:>13.1f} {best['epoch']:>7}"
        )
    print("="*65)


# ── Plot 1: Validation perplexity over epochs (all 4 runs) ───────────────────
def plot_val_perplexity(histories: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for run in RUNS:
        history = histories.get(run["name"])
        if history is None:
            continue
        epochs = [e["epoch"]   for e in history]
        ppls   = [e["val_ppl"] for e in history]
        style  = STYLES[run["name"]]
        ax.plot(epochs, ppls, label=run["name"], **style, linewidth=2, marker="o", markersize=4)

    ax.set_title("Validation Perplexity over Training Epochs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "val_perplexity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {path}")


# ── Plot 2: Train vs Val loss for each architecture ──────────────────────────
def plot_train_val_loss(histories: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Train vs Validation Loss", fontsize=13, fontweight="bold")

    for ax, arch in zip(axes, ["GRU", "LSTM"]):
        for emb in ["GloVe", "One-Hot"]:
            name    = f"{arch} + {emb}"
            history = histories.get(name)
            if history is None:
                continue
            epochs     = [e["epoch"]      for e in history]
            train_loss = [e["train_loss"] for e in history]
            val_loss   = [e["val_loss"]   for e in history]
            style      = STYLES[name]
            ax.plot(epochs, train_loss, label=f"{emb} train", color=style["color"],
                    linewidth=1.5, linestyle=":")
            ax.plot(epochs, val_loss,   label=f"{emb} val",   color=style["color"],
                    linewidth=2, linestyle=style["linestyle"], marker="o", markersize=4)

        ax.set_title(arch)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "train_val_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {path}")


# ── Plot 3: Final perplexity bar chart ───────────────────────────────────────
def plot_final_perplexity_bar(histories: dict) -> None:
    names = []
    ppls  = []
    colors = []

    color_map = {"GloVe": "#4CAF50", "One-Hot": "#FF9800"}

    for run in RUNS:
        history = histories.get(run["name"])
        if history is None:
            continue
        best = min(history, key=lambda e: e["val_loss"])
        names.append(run["name"])
        ppls.append(best["val_ppl"])
        colors.append(color_map[run["emb"]])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, ppls, color=colors, edgecolor="white", linewidth=0.5)

    # Label each bar with its value
    for bar, ppl in zip(bars, ppls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{ppl:.1f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_title("Best Validation Perplexity by Model Configuration",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    # Legend for embedding colours
    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=l) for l, c in color_map.items()]
    ax.legend(handles=legend, title="Embedding type")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "final_perplexity_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {path}")


# ── Plot 4: GloVe vs One-hot comparison side by side ─────────────────────────
def plot_embedding_comparison(histories: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GloVe vs One-Hot Embedding: Validation Perplexity",
                 fontsize=13, fontweight="bold")

    emb_styles = {
        "GloVe":   {"color": "#4CAF50", "linestyle": "-",  "linewidth": 2},
        "One-Hot": {"color": "#FF9800", "linestyle": "--", "linewidth": 2},
    }

    for ax, arch in zip(axes, ["GRU", "LSTM"]):
        for emb in ["GloVe", "One-Hot"]:
            name    = f"{arch} + {emb}"
            history = histories.get(name)
            if history is None:
                continue
            epochs = [e["epoch"]   for e in history]
            ppls   = [e["val_ppl"] for e in history]
            ax.plot(epochs, ppls, label=emb, marker="o", markersize=4,
                    **emb_styles[emb])

        ax.set_title(arch)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Perplexity")
        ax.legend(title="Embedding")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "embedding_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load all available histories
    histories = {}
    for run in RUNS:
        history = load_history(run)
        if history is not None:
            histories[run["name"]] = history

    if not histories:
        print("[ERROR] No history files found. Train at least one model first.")
        print(f"        Expected files in: {CKPT_DIR}")
        exit(1)

    print(f"[INFO] Loaded {len(histories)}/{len(RUNS)} run histories.")

    print_results_table(histories)

    print("\n[INFO] Generating plots...")
    plot_val_perplexity(histories)
    plot_train_val_loss(histories)
    plot_final_perplexity_bar(histories)
    plot_embedding_comparison(histories)

    print(f"\n[DONE] All plots saved to {PLOTS_DIR}")
    print("       Include these in your writeup for the evaluation section.")