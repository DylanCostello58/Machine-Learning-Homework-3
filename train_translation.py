"""
Step 5: Training loop for English-German seq2seq translation.

Trains any combination of architecture (lstm/gru) and embedding (glove/onehot).
Evaluates with cross-entropy loss on the val set after each epoch.
Saves the best checkpoint by validation loss.

Usage:
    python train_translation.py --arch gru  --embedding glove
    python train_translation.py --arch lstm --embedding glove
    python train_translation.py --arch gru  --embedding onehot
    python train_translation.py --arch lstm --embedding onehot

Run from your Homework 3 directory.
"""

import os
import re
import json
import math
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

from dataset_translation  import build_dataloaders, load_vocab
from models_translation   import build_seq2seq

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = "multi30k"
PROCESSED_DIR = "processed_translation"
GLOVE_DIR     = "glove"
CKPT_DIR      = "checkpoints_translation"

EN_VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab_en.json")
DE_VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab_de.json")

# ── Hyperparameters ───────────────────────────────────────────────────────────
EMBED_DIM   = 256
HIDDEN_DIM  = 512
N_LAYERS    = 2
DROPOUT     = 0.3
GLOVE_DIM   = 100
ONEHOT_PROJ = 256
BATCH_SIZE  = 128
N_EPOCHS    = 10
LR          = 0.001
CLIP        = 1.0
TF_RATIO    = 0.5    # teacher forcing ratio
LOG_EVERY   = 50     # print progress every N batches

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_IDX = 0


# ── Embedding helpers ─────────────────────────────────────────────────────────

def load_glove(glove_dir: str, dim: int) -> dict:
    path = os.path.join(glove_dir, f"glove.6B.{dim}d.txt")
    glove = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            glove[parts[0]] = np.array(parts[1:], dtype=np.float32)
    print(f"[INFO] Loaded {len(glove):,} GloVe vectors.")
    return glove


def apply_glove_embedding(embedding: nn.Embedding, vocab: dict, glove: dict) -> int:
    """Fill embedding weights with GloVe vectors where available."""
    found = 0
    with torch.no_grad():
        for word, idx in vocab.items():
            if word in glove:
                embedding.weight[idx] = torch.tensor(glove[word])
                found += 1
    return found


class OneHotProjected(nn.Module):
    """One-hot embedding with a linear projection to a smaller dimension."""
    def __init__(self, vocab_size: int, proj_dim: int, pad_idx: int = 0):
        super().__init__()
        emb = nn.Embedding(vocab_size, vocab_size, padding_idx=pad_idx)
        with torch.no_grad():
            emb.weight.copy_(torch.eye(vocab_size))
        emb.weight.requires_grad = False
        self.embedding  = emb
        self.projection = nn.Linear(vocab_size, proj_dim, bias=False)

    def forward(self, x):
        return self.projection(self.embedding(x).float())


def setup_embeddings(model, embedding_type, en_vocab, de_vocab, glove=None):
    """Replace model encoder/decoder embeddings with the chosen type."""
    src_vocab_size = len(en_vocab)
    tgt_vocab_size = len(de_vocab)

    if embedding_type == "glove":
        # Encoder embedding — English GloVe
        enc_found = apply_glove_embedding(model.encoder.embedding, en_vocab, glove)
        enc_cov   = enc_found / src_vocab_size * 100
        # Decoder embedding — German GloVe
        dec_found = apply_glove_embedding(model.decoder.embedding, de_vocab, glove)
        dec_cov   = dec_found / tgt_vocab_size * 100
        print(f"[INFO] EN GloVe coverage: {enc_found:,}/{src_vocab_size:,} ({enc_cov:.1f}%)")
        print(f"[INFO] DE GloVe coverage: {dec_found:,}/{tgt_vocab_size:,} ({dec_cov:.1f}%)")

    elif embedding_type == "onehot":
        model.encoder.embedding = OneHotProjected(src_vocab_size, ONEHOT_PROJ).to(DEVICE)
        model.decoder.embedding = OneHotProjected(tgt_vocab_size, ONEHOT_PROJ).to(DEVICE)


# ── Train / eval steps ────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimiser, epoch):
    model.train()
    total_loss = 0.0
    start = time.time()

    for batch_idx, (src, tgt) in enumerate(loader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimiser.zero_grad()

        # outputs: (batch, tgt_len, tgt_vocab_size)
        outputs = model(src, tgt)

        # Flatten for CrossEntropyLoss, skip the <SOS> position (index 0)
        # outputs[:, 1:] aligns with tgt[:, 1:] (the words after <SOS>)
        output_flat = outputs[:, 1:].contiguous().view(-1, outputs.size(-1))
        tgt_flat    = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(output_flat, tgt_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimiser.step()

        total_loss += loss.item()

        if (batch_idx + 1) % LOG_EVERY == 0:
            avg     = total_loss / (batch_idx + 1)
            elapsed = time.time() - start
            print(
                f"  Epoch {epoch} | Batch {batch_idx+1:>3}/{len(loader)} | "
                f"Loss: {avg:.4f} | PPL: {math.exp(avg):.1f} | "
                f"Elapsed: {elapsed:.0f}s"
            )

    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for src, tgt in loader:
        src, tgt    = src.to(DEVICE), tgt.to(DEVICE)
        outputs     = model(src, tgt)
        output_flat = outputs[:, 1:].contiguous().view(-1, outputs.size(-1))
        tgt_flat    = tgt[:, 1:].contiguous().view(-1)
        total_loss += criterion(output_flat, tgt_flat).item()

    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)


# ── Main ──────────────────────────────────────────────────────────────────────

def train(arch: str, embedding_type: str) -> None:
    run_name = f"{arch}_{embedding_type}"
    print(f"\n{'='*60}")
    print(f"  Training: {run_name.upper()}  |  Device: {DEVICE}")
    print(f"{'='*60}")

    en_vocab = load_vocab(EN_VOCAB_PATH)
    de_vocab = load_vocab(DE_VOCAB_PATH)

    train_loader, val_loader, _ = build_dataloaders(DATA_DIR, en_vocab, de_vocab, BATCH_SIZE)

    embed_dim = GLOVE_DIM if embedding_type == "glove" else ONEHOT_PROJ
    model = build_seq2seq(
        arch, len(en_vocab), len(de_vocab),
        embed_dim, HIDDEN_DIM, N_LAYERS, DROPOUT, TF_RATIO
    ).to(DEVICE)

    # Load GloVe once if needed
    glove = None
    if embedding_type == "glove":
        glove = load_glove(GLOVE_DIR, GLOVE_DIM)

    setup_embeddings(model, embedding_type, en_vocab, de_vocab, glove)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, N_EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{N_EPOCHS} ──────────────────────────────────")

        train_loss, train_ppl = train_epoch(model, train_loader, criterion, optimiser, epoch)
        val_loss,   val_ppl   = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"\n  Summary → Train loss: {train_loss:.4f} | Train PPL: {train_ppl:.1f} | "
              f"Val loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}")

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss, "train_ppl": train_ppl,
            "val_loss":   val_loss,   "val_ppl":   val_ppl,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(CKPT_DIR, f"{run_name}_best.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_ppl":     val_ppl,
                "arch":        arch,
                "embedding":   embedding_type,
            }, ckpt_path)
            print(f"  ✓ New best saved → {ckpt_path}")

    history_path = os.path.join(CKPT_DIR, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[DONE] {run_name} complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",      choices=["lstm", "gru"],     required=True)
    parser.add_argument("--embedding", choices=["glove", "onehot"], required=True)
    args = parser.parse_args()
    train(args.arch, args.embedding)