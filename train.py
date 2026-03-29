"""
Step 6: Training loop.

Trains any combination of architecture (lstm/gru) and embedding (glove/onehot).
Tracks loss and perplexity, saves the best checkpoint by validation loss.

Usage examples:
    python train.py --arch lstm --embedding glove
    python train.py --arch gru  --embedding glove
    python train.py --arch lstm --embedding onehot
    python train.py --arch gru  --embedding onehot
"""

import os
import json
import math
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset    import ShakespeareDataset, load_tokens, load_vocab, build_dataloaders
from models     import build_model
from embeddings import get_glove_embedding, get_onehot_embedding

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("processed")
GLOVE_DIR     = os.path.join("glove")
CKPT_DIR      = os.path.join("checkpoints")
VOCAB_PATH    = os.path.join(PROCESSED_DIR, "vocab.json")
TOKENS_PATH   = os.path.join(PROCESSED_DIR, "tokens.json")

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN        = 50
BATCH_SIZE     = 64
TRAIN_FRAC     = 0.9
HIDDEN_DIM     = 512
N_LAYERS       = 2
DROPOUT        = 0.3
GLOVE_DIM      = 100
ONEHOT_PROJ    = 256   # project one-hot down to this dim before the RNN (see note below)
LEARNING_RATE  = 0.001
N_EPOCHS       = 10
CLIP           = 1.0   # gradient clipping max norm
LOG_EVERY      = 200   # print progress every N batches
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── One-hot projection ────────────────────────────────────────────────────────
class OneHotProjected(nn.Module):
    """
    One-hot embedding + a linear projection layer.

    A raw one-hot embedding has dim=17440 which makes the LSTM/GRU input huge
    and training very slow on CPU. We add a small trainable Linear(17440 → 256)
    right after the embedding so the RNN only sees 256-dim vectors — same as
    if we had set embed_dim=256, but the embedding layer itself remains one-hot
    (frozen identity matrix).

    This is mathematically equivalent to a learned embedding initialised from
    one-hot, so it still satisfies the assignment requirement of using one-hot
    encoding — and it's practical to train.
    """
    def __init__(self, vocab_size: int, proj_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding  = get_onehot_embedding(vocab_size, pad_idx)
        self.projection = nn.Linear(vocab_size, proj_dim, bias=False)

    def forward(self, x):
        # x: (batch, seq_len) → (batch, seq_len, vocab_size) → (batch, seq_len, proj_dim)
        return self.projection(self.embedding(x).float())


# ── Model factory with embeddings ─────────────────────────────────────────────
def build_model_with_embedding(arch: str, embedding_type: str, vocab: dict) -> nn.Module:
    vocab_size = len(vocab)

    if embedding_type == "glove":
        embed_dim = GLOVE_DIM
        model = build_model(arch, vocab_size, embed_dim, HIDDEN_DIM, N_LAYERS, DROPOUT)
        glove_emb = get_glove_embedding(vocab, GLOVE_DIR, dim=GLOVE_DIM)
        # Copy GloVe weights into model's embedding layer
        model.embedding = glove_emb

    elif embedding_type == "onehot":
        # Replace the model's embedding with the projected one-hot module
        embed_dim = ONEHOT_PROJ
        model = build_model(arch, vocab_size, embed_dim, HIDDEN_DIM, N_LAYERS, DROPOUT)
        model.embedding = OneHotProjected(vocab_size, ONEHOT_PROJ)

    else:
        raise ValueError(f"Unknown embedding type '{embedding_type}'. Use 'glove' or 'onehot'.")

    return model.to(DEVICE)


# ── Training & evaluation steps ───────────────────────────────────────────────
def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    epoch:     int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    start      = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimiser.zero_grad()

        # Init fresh hidden state each batch (truncated BPTT)
        hidden = model.init_hidden(x.size(0), DEVICE)
        # Detach hidden so gradients don't flow back through previous batches
        if isinstance(hidden, tuple):
            hidden = tuple(h.detach() for h in hidden)
        else:
            hidden = hidden.detach()

        logits, _ = model(x, hidden)   # (batch, seq_len, vocab_size)

        # Reshape for CrossEntropyLoss: (batch*seq_len, vocab_size) vs (batch*seq_len,)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        loss.backward()
        # Gradient clipping prevents exploding gradients — common with RNNs
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimiser.step()

        total_loss += loss.item()

        if (batch_idx + 1) % LOG_EVERY == 0:
            avg  = total_loss / (batch_idx + 1)
            ppl  = math.exp(avg)
            elapsed = time.time() - start
            print(
                f"  Epoch {epoch} | Batch {batch_idx+1:>5}/{len(loader)} | "
                f"Loss: {avg:.4f} | Perplexity: {ppl:.1f} | "
                f"Elapsed: {elapsed:.0f}s"
            )

    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        hidden = model.init_hidden(x.size(0), DEVICE)
        if isinstance(hidden, tuple):
            hidden = tuple(h.detach() for h in hidden)
        else:
            hidden = hidden.detach()

        logits, _ = model(x, hidden)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)


# ── Main training loop ────────────────────────────────────────────────────────
def train(arch: str, embedding_type: str) -> None:
    run_name = f"{arch}_{embedding_type}"
    print(f"\n{'='*60}")
    print(f"  Training: {run_name.upper()}  |  Device: {DEVICE}")
    print(f"{'='*60}")

    # Load data
    vocab     = load_vocab(VOCAB_PATH)
    token_ids = load_tokens(TOKENS_PATH)
    train_loader, val_loader = build_dataloaders(token_ids, SEQ_LEN, BATCH_SIZE, TRAIN_FRAC)

    # Build model
    model = build_model_with_embedding(arch, embedding_type, vocab)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore <PAD> in loss
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
            "epoch": epoch,
            "train_loss": train_loss, "train_ppl": train_ppl,
            "val_loss":   val_loss,   "val_ppl":   val_ppl,
        })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(CKPT_DIR, f"{run_name}_best.pt")
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":   val_loss,
                "val_ppl":    val_ppl,
                "arch":       arch,
                "embedding":  embedding_type,
            }, ckpt_path)
            print(f"  ✓ New best model saved → {ckpt_path}")

    # Save training history for later comparison
    history_path = os.path.join(CKPT_DIR, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history saved → {history_path}")
    print(f"\n[DONE] {run_name} training complete. Best val loss: {best_val_loss:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Shakespeare language model")
    parser.add_argument("--arch",      choices=["lstm", "gru"],          required=True)
    parser.add_argument("--embedding", choices=["glove", "onehot"],      required=True)
    args = parser.parse_args()

    train(args.arch, args.embedding)