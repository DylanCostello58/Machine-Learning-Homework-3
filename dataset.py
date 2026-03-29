"""
Step 3: PyTorch Dataset and DataLoader.
Turns the token ID sequence into (input, target) pairs for next-word prediction.
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("processed")
VOCAB_PATH    = os.path.join(PROCESSED_DIR, "vocab.json")
TOKENS_PATH   = os.path.join(PROCESSED_DIR, "tokens.json")

SEQ_LEN    = 50       # number of tokens in each input sequence
BATCH_SIZE = 64       # sequences per batch (reduce to 32 if you run out of memory)
TRAIN_FRAC = 0.9      # 90% train, 10% validation

# Automatically use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


class ShakespeareDataset(Dataset):
    """
    Sliding-window dataset for next-word prediction.

    Each item is a pair:
      input  : token IDs [i   : i+seq_len]      (LongTensor of shape [seq_len])
      target : token IDs [i+1 : i+seq_len+1]    (LongTensor of shape [seq_len])

    The model sees 'input' and must predict 'target' one step ahead at each position.
    """

    def __init__(self, token_ids: list[int], seq_len: int):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len   = seq_len

    def __len__(self) -> int:
        # Last valid start index is len - seq_len - 1
        # (we need seq_len tokens for input AND one more for the final target)
        return len(self.token_ids) - self.seq_len - 1

    def __getitem__(self, idx: int):
        x = self.token_ids[idx           : idx + self.seq_len    ]
        y = self.token_ids[idx + 1       : idx + self.seq_len + 1]
        return x, y


def load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tokens(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    token_ids:  list[int],
    seq_len:    int,
    batch_size: int,
    train_frac: float,
) -> tuple[DataLoader, DataLoader]:
    """Split the dataset into train/val and return DataLoaders for each."""
    dataset    = ShakespeareDataset(token_ids, seq_len)
    train_size = int(len(dataset) * train_frac)
    val_size   = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


def sanity_check(loader: DataLoader, idx2word: dict) -> None:
    """Print one batch to confirm shapes and content look right."""
    x_batch, y_batch = next(iter(loader))
    print("\n── Sanity check (first batch) ────────────────────────")
    print(f"  x shape : {x_batch.shape}   (batch_size × seq_len)")
    print(f"  y shape : {y_batch.shape}")
    # Decode the first sequence in the batch back to words
    x_words = [idx2word.get(i.item(), "<UNK>") for i in x_batch[0]]
    y_words = [idx2word.get(i.item(), "<UNK>") for i in y_batch[0]]
    print(f"\n  Input  sequence: {' '.join(x_words[:15])} ...")
    print(f"  Target sequence: {' '.join(y_words[:15])} ...")
    print(f"\n  (Notice target is shifted one word to the right of input)")


if __name__ == "__main__":
    print(f"[INFO] Using device: {DEVICE}")

    print("[INFO] Loading vocab and tokens...")
    vocab     = load_vocab(VOCAB_PATH)
    token_ids = load_tokens(TOKENS_PATH)
    idx2word  = {v: k for k, v in vocab.items()}   # reverse lookup

    print("[INFO] Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        token_ids, SEQ_LEN, BATCH_SIZE, TRAIN_FRAC
    )

    print(f"\n── DataLoader statistics ─────────────────────────────")
    print(f"  Total sequences : {len(train_loader.dataset) + len(val_loader.dataset):,}")
    print(f"  Train batches   : {len(train_loader):,}")
    print(f"  Val batches     : {len(val_loader):,}")
    print(f"  Sequence length : {SEQ_LEN}")
    print(f"  Batch size      : {BATCH_SIZE}")

    sanity_check(train_loader, idx2word)

    print("\n[DONE] DataLoader ready. Next step: model definitions (LSTM & GRU).")