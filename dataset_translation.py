"""
Step 3: PyTorch Dataset and DataLoader for English-German translation.

Each item is an (english, german) sentence pair encoded as token ID tensors.
Since sentences have different lengths, we pad them to the longest sentence
in each batch using a custom collate function.

Run from your Homework 3 directory:
    python dataset_translation.py
"""

import os
import re
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = "multi30k"
PROCESSED_DIR = "processed_translation"
EN_VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab_en.json")
DE_VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab_de.json")

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
MAX_LEN    = 50    # skip sentence pairs where either side exceeds this length
# ──────────────────────────────────────────────────────────────────────────────

PAD_IDX = 0
SOS_IDX = 2
EOS_IDX = 3


def tokenise(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"([.,!?;:\"\'()\[\]{}—\-])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def encode(tokens: list[str], vocab: dict) -> list[int]:
    """Convert tokens to IDs, wrapping with <SOS> and <EOS>."""
    unk = vocab["<UNK>"]
    ids = [vocab["<SOS>"]]
    ids += [vocab.get(t, unk) for t in tokens]
    ids += [vocab["<EOS>"]]
    return ids


class TranslationDataset(Dataset):
    """
    Dataset of (source, target) sentence pairs.

    Each item is a tuple of two LongTensors:
      src : encoded English sentence  [<SOS>, w1, w2, ..., wN, <EOS>]
      tgt : encoded German sentence   [<SOS>, w1, w2, ..., wM, <EOS>]

    The DataLoader will batch these together using the collate_fn below,
    padding shorter sentences so all items in a batch have the same length.
    """

    def __init__(
        self,
        en_lines:  list[str],
        de_lines:  list[str],
        en_vocab:  dict,
        de_vocab:  dict,
        max_len:   int = MAX_LEN,
    ):
        self.pairs = []
        skipped = 0

        for en, de in zip(en_lines, de_lines):
            en_tokens = tokenise(en)
            de_tokens = tokenise(de)

            # Skip very long sentences — they slow training and are rare
            if len(en_tokens) > max_len or len(de_tokens) > max_len:
                skipped += 1
                continue

            src = torch.tensor(encode(en_tokens, en_vocab), dtype=torch.long)
            tgt = torch.tensor(encode(de_tokens, de_vocab), dtype=torch.long)
            self.pairs.append((src, tgt))

        if skipped:
            print(f"[INFO] Skipped {skipped} pairs exceeding max_len={max_len}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """
    Pad a batch of (src, tgt) pairs to the same length.

    pad_sequence stacks tensors and pads shorter ones with PAD_IDX (0)
    so every tensor in the batch has the same length as the longest one.

    Returns:
        src_batch : (batch_size, max_src_len)
        tgt_batch : (batch_size, max_tgt_len)
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def build_dataloaders(
    data_dir:   str,
    en_vocab:   dict,
    de_vocab:   dict,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders."""
    loaders = []
    for split in ["train", "val", "test"]:
        en_lines = load_lines(os.path.join(data_dir, f"{split}.en"))
        de_lines = load_lines(os.path.join(data_dir, f"{split}.de"))
        dataset  = TranslationDataset(en_lines, de_lines, en_vocab, de_vocab)
        shuffle  = (split == "train")
        loader   = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        loaders.append(loader)
        print(f"[INFO] {split:5s} — {len(dataset):,} pairs | {len(loader):,} batches")

    return tuple(loaders)


def sanity_check(loader: DataLoader, en_vocab: dict, de_vocab: dict) -> None:
    """Decode and print the first sentence pair from the first batch."""
    en_idx2word = {v: k for k, v in en_vocab.items()}
    de_idx2word = {v: k for k, v in de_vocab.items()}

    src_batch, tgt_batch = next(iter(loader))

    print(f"\n── Sanity check ──────────────────────────────────────")
    print(f"  src batch shape : {src_batch.shape}  (batch × src_len)")
    print(f"  tgt batch shape : {tgt_batch.shape}  (batch × tgt_len)")

    src_words = [en_idx2word.get(i.item(), "<UNK>") for i in src_batch[0]]
    tgt_words = [de_idx2word.get(i.item(), "<UNK>") for i in tgt_batch[0]]

    print(f"\n  EN : {' '.join(src_words)}")
    print(f"  DE : {' '.join(tgt_words)}")


if __name__ == "__main__":
    en_vocab = load_vocab(EN_VOCAB_PATH)
    de_vocab = load_vocab(DE_VOCAB_PATH)

    print("[INFO] Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        DATA_DIR, en_vocab, de_vocab, BATCH_SIZE
    )

    sanity_check(train_loader, en_vocab, de_vocab)

    print("\n[DONE] DataLoaders ready. Next step: seq2seq model definition.")