"""
Step 2: Tokenisation and vocabulary building.
Converts the raw Shakespeare text into integer sequences for model training.
"""

import re
import json
import os
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
TEXT_PATH = os.path.join("shakespeare.txt")
OUTPUT_DIR = os.path.join("processed")

VOCAB_PATH    = os.path.join(OUTPUT_DIR, "vocab.json")
TOKENS_PATH   = os.path.join(OUTPUT_DIR, "tokens.json")

MIN_FREQ = 2   # words appearing fewer than this many times get mapped to <UNK>
# ──────────────────────────────────────────────────────────────────────────────

# Special tokens
PAD_TOKEN = "<PAD>"   # used to pad shorter sequences in a batch
UNK_TOKEN = "<UNK>"   # unknown / rare words


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()


def tokenise(text: str) -> list[str]:
    """
    Simple word-level tokeniser:
      1. Lowercase everything
      2. Insert spaces around punctuation so  "thee!" → "thee !"
      3. Split on whitespace
    """
    text = text.lower()
    # Separate punctuation from words (keep punctuation as its own token)
    text = re.sub(r"([.,!?;:\"\'()\[\]{}—\-])", r" \1 ", text)
    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def build_vocab(tokens: list[str], min_freq: int) -> tuple[dict, dict]:
    """
    Build word→index and index→word mappings.
    Words below min_freq are excluded (they'll map to <UNK> at encode time).
    """
    counts = Counter(tokens)
    # Start vocab with special tokens
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counts.most_common():   # most_common keeps order deterministic
        if freq >= min_freq:
            vocab[word] = len(vocab)
    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, idx2word


def encode(tokens: list[str], vocab: dict) -> list[int]:
    unk_idx = vocab[UNK_TOKEN]
    return [vocab.get(t, unk_idx) for t in tokens]


def save_artifacts(vocab: dict, token_ids: list[int]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(VOCAB_PATH,  "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    with open(TOKENS_PATH, "w", encoding="utf-8") as f:
        json.dump(token_ids, f)
    print(f"[INFO] Vocabulary saved  → {VOCAB_PATH}")
    print(f"[INFO] Token IDs saved   → {TOKENS_PATH}")


def print_stats(tokens: list[str], vocab: dict, token_ids: list[int]) -> None:
    print("\n── Tokenisation statistics ───────────────────────────")
    print(f"  Total tokens       : {len(tokens):,}")
    print(f"  Vocabulary size    : {len(vocab):,}  (incl. <PAD> and <UNK>)")
    print(f"  Encoded sequence   : {len(token_ids):,} integers")
    print(f"\n  Sample tokens      : {tokens[:20]}")
    print(f"  Sample IDs         : {token_ids[:20]}")


if __name__ == "__main__":
    print("[INFO] Loading text...")
    text = load_text(TEXT_PATH)

    print("[INFO] Tokenising...")
    tokens = tokenise(text)

    print("[INFO] Building vocabulary...")
    vocab, idx2word = build_vocab(tokens, min_freq=MIN_FREQ)

    print("[INFO] Encoding token sequence...")
    token_ids = encode(tokens, vocab)

    print_stats(tokens, vocab, token_ids)
    save_artifacts(vocab, token_ids)

    print("\n[DONE] Tokenisation complete. Next step: dataset & dataloader creation.")