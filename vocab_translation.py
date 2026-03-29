"""
Step 2: Vocabulary building for English-German translation.

Builds separate vocabularies for English (source) and German (target).
Special tokens:
  <PAD> : padding                 (index 0)
  <UNK> : unknown words           (index 1)
  <SOS> : start of sequence       (index 2)
  <EOS> : end of sequence         (index 3)

Run from your Homework 3 directory:
    python vocab_translation.py
"""

import os
import re
import json
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = "multi30k"
PROCESSED_DIR = "processed_translation"

EN_VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab_en.json")
DE_VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab_de.json")

MIN_FREQ = 2   # words appearing fewer times get mapped to <UNK>
# ──────────────────────────────────────────────────────────────────────────────

PAD_TOKEN = "<PAD>"   # 0
UNK_TOKEN = "<UNK>"   # 1
SOS_TOKEN = "<SOS>"   # 2
EOS_TOKEN = "<EOS>"   # 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


def tokenise(text: str) -> list[str]:
    """
    Simple word-level tokeniser — same approach as Task 1.
    Lowercases and separates punctuation into its own token.
    """
    text = text.lower()
    text = re.sub(r"([.,!?;:\"\'()\[\]{}—\-])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def load_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def build_vocab(lines: list[str], min_freq: int) -> dict:
    """
    Build a word→index vocab from a list of sentences.
    Special tokens always occupy indices 0-3.
    """
    counts = Counter()
    for line in lines:
        counts.update(tokenise(line))

    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for word, freq in counts.most_common():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def save_vocab(vocab: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def print_stats(lang: str, vocab: dict, lines: list[str]) -> None:
    tokens_total = sum(len(tokenise(l)) for l in lines)
    unk_count    = sum(
        1 for l in lines for t in tokenise(l) if t not in vocab
    )
    print(f"\n── {lang} vocabulary ─────────────────────────────────")
    print(f"  Sentences      : {len(lines):,}")
    print(f"  Total tokens   : {tokens_total:,}")
    print(f"  Vocab size     : {len(vocab):,}  (incl. 4 special tokens)")
    print(f"  UNK rate       : {unk_count/tokens_total*100:.2f}%")
    print(f"  Sample entries : { {k: vocab[k] for k in list(vocab)[:8]} }")


if __name__ == "__main__":
    # Load training data only — vocab is built from train set
    # (val/test are held out and may contain unseen words → <UNK>)
    en_lines = load_lines(os.path.join(DATA_DIR, "train.en"))
    de_lines = load_lines(os.path.join(DATA_DIR, "train.de"))

    print("[INFO] Building English vocabulary...")
    en_vocab = build_vocab(en_lines, MIN_FREQ)

    print("[INFO] Building German vocabulary...")
    de_vocab = build_vocab(de_lines, MIN_FREQ)

    print_stats("English", en_vocab, en_lines)
    print_stats("German",  de_vocab, de_lines)

    save_vocab(en_vocab, EN_VOCAB_PATH)
    save_vocab(de_vocab, DE_VOCAB_PATH)

    print(f"\n[INFO] Saved → {EN_VOCAB_PATH}")
    print(f"[INFO] Saved → {DE_VOCAB_PATH}")
    print("\n[DONE] Vocabularies built. Next step: dataset & dataloader.")