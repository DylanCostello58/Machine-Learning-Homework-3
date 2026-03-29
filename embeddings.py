"""
Step 5: Embedding setup — GloVe (pre-trained) and one-hot encodings.

This module provides two functions:
  - get_onehot_embedding()  : fixed identity-style embedding, no semantic info
  - get_glove_embedding()   : pre-trained GloVe vectors aligned to our vocabulary

Both return a nn.Embedding layer that can be plugged directly into the models
defined in models.py.
"""

import os
import json
import zipfile
import urllib.request
import numpy as np
import torch
import torch.nn as nn

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("processed")
GLOVE_DIR     = os.path.join("glove")
VOCAB_PATH    = os.path.join(PROCESSED_DIR, "vocab.json")

GLOVE_DIM     = 100   # options: 50, 100, 200, 300  (100 is a good balance)
GLOVE_URL     = f"https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP     = os.path.join(GLOVE_DIR, "glove.6B.zip")
GLOVE_TXT     = os.path.join(GLOVE_DIR, f"glove.6B.{GLOVE_DIM}d.txt")
# ──────────────────────────────────────────────────────────────────────────────


def load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── One-hot embedding ─────────────────────────────────────────────────────────

def get_onehot_embedding(vocab_size: int, pad_idx: int = 0) -> nn.Embedding:
    """
    Returns a frozen nn.Embedding where each token is represented by a
    one-hot vector of length vocab_size.

    Implementation note:
      We use an (vocab_size × vocab_size) identity matrix as the weight.
      Row i is all zeros except position i which is 1 — that's a one-hot vector.
      We freeze the weights so they never change during training.

    Downside: embed_dim = vocab_size = 17,440 → very large, slow to train.
    This is intentional — it's the baseline to compare against GloVe.
    """
    embedding = nn.Embedding(vocab_size, vocab_size, padding_idx=pad_idx)
    # Initialise with identity matrix (one-hot vectors)
    with torch.no_grad():
        embedding.weight.copy_(torch.eye(vocab_size))
    # Freeze — one-hot is fixed by definition, not learned
    embedding.weight.requires_grad = False
    return embedding


# ── GloVe embedding ───────────────────────────────────────────────────────────

def download_glove(url: str, zip_path: str, extract_dir: str) -> None:
    """Download and extract GloVe vectors if not already present."""
    os.makedirs(extract_dir, exist_ok=True)
    txt_path = os.path.join(extract_dir, f"glove.6B.{GLOVE_DIM}d.txt")

    if os.path.exists(txt_path):
        print(f"[INFO] GloVe file already exists at '{txt_path}', skipping download.")
        return

    if not os.path.exists(zip_path):
        print(f"[INFO] Downloading GloVe vectors (~822 MB) — this may take a while...")
        print(f"[INFO] Alternatively download manually from: {url}")
        print(f"[INFO] and extract to: {extract_dir}")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[INFO] Download complete.")

    print(f"[INFO] Extracting GloVe zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"[INFO] Extracted to '{extract_dir}'.")


def load_glove_vectors(glove_txt: str, dim: int) -> dict[str, np.ndarray]:
    """Parse the GloVe .txt file into a word → vector dictionary."""
    print(f"[INFO] Loading GloVe {dim}d vectors...")
    glove = {}
    with open(glove_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    print(f"[INFO] Loaded {len(glove):,} GloVe vectors.")
    return glove


def get_glove_embedding(
    vocab:    dict,
    glove_dir: str,
    dim:      int  = GLOVE_DIM,
    pad_idx:  int  = 0,
    freeze:   bool = False,   # False = fine-tune GloVe weights during training
) -> nn.Embedding:
    """
    Build a nn.Embedding layer pre-loaded with GloVe vectors.

    For words in our vocab that exist in GloVe: use the pre-trained vector.
    For words not in GloVe (rare/archaic Shakespeare words): random initialisation.

    Args:
        vocab     : word → index dict from vocab.json
        glove_dir : directory containing the extracted glove.6B.Xd.txt file
        dim       : embedding dimension (must match the .txt file)
        pad_idx   : index of <PAD> token (kept as zero vector)
        freeze    : if True, GloVe weights are not updated during training
    """
    glove_txt = os.path.join(glove_dir, f"glove.6B.{dim}d.txt")
    download_glove(GLOVE_URL, GLOVE_ZIP, glove_dir)
    glove_vectors = load_glove_vectors(glove_txt, dim)

    vocab_size = len(vocab)
    # Start with small random vectors for all words
    embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, dim)).astype(np.float32)
    # Zero out the <PAD> token
    embedding_matrix[pad_idx] = 0.0

    # Fill in GloVe vectors for words we have
    found = 0
    for word, idx in vocab.items():
        if word in glove_vectors:
            embedding_matrix[idx] = glove_vectors[word]
            found += 1

    coverage = found / vocab_size * 100
    print(f"[INFO] GloVe coverage: {found:,} / {vocab_size:,} vocab words ({coverage:.1f}%)")

    # Build the nn.Embedding from our matrix
    embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
    with torch.no_grad():
        embedding.weight.copy_(torch.tensor(embedding_matrix))
    embedding.weight.requires_grad = not freeze

    return embedding


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vocab = load_vocab(VOCAB_PATH)
    vocab_size = len(vocab)

    print("\n── One-hot embedding ─────────────────────────────────")
    oh_emb = get_onehot_embedding(vocab_size)
    print(f"  Weight shape    : {oh_emb.weight.shape}")
    print(f"  Trainable       : {oh_emb.weight.requires_grad}")
    # Check that row 5 is a one-hot vector
    row = oh_emb.weight[5].detach()
    assert row[5].item() == 1.0 and row.sum().item() == 1.0, "One-hot check failed!"
    print(f"  One-hot check   : ✓")

    print("\n── GloVe embedding ───────────────────────────────────")
    glove_emb = get_glove_embedding(vocab, GLOVE_DIR, dim=GLOVE_DIM)
    print(f"  Weight shape    : {glove_emb.weight.shape}")
    print(f"  Trainable       : {glove_emb.weight.requires_grad}")
    print(f"  Sample vector   : {glove_emb.weight[4, :6].detach().tolist()}  (first 6 dims of 'the')")

    print("\n[DONE] Embeddings ready. Next step: training loop.")