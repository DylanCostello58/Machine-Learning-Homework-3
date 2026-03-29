"""
evaluate_translation.py — Task 2: Translation Evaluation
Evaluates all 4 checkpoints, prints BLEU scores, and shows qualitative examples.

Usage:
    python evaluate_translation.py

Run from your Homework 3 directory.
"""

import os
import re
import json

import torch
import sacrebleu

from dataset_translation import load_vocab, load_lines
from models_translation  import build_seq2seq
from train_translation   import setup_embeddings, load_glove

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = "multi30k"
PROCESSED_DIR = "processed_translation"
GLOVE_DIR     = "glove"
CKPT_DIR      = "checkpoints_translation"

EN_VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab_en.json")
DE_VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab_de.json")

RUNS = [
    "gru_glove_best.pt",
    "lstm_glove_best.pt",
    "gru_onehot_best.pt",
    "lstm_onehot_best.pt",
]

# ── Model config (must match train_translation.py) ────────────────────────────
HIDDEN_DIM  = 512
N_LAYERS    = 2
DROPOUT     = 0.0
GLOVE_DIM   = 100
ONEHOT_PROJ = 256
MAX_LEN     = 50
SOS_IDX     = 2
EOS_IDX     = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"([.,!?;:\"\'()\[\]{}—\-])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def load_checkpoint(ckpt_path: str, en_vocab: dict, de_vocab: dict):
    ckpt           = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    arch           = ckpt["arch"]
    embedding_type = ckpt["embedding"]
    embed_dim      = GLOVE_DIM if embedding_type == "glove" else ONEHOT_PROJ

    model = build_seq2seq(
        arch, len(en_vocab), len(de_vocab),
        embed_dim, HIDDEN_DIM, N_LAYERS, DROPOUT
    ).to(DEVICE)

    glove = None
    if embedding_type == "glove":
        glove = load_glove(GLOVE_DIR, GLOVE_DIM)
    setup_embeddings(model, embedding_type, en_vocab, de_vocab, glove)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, arch, embedding_type, ckpt["val_loss"]


def translate_sentence(model, sentence, en_vocab, de_vocab) -> list[str]:
    idx2de  = {v: k for k, v in de_vocab.items()}
    tokens  = tokenise(sentence)
    ids     = ([en_vocab["<SOS>"]]
               + [en_vocab.get(t, en_vocab["<UNK>"]) for t in tokens]
               + [en_vocab["<EOS>"]])
    src     = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        hidden = model.encoder(src)

    dec_input  = torch.tensor([SOS_IDX], dtype=torch.long).to(DEVICE)
    translated = []

    with torch.no_grad():
        for _ in range(MAX_LEN):
            logits, hidden = model.decoder(dec_input, hidden)
            next_token     = logits.argmax(dim=-1).item()
            if next_token == EOS_IDX:
                break
            word = idx2de.get(next_token, "<UNK>")
            if word not in {"<PAD>", "<UNK>", "<SOS>"}:
                translated.append(word)
            dec_input = torch.tensor([next_token], dtype=torch.long).to(DEVICE)

    return translated


def detokenise(tokens: list[str]) -> str:
    no_space_before = set(".,!?;:')")
    result = []
    for i, tok in enumerate(tokens):
        if i == 0:
            result.append(tok)
        elif tok in no_space_before:
            result[-1] += tok
        else:
            result.append(" " + tok)
    return "".join(result)


def compute_bleu(model, en_vocab, de_vocab) -> float:
    en_lines = load_lines(os.path.join(DATA_DIR, "test.en"))
    de_lines = load_lines(os.path.join(DATA_DIR, "test.de"))
    hypotheses = []
    references = []
    for en, de in zip(en_lines, de_lines):
        hypotheses.append(detokenise(translate_sentence(model, en, en_vocab, de_vocab)))
        references.append(de)
    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def show_examples(model, en_vocab, de_vocab, n: int = 5) -> None:
    en_lines = load_lines(os.path.join(DATA_DIR, "test.en"))
    de_lines = load_lines(os.path.join(DATA_DIR, "test.de"))
    indices  = [0, 1, 2, 10, 50][:n]
    for i in indices:
        hyp = detokenise(translate_sentence(model, en_lines[i], en_vocab, de_vocab))
        print(f"\n  [{i}] EN  : {en_lines[i]}")
        print(f"       REF : {de_lines[i]}")
        print(f"       HYP : {hyp}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    en_vocab = load_vocab(EN_VOCAB_PATH)
    de_vocab = load_vocab(DE_VOCAB_PATH)
    results  = []

    for fname in RUNS:
        path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(path):
            print(f"[WARN] Missing checkpoint: {path} — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {fname}")
        print(f"{'='*60}")

        model, arch, embedding_type, val_loss = load_checkpoint(path, en_vocab, de_vocab)

        print(f"[INFO] Computing BLEU score...")
        bleu = compute_bleu(model, en_vocab, de_vocab)
        print(f"[INFO] BLEU: {bleu:.2f}")

        show_examples(model, en_vocab, de_vocab)

        results.append({
            "name":      f"{arch.upper()} + {embedding_type}",
            "val_loss":  val_loss,
            "bleu":      bleu,
        })

    # Summary table
    print(f"\n{'='*60}")
    print(f"  {'Run':<22} {'Val Loss':>10} {'BLEU':>8}")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']:<22} {r['val_loss']:>10.4f} {r['bleu']:>8.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
