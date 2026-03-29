"""
Step 6: Translation generation and BLEU score evaluation.

Loads a trained checkpoint, translates the test set, computes BLEU score,
and shows qualitative translation examples.

Usage:
    python evaluate_translation.py --checkpoint checkpoints_translation/gru_glove_best.pt
    python evaluate_translation.py --checkpoint checkpoints_translation/lstm_glove_best.pt
    python evaluate_translation.py --checkpoint checkpoints_translation/gru_onehot_best.pt
    python evaluate_translation.py --checkpoint checkpoints_translation/lstm_onehot_best.pt

To compare all 4 runs at once:
    python evaluate_translation.py --all

Run from your Homework 3 directory.
"""

import os
import re
import json
import argparse

import torch
import sacrebleu

from dataset_translation import load_vocab, load_lines, TranslationDataset, collate_fn
from models_translation  import build_seq2seq
from train_translation   import setup_embeddings, load_glove, OneHotProjected

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = "multi30k"
PROCESSED_DIR = "processed_translation"
GLOVE_DIR     = "glove"
CKPT_DIR      = "checkpoints_translation"

EN_VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab_en.json")
DE_VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab_de.json")

# ── Model config (must match train_translation.py) ────────────────────────────
EMBED_DIM   = 256
HIDDEN_DIM  = 512
N_LAYERS    = 2
DROPOUT     = 0.0   # disabled at inference
GLOVE_DIM   = 100
ONEHOT_PROJ = 256
MAX_LEN     = 50
SOS_IDX     = 2
EOS_IDX     = 3
PAD_IDX     = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"([.,!?;:\"\'()\[\]{}—\-])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def load_checkpoint(ckpt_path: str, en_vocab: dict, de_vocab: dict) -> torch.nn.Module:
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

    print(f"[INFO] Loaded {arch.upper()} + {embedding_type} "
          f"(epoch {ckpt['epoch']} | val loss: {ckpt['val_loss']:.4f})")
    return model, arch, embedding_type


def translate_sentence(
    model:    torch.nn.Module,
    sentence: str,
    en_vocab: dict,
    de_vocab: dict,
    max_len:  int = MAX_LEN,
) -> list[str]:
    """
    Translate a single English sentence to German tokens.

    At inference time we don't have ground truth tokens to feed the decoder,
    so we always feed the model's own previous prediction (no teacher forcing).
    We stop when the model outputs <EOS> or we hit max_len.
    """
    model.eval()
    idx2de = {v: k for k, v in de_vocab.items()}

    # Encode source
    tokens = tokenise(sentence)
    ids    = [en_vocab["<SOS>"]] + \
             [en_vocab.get(t, en_vocab["<UNK>"]) for t in tokens] + \
             [en_vocab["<EOS>"]]
    src    = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        hidden = model.encoder(src)

    # Decode one token at a time
    dec_input  = torch.tensor([SOS_IDX], dtype=torch.long).to(DEVICE)
    translated = []

    with torch.no_grad():
        for _ in range(max_len):
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
    """Join tokens into a readable string, fixing punctuation spacing."""
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


# ── BLEU evaluation ───────────────────────────────────────────────────────────

def compute_bleu(
    model:    torch.nn.Module,
    en_vocab: dict,
    de_vocab: dict,
    split:    str = "test",
) -> float:
    """
    Translate the full test set and compute corpus BLEU score.

    BLEU (Bilingual Evaluation Understudy) measures how similar the machine
    translations are to the human reference translations.
    Score ranges 0-100. For reference:
      < 10  : almost unusable
      10-20 : hard to understand
      20-30 : getting the gist
      30-40 : understandable, some errors
      40-50 : high quality
      > 50  : better than most humans
    A simple seq2seq without attention typically scores 10-20 on Multi30K.
    """
    en_lines  = load_lines(os.path.join(DATA_DIR, f"{split}.en"))
    de_lines  = load_lines(os.path.join(DATA_DIR, f"{split}.de"))

    hypotheses = []   # model translations
    references = []   # ground truth translations

    print(f"[INFO] Translating {len(en_lines):,} sentences for BLEU evaluation...")

    for en, de in zip(en_lines, de_lines):
        translated = translate_sentence(model, en, en_vocab, de_vocab)
        hypotheses.append(detokenise(translated))
        references.append(de)   # sacrebleu handles tokenisation of references

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


# ── Qualitative examples ──────────────────────────────────────────────────────

def show_examples(
    model:    torch.nn.Module,
    en_vocab: dict,
    de_vocab: dict,
    n:        int = 8,
) -> None:
    """Print n example translations from the test set."""
    en_lines = load_lines(os.path.join(DATA_DIR, "test.en"))
    de_lines = load_lines(os.path.join(DATA_DIR, "test.de"))

    # Pick a mix of short and longer sentences for variety
    indices = [0, 1, 2, 10, 50, 100, 200, 500][:n]

    print(f"\n── Qualitative examples ──────────────────────────────")
    for i in indices:
        en  = en_lines[i]
        ref = de_lines[i]
        hyp = detokenise(translate_sentence(model, en, en_vocab, de_vocab))
        print(f"\n  [{i}] EN  : {en}")
        print(f"       REF : {ref}")
        print(f"       HYP : {hyp}")


# ── Single run evaluation ─────────────────────────────────────────────────────

def evaluate_checkpoint(ckpt_path: str) -> dict:
    en_vocab = load_vocab(EN_VOCAB_PATH)
    de_vocab = load_vocab(DE_VOCAB_PATH)

    model, arch, embedding_type = load_checkpoint(ckpt_path, en_vocab, de_vocab)

    bleu = compute_bleu(model, en_vocab, de_vocab)
    print(f"\n  BLEU score: {bleu:.2f}")

    show_examples(model, en_vocab, de_vocab)

    return {"arch": arch, "embedding": embedding_type, "bleu": bleu}


# ── Compare all 4 runs ────────────────────────────────────────────────────────

def evaluate_all() -> None:
    runs = [
        "gru_glove_best.pt",
        "lstm_glove_best.pt",
        "gru_onehot_best.pt",
        "lstm_onehot_best.pt",
    ]

    results = []
    for fname in runs:
        path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(path):
            print(f"[WARN] Missing checkpoint: {path} — skipping")
            continue
        print(f"\n{'='*60}")
        print(f"  Evaluating: {fname}")
        print(f"{'='*60}")
        result = evaluate_checkpoint(path)
        results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  {'Run':<22} {'BLEU':>8}")
    print(f"{'='*60}")
    for r in results:
        name = f"{r['arch'].upper()} + {r['embedding']}"
        print(f"  {name:<22} {r['bleu']:>8.2f}")
    print(f"{'='*60}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Path to a single checkpoint .pt file")
    group.add_argument("--all",        action="store_true", help="Evaluate all 4 checkpoints")
    args = parser.parse_args()

    if args.all:
        evaluate_all()
    else:
        evaluate_checkpoint(args.checkpoint)