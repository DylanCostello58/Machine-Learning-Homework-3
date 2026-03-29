"""
Step 7: Text generation.

Loads a saved checkpoint and generates Shakespeare-style text from a seed prompt.

Usage:
    python generate.py --checkpoint checkpoints/gru_glove_best.pt --prompt "to be or not" --words 100
    python generate.py --checkpoint checkpoints/lstm_glove_best.pt --prompt "shall i compare" --words 200 --temperature 0.8
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F

from models     import build_model
from embeddings import get_glove_embedding, get_onehot_embedding
from train      import OneHotProjected

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("processed")
GLOVE_DIR     = os.path.join("glove")
VOCAB_PATH    = os.path.join(PROCESSED_DIR, "vocab.json")

# ── Model config (must match train.py) ───────────────────────────────────────
HIDDEN_DIM  = 512
N_LAYERS    = 2
DROPOUT     = 0.3
GLOVE_DIM   = 100
ONEHOT_PROJ = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vocab(path: str) -> tuple[dict, dict]:
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    idx2word = {v: k for k, v in vocab.items()}
    return vocab, idx2word


def load_checkpoint(ckpt_path: str, vocab: dict) -> torch.nn.Module:
    """Load a saved checkpoint and reconstruct the model."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    arch           = ckpt["arch"]
    embedding_type = ckpt["embedding"]
    vocab_size     = len(vocab)

    if embedding_type == "glove":
        embed_dim = GLOVE_DIM
    else:
        embed_dim = ONEHOT_PROJ

    model = build_model(arch, vocab_size, embed_dim, HIDDEN_DIM, N_LAYERS, DROPOUT)

    # Rebuild the correct embedding layer
    if embedding_type == "glove":
        model.embedding = get_glove_embedding(vocab, GLOVE_DIR, dim=GLOVE_DIM)
    else:
        model.embedding = OneHotProjected(vocab_size, ONEHOT_PROJ)

    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    print(f"[INFO] Loaded {arch.upper()} + {embedding_type} checkpoint from epoch {ckpt['epoch']}")
    print(f"[INFO] Checkpoint val loss: {ckpt['val_loss']:.4f} | val PPL: {ckpt['val_ppl']:.1f}")
    return model


def tokenise_prompt(prompt: str, vocab: dict) -> list[int]:
    """Convert a text prompt into token IDs using the same rules as tokenise.py."""
    import re
    prompt = prompt.lower()
    prompt = re.sub(r"([.,!?;:\"\'()\[\]{}—\-])", r" \1 ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    words  = prompt.split()
    unk    = vocab["<UNK>"]
    ids    = [vocab.get(w, unk) for w in words]
    unknown = [w for w in words if w not in vocab]
    if unknown:
        print(f"[WARN] These prompt words aren't in vocab (mapped to <UNK>): {unknown}")
    return ids


def generate(
    model:       torch.nn.Module,
    prompt_ids:  list[int],
    vocab:       dict,
    idx2word:    dict,
    n_words:     int  = 100,
    temperature: float = 1.0,
    top_k:       int  = 50,
) -> str:
    """
    Generate text autoregressively from a seed prompt.

    Args:
        model       : trained language model
        prompt_ids  : token IDs of the seed prompt
        vocab       : word → index mapping
        idx2word    : index → word mapping
        n_words     : number of new words to generate
        temperature : controls randomness.
                        < 1.0 → more conservative/repetitive
                        = 1.0 → sample directly from model distribution
                        > 1.0 → more creative/chaotic
        top_k       : only sample from the top-k most likely next words.
                      Prevents very unlikely words from ever being picked.

    Returns:
        The full generated text as a string (prompt + generated words).
    """
    model.eval()

    # Seed the hidden state by feeding the prompt through the model
    input_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    hidden = model.init_hidden(1, DEVICE)

    with torch.no_grad():
        _, hidden = model(input_tensor, hidden)

    # The last token of the prompt is our first input for generation
    current_token = torch.tensor([[prompt_ids[-1]]], dtype=torch.long).to(DEVICE)
    generated_ids = list(prompt_ids)

    with torch.no_grad():
        for _ in range(n_words):
            logits, hidden = model(current_token, hidden)
            # logits: (1, 1, vocab_size) → (vocab_size,)
            logits = logits.squeeze() / temperature

            # Top-k filtering: zero out all logits outside the top k
            if top_k > 0:
                top_values, _ = torch.topk(logits, top_k)
                min_val = top_values[-1]
                logits  = logits.masked_fill(logits < min_val, float("-inf"))

            # Sample from the filtered distribution
            probs        = F.softmax(logits, dim=-1)
            next_token   = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_token)
            current_token = torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)

    # Decode back to words, skipping special tokens
    special = {"<PAD>", "<UNK>"}
    words   = [idx2word[i] for i in generated_ids if idx2word.get(i) not in special]
    return detokenise(words)


def detokenise(words: list[str]) -> str:
    """
    Convert a list of word tokens back into readable text.
    Handles spacing around punctuation so output doesn't look like:
        "to be , or not to be ."
    but instead:
        "to be, or not to be."
    """
    punct_no_space_before = set(".,!?;:')")
    punct_no_space_after  = set("(")
    result = []
    for i, word in enumerate(words):
        if i == 0:
            result.append(word.capitalize())
        elif word in punct_no_space_before:
            result[-1] = result[-1] + word   # attach to previous word
        elif result and result[-1].endswith("("):
            result.append(word)              # no space after open paren
        else:
            result.append(" " + word)
    return "".join(result)


def run_generation(ckpt_path: str, prompt: str, n_words: int, temperature: float) -> None:
    vocab, idx2word = load_vocab(VOCAB_PATH)
    model           = load_checkpoint(ckpt_path, vocab)
    prompt_ids      = tokenise_prompt(prompt, vocab)

    print(f"\n── Prompt ────────────────────────────────────────────")
    print(f"  '{prompt}'")
    print(f"\n── Generated text (temp={temperature}, top_k=50) ────")

    text = generate(model, prompt_ids, vocab, idx2word, n_words, temperature)
    print(f"\n{text}\n")

    # Also show a few samples at different temperatures for comparison
    print("── Temperature comparison ────────────────────────────")
    for temp in [0.5, 1.0, 1.2]:
        sample = generate(model, prompt_ids, vocab, idx2word, 30, temp)
        print(f"  temp={temp}: {sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Shakespeare text from a trained model")
    parser.add_argument("--checkpoint",  required=True,         help="Path to .pt checkpoint file")
    parser.add_argument("--prompt",      default="to be or not", help="Seed text for generation")
    parser.add_argument("--words",       type=int, default=100,  help="Number of words to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()

    run_generation(args.checkpoint, args.prompt, args.words, args.temperature)