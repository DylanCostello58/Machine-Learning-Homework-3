"""
Step 4: Model definitions — LSTM and GRU architectures.
Both models share the same interface so they can be swapped in the training loop
with a single argument change.
"""

import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    """
    LSTM-based language model for next-word prediction.

    Architecture:
        Embedding → LSTM (n_layers) → Dropout → Linear → logits

    Args:
        vocab_size   : number of unique tokens (size of embedding table)
        embed_dim    : dimension of each token embedding vector
        hidden_dim   : number of features in the LSTM hidden state
        n_layers     : number of stacked LSTM layers
        dropout      : dropout probability (applied between LSTM layers and before Linear)
        pad_idx      : index of <PAD> token (its embedding is frozen at zero)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        hidden_dim: int,
        n_layers:   int,
        dropout:    float,
        pad_idx:    int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # batch_first=True → tensors are (batch, seq_len, features) throughout
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,  # dropout only between layers
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x      : LongTensor of shape (batch, seq_len)
            hidden : optional tuple of (h_0, c_0) to continue generation from a state

        Returns:
            logits : FloatTensor of shape (batch, seq_len, vocab_size)
            hidden : updated hidden state tuple — carry this between batches
                     during text generation
        """
        embedded = self.dropout(self.embedding(x))        # (batch, seq_len, embed_dim)
        output, hidden = self.lstm(embedded, hidden)      # (batch, seq_len, hidden_dim)
        output  = self.dropout(output)
        logits  = self.fc(output)                         # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Return zeroed (h_0, c_0) for the start of a new sequence."""
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)


class GRULanguageModel(nn.Module):
    """
    GRU-based language model for next-word prediction.

    Architecture is identical to the LSTM model except GRU only has one hidden
    state (no cell state), making it slightly simpler and often faster to train.

    Args: same as LSTMLanguageModel
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        hidden_dim: int,
        n_layers:   int,
        dropout:    float,
        pad_idx:    int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x      : LongTensor of shape (batch, seq_len)
            hidden : optional h_0 tensor to continue from a previous state

        Returns:
            logits : FloatTensor of shape (batch, seq_len, vocab_size)
            hidden : updated hidden state — just a tensor (not a tuple like LSTM)
        """
        embedded = self.dropout(self.embedding(x))      # (batch, seq_len, embed_dim)
        output, hidden = self.gru(embedded, hidden)     # (batch, seq_len, hidden_dim)
        output = self.dropout(output)
        logits = self.fc(output)                        # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Return zeroed h_0 for the start of a new sequence."""
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)


def build_model(
    arch:       str,
    vocab_size: int,
    embed_dim:  int  = 256,
    hidden_dim: int  = 512,
    n_layers:   int  = 2,
    dropout:    float = 0.3,
    pad_idx:    int  = 0,
) -> nn.Module:
    """
    Factory function — returns an LSTM or GRU model by name.
    Use arch='lstm' or arch='gru'.
    """
    arch = arch.lower()
    if arch == "lstm":
        return LSTMLanguageModel(vocab_size, embed_dim, hidden_dim, n_layers, dropout, pad_idx)
    elif arch == "gru":
        return GRULanguageModel(vocab_size, embed_dim, hidden_dim, n_layers, dropout, pad_idx)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose 'lstm' or 'gru'.")


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB_SIZE = 17_440
    BATCH_SIZE = 4
    SEQ_LEN    = 50

    for arch in ["lstm", "gru"]:
        model = build_model(arch, VOCAB_SIZE).to(DEVICE)

        # Count trainable parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Run a dummy forward pass
        dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        hidden      = model.init_hidden(BATCH_SIZE, DEVICE)
        logits, _   = model(dummy_input, hidden)

        print(f"\n── {arch.upper()} ───────────────────────────────────────────")
        print(f"  Trainable params : {n_params:,}")
        print(f"  Input shape      : {dummy_input.shape}")
        print(f"  Output shape     : {logits.shape}   (batch × seq_len × vocab_size)")
        assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), "Shape mismatch!"
        print(f"  Shape check      : ✓")

    print("\n[DONE] Models look good. Next step: embeddings (GloVe + one-hot).")