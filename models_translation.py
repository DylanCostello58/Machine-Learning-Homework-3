"""
Step 4: Seq2Seq model definition.

Architecture:
    Encoder  : reads the full English sentence → produces a context vector
    Decoder  : takes the context vector and generates the German translation
    Seq2Seq  : wraps Encoder + Decoder and handles the training loop logic

Both LSTM and GRU variants are supported via the arch argument.

Run from your Homework 3 directory:
    python models_translation.py
"""

import random
import torch
import torch.nn as nn


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Encodes the source (English) sentence into a context vector.

    Passes the full sentence through an embedding layer and RNN.
    The final hidden state(s) of the RNN summarise the entire input sentence
    and are passed to the Decoder as the initial hidden state.

    Args:
        vocab_size  : size of the source vocabulary
        embed_dim   : embedding dimension
        hidden_dim  : RNN hidden state size
        n_layers    : number of stacked RNN layers
        dropout     : dropout probability
        arch        : 'lstm' or 'gru'
        pad_idx     : index of <PAD> token
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        hidden_dim: int,
        n_layers:   int,
        dropout:    float,
        arch:       str = "gru",
        pad_idx:    int = 0,
    ):
        super().__init__()
        self.arch      = arch.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout   = nn.Dropout(dropout)

        rnn_cls = nn.LSTM if self.arch == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, src):
        """
        Args:
            src : LongTensor (batch, src_len)
        Returns:
            hidden : final hidden state — passed to Decoder as initial state
                     GRU  → tensor  (n_layers, batch, hidden_dim)
                     LSTM → tuple of (h_n, c_n), each (n_layers, batch, hidden_dim)
        """
        embedded = self.dropout(self.embedding(src))   # (batch, src_len, embed_dim)
        _, hidden = self.rnn(embedded)                 # discard output, keep hidden
        return hidden


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Generates the target (German) sentence one word at a time.

    At each step the decoder receives:
      - the previous target token (teacher forcing during training)
      - the current hidden state

    And produces:
      - a prediction over the target vocabulary
      - an updated hidden state

    Args: same as Encoder plus tgt_vocab_size for the output projection
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        hidden_dim: int,
        n_layers:   int,
        dropout:    float,
        arch:       str = "gru",
        pad_idx:    int = 0,
    ):
        super().__init__()
        self.arch      = arch.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout   = nn.Dropout(dropout)

        rnn_cls = nn.LSTM if self.arch == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_token, hidden):
        """
        One decoding step.

        Args:
            tgt_token : LongTensor (batch,) — the previous target token
            hidden    : current hidden state from encoder or previous decoder step

        Returns:
            logits : (batch, vocab_size) — unnormalised scores over target vocab
            hidden : updated hidden state
        """
        # Add sequence dimension: (batch,) → (batch, 1)
        tgt_token = tgt_token.unsqueeze(1)
        embedded  = self.dropout(self.embedding(tgt_token))  # (batch, 1, embed_dim)
        output, hidden = self.rnn(embedded, hidden)          # (batch, 1, hidden_dim)
        logits = self.fc(output.squeeze(1))                  # (batch, vocab_size)
        return logits, hidden


# ── Seq2Seq ───────────────────────────────────────────────────────────────────

class Seq2Seq(nn.Module):
    """
    Combines Encoder and Decoder into a full sequence-to-sequence model.

    During training we use teacher forcing: with probability `teacher_forcing_ratio`
    we feed the ground-truth previous token to the decoder instead of its own
    prediction. This speeds up training and stabilises early learning.

    Args:
        encoder              : Encoder instance
        decoder              : Decoder instance
        tgt_vocab_size       : size of the target vocabulary
        teacher_forcing_ratio: probability of using ground truth as next input
    """

    def __init__(
        self,
        encoder:               Encoder,
        decoder:               Decoder,
        tgt_vocab_size:        int,
        teacher_forcing_ratio: float = 0.5,
    ):
        super().__init__()
        self.encoder               = encoder
        self.decoder               = decoder
        self.tgt_vocab_size        = tgt_vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, tgt):
        """
        Args:
            src : LongTensor (batch, src_len) — encoded English sentence
            tgt : LongTensor (batch, tgt_len) — encoded German sentence
                  (tgt[:, 0] is always <SOS>)

        Returns:
            outputs : FloatTensor (batch, tgt_len, tgt_vocab_size)
                      logits at each decoder step
        """
        batch_size  = src.size(0)
        tgt_len     = tgt.size(1)
        device      = src.device

        # Store decoder outputs at each step
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=device)

        # Encode the full source sentence
        hidden = self.encoder(src)

        # First decoder input is always <SOS>
        dec_input = tgt[:, 0]   # (batch,)

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(dec_input, hidden)
            outputs[:, t, :] = logits

            # Teacher forcing: use ground truth or model prediction as next input
            use_teacher = random.random() < self.teacher_forcing_ratio
            dec_input   = tgt[:, t] if use_teacher else logits.argmax(dim=-1)

        return outputs


# ── Factory ───────────────────────────────────────────────────────────────────

def build_seq2seq(
    arch:           str,
    src_vocab_size: int,
    tgt_vocab_size: int,
    embed_dim:      int   = 256,
    hidden_dim:     int   = 512,
    n_layers:       int   = 2,
    dropout:        float = 0.3,
    tf_ratio:       float = 0.5,
    pad_idx:        int   = 0,
) -> Seq2Seq:
    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, n_layers, dropout, arch, pad_idx)
    decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, n_layers, dropout, arch, pad_idx)
    return Seq2Seq(encoder, decoder, tgt_vocab_size, tf_ratio)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SRC_VOCAB_SIZE = 5_895
    TGT_VOCAB_SIZE = 7_842
    BATCH_SIZE     = 4
    SRC_LEN        = 15
    TGT_LEN        = 13

    for arch in ["gru", "lstm"]:
        model = build_seq2seq(arch, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        src = torch.randint(0, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_LEN), device=DEVICE)
        tgt = torch.randint(0, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_LEN), device=DEVICE)

        outputs = model(src, tgt)

        print(f"\n── {arch.upper()} Seq2Seq ──────────────────────────────────")
        print(f"  Trainable params : {n_params:,}")
        print(f"  src shape        : {src.shape}")
        print(f"  tgt shape        : {tgt.shape}")
        print(f"  output shape     : {outputs.shape}   (batch × tgt_len × tgt_vocab)")
        assert outputs.shape == (BATCH_SIZE, TGT_LEN, TGT_VOCAB_SIZE), "Shape mismatch!"
        print(f"  Shape check      : ✓")

    print("\n[DONE] Seq2Seq models look good. Next step: training loop.")