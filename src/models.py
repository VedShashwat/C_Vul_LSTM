from typing import Tuple

import torch
import torch.nn as nn


def _lstm_dropout(num_layers: int, dropout: float) -> float:
    return dropout if num_layers > 1 else 0.0


class VanillaLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=_lstm_dropout(num_layers, dropout),
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        seq_repr = h_n[-1]
        logits = self.fc(seq_repr).squeeze(-1)
        return logits


class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=_lstm_dropout(num_layers, dropout),
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)

        forward_final = h_n[-2]
        backward_final = h_n[-1]
        seq_repr = torch.cat([forward_final, backward_final], dim=1)

        logits = self.fc(seq_repr).squeeze(-1)
        return logits


class BiLSTMAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
        attention_dim: int = 64,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=_lstm_dropout(num_layers, dropout),
            bidirectional=True,
        )

        self.attn_proj = nn.Linear(hidden_size * 2, attention_dim)
        self.attn_score = nn.Linear(attention_dim, 1)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        outputs, _ = self.lstm(emb)

        attn_hidden = torch.tanh(self.attn_proj(outputs))
        attn_logits = self.attn_score(attn_hidden).squeeze(-1)

        pad_mask = x.eq(self.pad_idx)
        attn_logits = attn_logits.masked_fill(pad_mask, float("-inf"))
        all_pad_rows = pad_mask.all(dim=1)
        if all_pad_rows.any():
            attn_logits[all_pad_rows] = 0.0

        alpha = torch.softmax(attn_logits, dim=1)

        context = torch.bmm(alpha.unsqueeze(1), outputs).squeeze(1)
        logits = self.fc(context).squeeze(-1)

        return logits, alpha


def build_model(
    model_name: str,
    vocab_size: int,
    embed_dim: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    pad_idx: int,
    attention_dim: int = 64,
) -> nn.Module:
    name = model_name.lower()

    if name == "lstm":
        return VanillaLSTM(vocab_size, embed_dim, hidden_size, num_layers, dropout, pad_idx)
    if name == "bilstm":
        return BiLSTM(vocab_size, embed_dim, hidden_size, num_layers, dropout, pad_idx)
    if name == "bilstm_attn":
        return BiLSTMAttention(
            vocab_size,
            embed_dim,
            hidden_size,
            num_layers,
            dropout,
            pad_idx,
            attention_dim=attention_dim,
        )

    raise ValueError(f"Unknown model name: {model_name}")
