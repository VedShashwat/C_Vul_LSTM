from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _lstm_dropout(num_layers: int, dropout: float) -> float:
    return dropout if num_layers > 1 else 0.0


def _concat_vuln_features(
    seq_repr: torch.Tensor,
    vuln_features: Optional[torch.Tensor],
    feature_dim: int,
) -> torch.Tensor:
    if vuln_features is None:
        feat = torch.zeros(seq_repr.size(0), feature_dim, device=seq_repr.device, dtype=seq_repr.dtype)
    else:
        feat = vuln_features.to(seq_repr.device, dtype=seq_repr.dtype)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        if feat.size(1) != feature_dim:
            if feat.size(1) > feature_dim:
                feat = feat[:, :feature_dim]
            else:
                pad = torch.zeros(feat.size(0), feature_dim - feat.size(1), device=feat.device, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=1)
    return torch.cat([seq_repr, feat], dim=1)


class VanillaLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
        vuln_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        self.vuln_feature_dim = vuln_feature_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=_lstm_dropout(num_layers, dropout),
            bidirectional=False,
        )
        self.seq_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.fc = nn.Linear((hidden_size * 2) + vuln_feature_dim, 1)

    def forward(self, x: torch.Tensor, vuln_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        seq_repr = h_n[-1]
        seq_repr = self.seq_proj(seq_repr)
        fused = _concat_vuln_features(seq_repr, vuln_features, self.vuln_feature_dim)
        logits = self.fc(fused).squeeze(-1)
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
        vuln_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        self.vuln_feature_dim = vuln_feature_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=_lstm_dropout(num_layers, dropout),
            bidirectional=True,
        )
        self.fc = nn.Linear((hidden_size * 2) + vuln_feature_dim, 1)

    def forward(self, x: torch.Tensor, vuln_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)

        forward_final = h_n[-2]
        backward_final = h_n[-1]
        seq_repr = torch.cat([forward_final, backward_final], dim=1)
        fused = _concat_vuln_features(seq_repr, vuln_features, self.vuln_feature_dim)

        logits = self.fc(fused).squeeze(-1)
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
        vuln_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.vuln_feature_dim = vuln_feature_dim

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
        self.fc = nn.Linear((hidden_size * 2) + vuln_feature_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        vuln_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        fused = _concat_vuln_features(context, vuln_features, self.vuln_feature_dim)
        logits = self.fc(fused).squeeze(-1)

        return logits, alpha


class CNN_BiLSTM(nn.Module):
    """
    CNN extracts local n-gram vulnerability patterns and BiLSTM captures long-range context.
    The final classifier uses fused CNN and BiLSTM features.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
        cnn_num_filters: int = 128,
        cnn_kernel_sizes: Sequence[int] = (3, 5, 7),
        vuln_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        self.vuln_feature_dim = vuln_feature_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(0.2)

        self.conv_towers = nn.ModuleList(
            [nn.Conv1d(embed_dim, cnn_num_filters, kernel_size=k) for k in cnn_kernel_sizes]
        )
        self.cnn_pool = nn.AdaptiveMaxPool1d(1)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=_lstm_dropout(num_layers, dropout),
            bidirectional=True,
        )

        cnn_out_dim = cnn_num_filters * len(cnn_kernel_sizes)
        fusion_in_dim = cnn_out_dim + (hidden_size * 2)

        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_fc = nn.Linear(fusion_in_dim, hidden_size * 2)
        self.fc = nn.Linear((hidden_size * 2) + vuln_feature_dim, 1)

    def forward(self, x: torch.Tensor, vuln_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embed_dropout(self.embedding(x))

        conv_in = emb.transpose(1, 2)
        cnn_features = []
        for conv in self.conv_towers:
            feat = torch.relu(conv(conv_in))
            feat = self.cnn_pool(feat).squeeze(-1)
            cnn_features.append(feat)
        cnn_out = torch.cat(cnn_features, dim=1)

        _, (h_n, _) = self.lstm(emb)
        lstm_out = torch.cat([h_n[-2], h_n[-1]], dim=1)

        fused = torch.cat([cnn_out, lstm_out], dim=1)
        fused = self.fusion_dropout(fused)
        fused = torch.relu(self.fusion_fc(fused))
        fused = _concat_vuln_features(fused, vuln_features, self.vuln_feature_dim)
        logits = self.fc(fused).squeeze(-1)
        return logits


class BiLSTM_MultiHeadAttention(nn.Module):
    """
    BiLSTM encoder followed by multi-head self-attention over token states.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
        num_attention_heads: int = 4,
        vuln_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.vuln_feature_dim = vuln_feature_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=_lstm_dropout(num_layers, dropout),
            bidirectional=True,
        )

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.act = nn.GELU()
        self.fc = nn.Linear((hidden_size * 2) + vuln_feature_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        vuln_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        outputs, _ = self.lstm(emb)

        pad_mask = x.eq(self.pad_idx)
        attn_pad_mask = pad_mask.clone()
        all_pad_rows = attn_pad_mask.all(dim=1)
        if all_pad_rows.any():
            attn_pad_mask[all_pad_rows, 0] = False

        attn_out, attn_weights = self.mha(
            outputs,
            outputs,
            outputs,
            key_padding_mask=attn_pad_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        attn_out = torch.nan_to_num(attn_out)
        attn_weights = torch.nan_to_num(attn_weights)

        valid_mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (attn_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)

        pooled = self.norm(pooled)
        hidden = self.act(self.proj(self.dropout(pooled)))
        hidden = _concat_vuln_features(hidden, vuln_features, self.vuln_feature_dim)
        logits = self.fc(hidden).squeeze(-1)
        return logits, attn_weights


def build_model(
    model_name: str,
    vocab_size: int,
    embed_dim: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    pad_idx: int,
    attention_dim: int = 64,
    cnn_num_filters: int = 128,
    cnn_kernel_sizes: Sequence[int] = (3, 5, 7),
    num_attention_heads: int = 4,
    vuln_feature_dim: int = 32,
) -> nn.Module:
    name = model_name.lower()

    if name == "lstm":
        return VanillaLSTM(vocab_size, embed_dim, hidden_size, num_layers, dropout, pad_idx, vuln_feature_dim)
    if name == "bilstm":
        return BiLSTM(vocab_size, embed_dim, hidden_size, num_layers, dropout, pad_idx, vuln_feature_dim)
    if name == "bilstm_attn":
        return BiLSTMAttention(
            vocab_size,
            embed_dim,
            hidden_size,
            num_layers,
            dropout,
            pad_idx,
            attention_dim=attention_dim,
            vuln_feature_dim=vuln_feature_dim,
        )
    if name == "cnn_bilstm":
        return CNN_BiLSTM(
            vocab_size,
            embed_dim,
            hidden_size,
            num_layers,
            dropout,
            pad_idx,
            cnn_num_filters=cnn_num_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            vuln_feature_dim=vuln_feature_dim,
        )
    if name == "bilstm_multihead":
        return BiLSTM_MultiHeadAttention(
            vocab_size,
            embed_dim,
            hidden_size,
            num_layers,
            dropout,
            pad_idx,
            num_attention_heads=num_attention_heads,
            vuln_feature_dim=vuln_feature_dim,
        )

    raise ValueError(f"Unknown model name: {model_name}")
