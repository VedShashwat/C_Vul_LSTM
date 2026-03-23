from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.preprocess import preprocess_single
from src.utils import ensure_dir, resolve_device

DANGEROUS_FUNCS = {"strcpy", "gets", "sprintf", "malloc", "memcpy"}


def _build_vocab_from_inverse(vocab_inv: Dict) -> Dict[str, int]:
    if not vocab_inv:
        raise ValueError("vocab_inv is empty.")

    first_key = next(iter(vocab_inv.keys()))
    if isinstance(first_key, int) or (isinstance(first_key, str) and first_key.isdigit()):
        return {token: int(idx) for idx, token in vocab_inv.items()}

    return {token: idx for idx, token in vocab_inv.items()}


def _token_list_from_sequence(seq: torch.Tensor, vocab_inv: Dict[int, str], pad_idx: int) -> List[str]:
    ids = seq.squeeze(0).tolist()
    tokens = []
    for idx in ids:
        if idx == pad_idx:
            continue
        token = vocab_inv.get(idx, "<UNK>")
        tokens.append(token)
    return tokens


def visualize_attention(model, vocab_inv: Dict, code_snippet: str, config: Dict, save_path: str) -> None:
    vocab = _build_vocab_from_inverse(vocab_inv)
    max_seq_len = int(config["data"]["max_seq_len"])
    device = resolve_device(config["training"].get("device", "cpu"))
    pad_idx = vocab.get("<PAD>", 0)

    seq = preprocess_single(code_snippet, vocab, max_seq_len=max_seq_len).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(seq)
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("visualize_attention requires a model that returns (logits, attention).")
        _, alpha = output

    alpha = alpha.squeeze(0).detach().cpu().numpy()

    inv_as_int = {}
    for k, v in vocab_inv.items():
        if isinstance(k, int):
            inv_as_int[k] = v
        elif isinstance(k, str) and k.isdigit():
            inv_as_int[int(k)] = v

    tokens = _token_list_from_sequence(seq.cpu(), inv_as_int, pad_idx=pad_idx)
    if not tokens:
        raise ValueError("No non-PAD tokens found for attention visualization.")

    n_tokens = min(50, len(tokens))
    tokens = tokens[:n_tokens]
    attn_vals = alpha[:n_tokens]

    attn_sum = float(np.sum(attn_vals))
    if attn_sum > 0:
        attn_vals = attn_vals / attn_sum

    ensure_dir(str(Path(save_path).parent))

    fig_height = max(5, int(0.35 * n_tokens))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    cmap = plt.get_cmap("YlOrRd")
    colors = cmap(attn_vals / (attn_vals.max() + 1e-8))

    y_pos = np.arange(n_tokens)
    bars = ax.barh(y_pos, attn_vals, color=colors, edgecolor="black", linewidth=0.5)

    for idx, tok in enumerate(tokens):
        if tok in DANGEROUS_FUNCS:
            bars[idx].set_edgecolor("red")
            bars[idx].set_linewidth(2.0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Attention Weight")
    ax.set_title("Token Attention Visualization (Top 50 Non-PAD Tokens)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()
