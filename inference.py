import argparse
import json
from pathlib import Path

import torch

from src.attention_viz import visualize_attention
from src.models import build_model
from src.preprocess import preprocess_single
from src.utils import ensure_dir, load_checkpoint, load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for C vulnerability detection")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "bilstm", "bilstm_attn"])
    parser.add_argument("--file", type=str, default=None, help="Path to C file snippet")
    parser.add_argument("--code", type=str, default=None, help="Raw C code string")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    return parser.parse_args()


def _load_code(args: argparse.Namespace) -> str:
    if args.file is None and args.code is None:
        raise ValueError("Provide either --file or --code for inference.")
    if args.file is not None and args.code is not None:
        raise ValueError("Use only one of --file or --code.")

    if args.file is not None:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    return args.code


def _load_vocab(vocab_path: Path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    code_str = _load_code(args)

    processed_dir = Path(config["data"]["processed_dir"])
    results_dir = Path(config["results_dir"])
    ensure_dir(str(results_dir))

    vocab = _load_vocab(processed_dir / "vocab.json")
    pad_idx = vocab.get("<PAD>", 0)

    model = build_model(
        model_name=args.model,
        vocab_size=len(vocab),
        embed_dim=int(config["model"]["embed_dim"]),
        hidden_size=int(config["model"]["hidden_size"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        pad_idx=pad_idx,
        attention_dim=int(config["model"]["attention_dim"]),
    )

    device = resolve_device(config["training"].get("device", "cpu"))

    checkpoint_path = results_dir / f"{args.model}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_checkpoint(model=model, checkpoint_path=str(checkpoint_path), map_location=device)
    model = model.to(device)
    model.eval()

    seq = preprocess_single(code_str, vocab, max_seq_len=int(config["data"]["max_seq_len"])).to(device)

    with torch.no_grad():
        output = model(seq)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        prob_vuln = torch.sigmoid(logits).item()

    if prob_vuln >= 0.5:
        print(f"Prediction: VULNERABLE (confidence: {prob_vuln * 100:.1f}%)")
    else:
        print(f"Prediction: SAFE (confidence: {(1 - prob_vuln) * 100:.1f}%)")

    if args.model == "bilstm_attn":
        vocab_inv = {idx: tok for tok, idx in vocab.items()}
        attention_path = results_dir / "inference_attention.png"
        visualize_attention(
            model=model,
            vocab_inv=vocab_inv,
            code_snippet=code_str,
            config=config,
            save_path=str(attention_path),
        )
        print(f"Attention visualization saved to: {attention_path}")


if __name__ == "__main__":
    main()
