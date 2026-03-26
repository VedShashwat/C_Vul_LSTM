import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from src.attention_viz import visualize_attention
from src.models import build_model
from src.preprocess import preprocess_single
from src.utils import ensure_dir, load_checkpoint, load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for C vulnerability detection")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "bilstm", "bilstm_attn", "cnn_bilstm", "bilstm_multihead", "ensemble"],
    )
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


def ensure_function_body(code: str) -> str:
    """Wrap bare snippets in a function body for context."""
    stripped = code.strip()
    if any(kw in stripped for kw in ["void ", "int ", "char ", "static "]):
        if "{" in stripped and "}" in stripped:
            return stripped
    return f"void demo_function(char *input, char *dest, char *src) {{\n{stripped}\n}}"


def _load_vocab(vocab_path: Path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_thresholds(results_dir: Path) -> Dict[str, float]:
    path = results_dir / "optimal_thresholds.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: float(v) for k, v in data.items()}


def _build_model_from_config(model_name: str, config: Dict, vocab: Dict[str, int]):
    return build_model(
        model_name=model_name,
        vocab_size=len(vocab),
        embed_dim=int(config["model"]["embed_dim"]),
        hidden_size=int(config["model"]["hidden_size"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        pad_idx=vocab.get("<PAD>", 0),
        attention_dim=int(config["model"]["attention_dim"]),
        cnn_num_filters=int(config["model"].get("cnn_num_filters", 128)),
        cnn_kernel_sizes=config["model"].get("cnn_kernel_sizes", [3, 5, 7]),
        num_attention_heads=int(config["model"].get("num_attention_heads", 4)),
    )


def _predict_single_model(model, seq: torch.Tensor) -> float:
    with torch.no_grad():
        output = model(seq)
        logits = output[0] if isinstance(output, tuple) else output
        return float(torch.sigmoid(logits).item())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    code_str = ensure_function_body(_load_code(args))

    processed_dir = Path(config["data"]["processed_dir"])
    results_dir = Path(config["results_dir"])
    ensure_dir(str(results_dir))
    thresholds = _load_thresholds(results_dir)

    vocab = _load_vocab(processed_dir / "vocab.json")

    device = resolve_device(config["training"].get("device", "cpu"))

    seq = preprocess_single(code_str, vocab, max_seq_len=int(config["data"]["max_seq_len"])).to(device)

    if args.model == "ensemble":
        ensemble_models = ["lstm", "bilstm", "bilstm_attn", "cnn_bilstm", "bilstm_multihead"]
        probs: List[float] = []
        for model_name in ensemble_models:
            model = _build_model_from_config(model_name, config, vocab)
            checkpoint_path = results_dir / f"{model_name}_best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            load_checkpoint(model=model, checkpoint_path=str(checkpoint_path), map_location=device)
            model = model.to(device)
            model.eval()
            probs.append(_predict_single_model(model, seq))

        prob_vuln = float(sum(probs) / len(probs))
        decision_threshold = thresholds.get("ensemble", 0.5)
    else:
        model = _build_model_from_config(args.model, config, vocab)
        checkpoint_path = results_dir / f"{args.model}_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        load_checkpoint(model=model, checkpoint_path=str(checkpoint_path), map_location=device)
        model = model.to(device)
        model.eval()

        prob_vuln = _predict_single_model(model, seq)
        decision_threshold = thresholds.get(args.model, 0.5)

    if prob_vuln >= decision_threshold:
        print(f"Prediction: VULNERABLE (confidence: {prob_vuln * 100:.1f}%, threshold: {decision_threshold:.2f})")
    else:
        print(f"Prediction: SAFE (confidence: {(1 - prob_vuln) * 100:.1f}%, threshold: {decision_threshold:.2f})")

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
