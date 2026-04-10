import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.attention_viz import visualize_attention
from src.models import build_model
from src.preprocess import extract_vulnerability_features, preprocess_single
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


def _load_ensemble_meta(results_dir: Path) -> Dict:
    path = results_dir / "ensemble_meta.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


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
        vuln_feature_dim=int(config["data"].get("vuln_feature_dim", 32)),
    )


def _predict_single_model(model, seq: torch.Tensor, vuln_features: torch.Tensor | None = None) -> float:
    with torch.no_grad():
        if vuln_features is not None:
            try:
                output = model(seq, vuln_features=vuln_features)
            except TypeError:
                output = model(seq)
        else:
            output = model(seq)
        logits = output[0] if isinstance(output, tuple) else output
        return float(torch.sigmoid(logits).item())


def _sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def should_hard_block_veto(code: str) -> Tuple[bool, str | None]:
    """Never allow veto when clearly dangerous sink patterns are present."""
    never_veto_if_present = ["gets", "strcpy", "sprintf", "scanf", "system", "popen", "alloca"]
    for fn in never_veto_if_present:
        if re.search(r"\b" + re.escape(fn) + r"\s*\(", code):
            return True, fn
    if re.search(r"\bprintf\s*\(\s*[^\s\"'%\n]", code):
        return True, "untrusted_format"
    if re.search(r"\bfprintf\s*\(\s*[^,]+,\s*[^\s\"'%\n]", code):
        return True, "untrusted_format"
    return False, None


def safety_veto(code: str, ensemble_pred: int, ensemble_prob: float) -> tuple[int, str]:
    """
    Deterministic post-processing veto.
    If ensemble says VULNERABLE but code exhibits clear safety patterns,
    override to SAFE. Returns (final_pred, veto_reason).
    """
    if ensemble_pred == 0:
        return 0, "no_veto"

    blocked, blocked_reason = should_hard_block_veto(code)
    if blocked:
        return 1, f"hard_block({blocked_reason})"

    safety_signals: List[str] = []

    # Signal 1: Explicit bounded copy with sizeof.
    if re.search(
        r"\b(snprintf|strncpy|strncat)\s*\([^,]+,\s*sizeof|"
        r"\bmemcpy\s*\([^,]+,[^,]+,\s*sizeof",
        code,
    ):
        safety_signals.append("bounded_copy_sizeof")

    # Signal 2: Null pointer guard before use.
    if re.search(
        r"if\s*\(\s*!\s*\w+\s*\)\s*(return|\{)|"
        r"if\s*\(\s*\w+\s*==\s*NULL\s*\)\s*(return|\{)",
        code,
    ):
        safety_signals.append("null_guard")

    # Signal 3: Explicit size bound checks.
    if re.search(
        r"if\s*\([^)]*\b(len|size|length|count|n)\b[^)]*[<>]=?\s*\d|"
        r"if\s*\(\s*\d+\s*[<>]=?\s*[^)]*\b(len|size|length)\b",
        code,
    ):
        safety_signals.append("explicit_size_check")

    # Signal 4: Path normalization / validation hints.
    if re.search(
        r"\b(realpath|canonicalize_file_name|basename|dirname)\s*\(",
        code,
    ):
        safety_signals.append("path_validated")

    # Signal 5: Path input sanitization against traversal.
    if (
        re.search(r"\b\w+\s*\[\s*\w+\s*\]\s*==\s*'/'", code)
        and re.search(r"==\s*'\.'\s*&&[^\n]*\+\s*1\s*\]\s*==\s*'\.'", code)
    ):
        safety_signals.append("path_input_sanitized")

    # Signal 6: Whitelist / validation gate.
    if re.search(
        r"\b(isalnum|isdigit|isalpha|isprint)\s*\(|"
        r"(whitelist|allowlist|sanitize|validate)\s*\(",
        code,
        re.IGNORECASE,
    ):
        safety_signals.append("input_validated")

    # Signal 7: Safe tempfile construction pattern.
    if re.search(r"O_CREAT.*O_EXCL|O_EXCL.*O_CREAT|mkstemp\s*\(", code):
        safety_signals.append("safe_tempfile")

    # Signal 8: Safe exec dispatch without shell command composition.
    if (
        re.search(r"\b(execv|execve|execl)\s*\(", code)
        and not re.search(r"\b(system|popen)\s*\(", code)
        and not re.search(r"(sprintf|snprintf|strcat|strcpy)[^;]+cmd", code)
    ):
        safety_signals.append("safe_exec_dispatch")

    # Signal 9: Literal format string usage.
    has_literal_format = bool(
        re.search(r"\bprintf\s*\(\s*\"", code)
        or re.search(r"\bfprintf\s*\(\s*[^,]+,\s*\"", code)
        or re.search(r"\bsyslog\s*\(\s*\"", code)
    )
    has_untrusted_format = bool(
        re.search(r"\bprintf\s*\(\s*[^\s\"'%\n]", code)
        or re.search(r"\bfprintf\s*\(\s*[^,]+,\s*[^\s\"'%\n]", code)
    )
    if has_literal_format and not has_untrusted_format:
        safety_signals.append("literal_format_string")

    # Signal 10: Read-only token scan loop (no buffer writes).
    if (
        re.search(r"while\s*\([^)]*!=\s*'\\0'[^)]*\)", code)
        and not re.search(r"\w+\s*\[\s*\w+\s*\]\s*=\s*[^=]", code)
        and re.search(r"return\s*-?1\s*;", code)
    ):
        safety_signals.append("read_only_scan")

    # Signal 11: Bounded character copy loop with explicit null termination.
    if (
        re.search(r"\w+\s*\[\s*\w+\s*\]\s*=\s*\w+\s*\[\s*\w+\s*\]", code)
        and re.search(r"\b\w+\s*<\s*\d+", code)
        and not re.search(r"\b\w+\s*<=\s*\d+", code)
        and re.search(r"\w+\s*\[\s*\w+\s*\]\s*=\s*'\\0'", code)
    ):
        safety_signals.append("strict_bounded_copy_loop")

    # Signal 12: Delimiter-aware bounded token parser pattern.
    if (
        re.search(r"!=\s*'='", code)
        and re.search(r"\b\w+\s*<\s*\d+", code)
        and not re.search(r"\b\w+\s*<=\s*\d+", code)
        and re.search(r"\w+\s*\[\s*\w+\s*\]\s*=\s*\w+\s*\[\s*\w+\s*\]", code)
        and re.search(r"\w+\s*\[\s*\w+\s*\]\s*=\s*'\\0'", code)
    ):
        safety_signals.append("delimiter_bounded_parser")

    # High-confidence override is only allowed for very strong safety signatures.
    strong_combo_override = (
        ("bounded_copy_sizeof" in safety_signals and "path_input_sanitized" in safety_signals)
        or ("literal_format_string" in safety_signals and "strict_bounded_copy_loop" in safety_signals)
        or ("strict_bounded_copy_loop" in safety_signals and "delimiter_bounded_parser" in safety_signals)
        or ("safe_tempfile" in safety_signals)
        or ("safe_exec_dispatch" in safety_signals)
    )

    if ensemble_prob >= 0.85 and not strong_combo_override:
        return 1, "no_veto"

    if len(safety_signals) >= 2:
        return 0, f"safety_veto({','.join(safety_signals)})"

    strong_singles = {"safe_tempfile", "safe_exec_dispatch", "read_only_scan"}
    if len(safety_signals) == 1 and safety_signals[0] in strong_singles:
        return 0, f"safety_veto({safety_signals[0]})"

    return 1, "no_veto"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    original_code_str = _load_code(args)
    code_str = ensure_function_body(original_code_str)

    processed_dir = Path(config["data"]["processed_dir"])
    results_dir = Path(config["results_dir"])
    ensure_dir(str(results_dir))
    thresholds = _load_thresholds(results_dir)
    ensemble_meta = _load_ensemble_meta(results_dir)

    vocab = _load_vocab(processed_dir / "vocab.json")

    device = resolve_device(config["training"].get("device", "cpu"))

    seq = preprocess_single(code_str, vocab, max_seq_len=int(config["data"]["max_seq_len"])).to(device)
    vuln_feats = extract_vulnerability_features(code_str)
    vuln_feat_tensor = torch.tensor(vuln_feats, dtype=torch.float32).unsqueeze(0).to(device)

    if args.model == "ensemble":
        ensemble_models = ["lstm", "bilstm", "bilstm_attn", "cnn_bilstm", "bilstm_multihead"]
        probs_by_name: Dict[str, float] = {}
        for model_name in ensemble_models:
            model = _build_model_from_config(model_name, config, vocab)
            checkpoint_path = results_dir / f"{model_name}_best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            load_checkpoint(model=model, checkpoint_path=str(checkpoint_path), map_location=device)
            model = model.to(device)
            model.eval()
            probs_by_name[model_name] = _predict_single_model(model, seq, vuln_features=vuln_feat_tensor)

        method = str(ensemble_meta.get("method", "avg"))

        if method == "logistic_meta":
            ordered_names = ensemble_meta.get("model_names", ensemble_models)
            coeffs = ensemble_meta.get("coefficients", [])
            intercept = float(ensemble_meta.get("intercept", 0.0))

            if len(ordered_names) == len(coeffs) and all(name in probs_by_name for name in ordered_names):
                score = intercept
                for name, coeff in zip(ordered_names, coeffs):
                    score += float(coeff) * float(probs_by_name[name])
                prob_vuln = float(_sigmoid_scalar(score))
                decision_threshold = float(ensemble_meta.get("threshold", thresholds.get("ensemble", 0.5)))
                ensemble_method = "meta"
            else:
                prob_vuln = float(sum(probs_by_name.values()) / len(probs_by_name))
                decision_threshold = thresholds.get("ensemble", 0.5)
                ensemble_method = "avg-fallback"
        elif method == "avg":
            prob_vuln = float(sum(probs_by_name.values()) / len(probs_by_name))
            decision_threshold = float(ensemble_meta.get("threshold", thresholds.get("ensemble", 0.5)))
            ensemble_method = "avg"
        elif method == "quorum_vote":
            ordered_names = ensemble_meta.get("model_names", ensemble_models)
            per_model_thresholds = ensemble_meta.get("per_model_thresholds", {})
            quorum = int(ensemble_meta.get("quorum", max(1, len(ordered_names) // 2)))

            votes = 0
            for name in ordered_names:
                prob = float(probs_by_name.get(name, 0.0))
                model_threshold = float(per_model_thresholds.get(name, thresholds.get(name, 0.5)))
                if prob >= model_threshold:
                    votes += 1

            prob_vuln = float(votes / max(1, len(ordered_names)))
            decision_threshold = float(quorum / max(1, len(ordered_names)))
            ensemble_method = f"quorum-vote(q={quorum})"
        else:
            prob_vuln = float(sum(probs_by_name.values()) / len(probs_by_name))
            decision_threshold = thresholds.get("ensemble", 0.5)
            ensemble_method = "avg"
    else:
        model = _build_model_from_config(args.model, config, vocab)
        checkpoint_path = results_dir / f"{args.model}_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        load_checkpoint(model=model, checkpoint_path=str(checkpoint_path), map_location=device)
        model = model.to(device)
        model.eval()

        prob_vuln = _predict_single_model(model, seq, vuln_features=vuln_feat_tensor)
        decision_threshold = thresholds.get(args.model, 0.5)

    if args.model == "ensemble":
        ensemble_pred = 1 if prob_vuln >= decision_threshold else 0
        final_pred, veto_reason = safety_veto(original_code_str, ensemble_pred, prob_vuln)
        if veto_reason != "no_veto":
            print(f"  [Safety veto applied: {veto_reason}]")
    else:
        final_pred = 1 if prob_vuln >= decision_threshold else 0

    if final_pred == 1:
        print(f"Prediction: VULNERABLE (confidence: {prob_vuln * 100:.1f}%, threshold: {decision_threshold:.2f})")
    else:
        print(f"Prediction: SAFE (confidence: {(1 - prob_vuln) * 100:.1f}%, threshold: {decision_threshold:.2f})")

    if args.model == "ensemble":
        print(f"Ensemble method: {ensemble_method}")

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
