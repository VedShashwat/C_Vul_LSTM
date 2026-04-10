import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from inference import (
    _build_model_from_config,
    _load_ensemble_meta,
    _load_thresholds,
    _load_vocab,
    _predict_single_model,
    _sigmoid_scalar,
    ensure_function_body,
    safety_veto,
)
from src.preprocess import extract_vulnerability_features, preprocess_single
from src.utils import load_checkpoint, load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run snippet benchmark suite")
    parser.add_argument("--suite", type=str, required=True, help="Suite folder (e.g., data/snippets_v2)")
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["lstm", "bilstm", "bilstm_attn", "cnn_bilstm", "bilstm_multihead", "ensemble"],
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--save-json", action="store_true", help="Persist detailed benchmark JSON in results/")
    return parser.parse_args()


def ensemble_predict(
    code: str,
    config: Dict,
    models: Dict[str, torch.nn.Module],
    vocab: Dict[str, int],
    thresholds: Dict[str, float],
    ensemble_meta: Dict,
    device: torch.device,
) -> Dict:
    code_str = ensure_function_body(code)
    seq = preprocess_single(code_str, vocab, max_seq_len=int(config["data"]["max_seq_len"])).to(device)
    vuln_feats = extract_vulnerability_features(code_str)
    vuln_feat_tensor = torch.tensor(vuln_feats, dtype=torch.float32).unsqueeze(0).to(device)

    ordered_default = ["lstm", "bilstm", "bilstm_attn", "cnn_bilstm", "bilstm_multihead"]
    probs_by_name: Dict[str, float] = {}
    for model_name, model in models.items():
        probs_by_name[model_name] = _predict_single_model(model, seq, vuln_features=vuln_feat_tensor)

    method = str(ensemble_meta.get("method", "avg"))

    if method == "logistic_meta":
        ordered_names = ensemble_meta.get("model_names", ordered_default)
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
            decision_threshold = float(thresholds.get("ensemble", 0.5))
            ensemble_method = "avg-fallback"
    elif method == "avg":
        prob_vuln = float(sum(probs_by_name.values()) / len(probs_by_name))
        decision_threshold = float(ensemble_meta.get("threshold", thresholds.get("ensemble", 0.5)))
        ensemble_method = "avg"
    elif method == "quorum_vote":
        ordered_names = ensemble_meta.get("model_names", ordered_default)
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
        decision_threshold = float(thresholds.get("ensemble", 0.5))
        ensemble_method = "avg"

    ensemble_pred = 1 if prob_vuln >= decision_threshold else 0
    final_pred, veto_reason = safety_veto(code, ensemble_pred, prob_vuln)
    prediction = "VULNERABLE" if final_pred == 1 else "SAFE"
    return {
        "prediction": prediction,
        "prob_vuln": prob_vuln,
        "threshold": decision_threshold,
        "method": ensemble_method,
        "veto_reason": veto_reason,
    }


def single_predict(
    code: str,
    model_name: str,
    model: torch.nn.Module,
    config: Dict,
    vocab: Dict[str, int],
    thresholds: Dict[str, float],
    device: torch.device,
) -> Dict:
    code_str = ensure_function_body(code)
    seq = preprocess_single(code_str, vocab, max_seq_len=int(config["data"]["max_seq_len"])).to(device)
    vuln_feats = extract_vulnerability_features(code_str)
    vuln_feat_tensor = torch.tensor(vuln_feats, dtype=torch.float32).unsqueeze(0).to(device)

    prob_vuln = _predict_single_model(model, seq, vuln_features=vuln_feat_tensor)
    threshold = float(thresholds.get(model_name, 0.5))
    prediction = "VULNERABLE" if prob_vuln >= threshold else "SAFE"
    return {
        "prediction": prediction,
        "prob_vuln": prob_vuln,
        "threshold": threshold,
        "method": model_name,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    suite_dir = Path(args.suite)
    if not suite_dir.exists() or not suite_dir.is_dir():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

    results_dir = Path(config["results_dir"])
    processed_dir = Path(config["data"]["processed_dir"])

    vocab = _load_vocab(processed_dir / "vocab.json")
    thresholds = _load_thresholds(results_dir)
    ensemble_meta = _load_ensemble_meta(results_dir)
    device = resolve_device(config["training"].get("device", "cpu"))

    requested_models = [args.model]
    if args.model == "ensemble":
        requested_models = ["lstm", "bilstm", "bilstm_attn", "cnn_bilstm", "bilstm_multihead"]

    models: Dict[str, torch.nn.Module] = {}
    for model_name in requested_models:
        model = _build_model_from_config(model_name, config, vocab)
        checkpoint_path = results_dir / f"{model_name}_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        load_checkpoint(model=model, checkpoint_path=str(checkpoint_path), map_location=device)
        model = model.to(device)
        model.eval()
        models[model_name] = model

    files = sorted(list(suite_dir.glob("*.c")) + list(suite_dir.glob("*.txt")))
    if not files:
        raise RuntimeError(f"No benchmark files found under {suite_dir}")

    tp = tn = fp = fn = 0
    details: List[Dict] = []

    for path in files:
        expected = "VULNERABLE" if path.name.startswith("vuln_") else "SAFE"
        code = path.read_text(encoding="utf-8")

        if args.model == "ensemble":
            out = ensemble_predict(code, config, models, vocab, thresholds, ensemble_meta, device)
        else:
            out = single_predict(code, args.model, models[args.model], config, vocab, thresholds, device)

        pred = out["prediction"]
        if expected == "VULNERABLE" and pred == "VULNERABLE":
            tp += 1
        elif expected == "SAFE" and pred == "SAFE":
            tn += 1
        elif expected == "SAFE" and pred == "VULNERABLE":
            fp += 1
        else:
            fn += 1

        details.append(
            {
                "file": path.name,
                "expected": expected,
                "predicted": pred,
                "prob_vuln": round(float(out["prob_vuln"]), 6),
                "threshold": round(float(out["threshold"]), 6),
                "method": out["method"],
                "veto_reason": out.get("veto_reason", "no_veto"),
                "correct": expected == pred,
            }
        )

    total = tp + tn + fp + fn
    score = tp + tn
    accuracy = score / total if total else 0.0
    vuln_recall = tp / (tp + fn) if (tp + fn) else 0.0
    safe_recall = tn / (tn + fp) if (tn + fp) else 0.0

    print(f"SUITE: {suite_dir}")
    print(f"MODEL: {args.model}")
    print(f"SCORE: {score}/{total}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"ACCURACY={accuracy:.4f} VULN_RECALL={vuln_recall:.4f} SAFE_RECALL={safe_recall:.4f}")

    failed = [d for d in details if not d["correct"]]
    if failed:
        print("FAILURES:")
        for row in failed:
            print(
                f"  {row['file']}: expected={row['expected']} predicted={row['predicted']} "
                f"prob={row['prob_vuln']} thr={row['threshold']}"
            )

    if args.save_json:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "suite": str(suite_dir),
            "model": args.model,
            "score": score,
            "total": total,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "vuln_recall": vuln_recall,
            "safe_recall": safe_recall,
            "details": details,
        }
        out = results_dir / f"benchmark_{suite_dir.name}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"SAVED_JSON: {out}")


if __name__ == "__main__":
    main()
