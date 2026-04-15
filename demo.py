import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from inference import (
    _build_model_from_config,
    _load_ensemble_meta,
    _load_thresholds,
    _load_vocab,
)
from run_benchmark import ensemble_predict, single_predict
from src.utils import ensure_dir, load_checkpoint, load_config, resolve_device


BASE_MODELS = ["lstm", "bilstm", "bilstm_attn", "cnn_bilstm", "bilstm_multihead"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="demo runner: benchmark all models across curated snippet suites"
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["data/snippets_v2", "data/snippets_v3", "data/snippets_v4_unseen"],
        help="Suite folders to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=BASE_MODELS + ["ensemble"],
        choices=BASE_MODELS + ["ensemble"],
        help="Models to include in the demo report",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--save-json", action="store_true", help="Save full report JSON to results/")
    parser.add_argument(
        "--show-ensemble-failures",
        action="store_true",
        help="Print per-file mistakes for ensemble on each suite",
    )
    return parser.parse_args()


def _load_models(
    requested_models: List[str],
    config: Dict,
    vocab: Dict[str, int],
    results_dir: Path,
    device: torch.device,
) -> Dict[str, torch.nn.Module]:
    needed = set(requested_models)
    if "ensemble" in needed:
        needed.remove("ensemble")
        needed.update(BASE_MODELS)

    loaded: Dict[str, torch.nn.Module] = {}
    for model_name in sorted(needed):
        model = _build_model_from_config(model_name, config, vocab)
        checkpoint_path = results_dir / f"{model_name}_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        load_checkpoint(model=model, checkpoint_path=str(checkpoint_path), map_location=device)
        model = model.to(device)
        model.eval()
        loaded[model_name] = model
    return loaded


def _suite_files(suite: Path) -> List[Path]:
    files = sorted(list(suite.glob("*.c")) + list(suite.glob("*.txt")))
    if not files:
        raise RuntimeError(f"No benchmark files found under {suite}")
    return files


def _evaluate_model_on_suite(
    model_name: str,
    suite: Path,
    config: Dict,
    loaded_models: Dict[str, torch.nn.Module],
    vocab: Dict[str, int],
    thresholds: Dict[str, float],
    ensemble_meta: Dict,
    device: torch.device,
) -> Dict:
    tp = tn = fp = fn = 0
    details: List[Dict] = []

    for path in _suite_files(suite):
        expected = "VULNERABLE" if path.name.startswith("vuln_") else "SAFE"
        code = path.read_text(encoding="utf-8")

        if model_name == "ensemble":
            out = ensemble_predict(code, config, loaded_models, vocab, thresholds, ensemble_meta, device)
        else:
            out = single_predict(
                code,
                model_name,
                loaded_models[model_name],
                config,
                vocab,
                thresholds,
                device,
            )

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
                "correct": expected == pred,
                "prob_vuln": float(out["prob_vuln"]),
                "threshold": float(out["threshold"]),
                "method": out["method"],
                "veto_reason": out.get("veto_reason", "no_veto"),
            }
        )

    total = tp + tn + fp + fn
    score = tp + tn
    accuracy = score / total if total else 0.0
    return {
        "suite": suite.name,
        "model": model_name,
        "score": score,
        "total": total,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "details": details,
    }


def _print_suite_table(suite_name: str, rows: List[Dict]) -> None:
    print(f"\n=== {suite_name} ===")
    print("model              score   acc     tp  tn  fp  fn")
    for row in rows:
        print(
            f"{row['model']:<18} {row['score']:>2}/{row['total']:<4} {row['accuracy']:.4f} "
            f"  {row['tp']:>2}  {row['tn']:>2}  {row['fp']:>2}  {row['fn']:>2}"
        )


def _print_standard_test_results(results_dir: Path, model_order: List[str]) -> None:
    comparison_path = results_dir / "comparison_table.md"
    print("\n=== STANDARD TEST-SET RESULTS (from comparison_table.md) ===")
    if not comparison_path.exists():
        print(f"comparison table not found: {comparison_path}")
        return

    lines = [line.rstrip() for line in comparison_path.read_text(encoding="utf-8").splitlines()]
    table_lines = [line for line in lines if line.strip().startswith("|")]
    if len(table_lines) < 2:
        print(f"comparison table format is invalid: {comparison_path}")
        return

    model_set = set(model_order)
    body_lines: List[str] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] in model_set:
            body_lines.append(line)

    print(table_lines[0])
    print(table_lines[1])
    if body_lines:
        for line in body_lines:
            print(line)
    else:
        for line in table_lines[2:]:
            print(line)


def _print_overall_table(all_rows: List[Dict], model_order: List[str]) -> None:
    agg: Dict[str, Dict[str, int]] = {
        name: {"score": 0, "total": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0} for name in model_order
    }

    for row in all_rows:
        model_name = row["model"]
        agg[model_name]["score"] += int(row["score"])
        agg[model_name]["total"] += int(row["total"])
        agg[model_name]["tp"] += int(row["tp"])
        agg[model_name]["tn"] += int(row["tn"])
        agg[model_name]["fp"] += int(row["fp"])
        agg[model_name]["fn"] += int(row["fn"])

    print("\n=== OVERALL (across selected suites) ===")
    print("model              score   acc     tp  tn  fp  fn")
    for model_name in model_order:
        row = agg[model_name]
        total = row["total"]
        acc = (row["score"] / total) if total else 0.0
        print(
            f"{model_name:<18} {row['score']:>2}/{total:<4} {acc:.4f} "
            f"  {row['tp']:>2}  {row['tn']:>2}  {row['fp']:>2}  {row['fn']:>2}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    suites = [Path(s) for s in args.suites]
    for suite in suites:
        if not suite.exists() or not suite.is_dir():
            raise FileNotFoundError(f"Suite directory not found: {suite}")

    results_dir = Path(config["results_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    ensure_dir(str(results_dir))

    vocab = _load_vocab(processed_dir / "vocab.json")
    thresholds = _load_thresholds(results_dir)
    ensemble_meta = _load_ensemble_meta(results_dir)
    device = resolve_device(config["training"].get("device", "cpu"))

    _print_standard_test_results(results_dir, args.models)

    loaded_models = _load_models(args.models, config, vocab, results_dir, device)

    all_rows: List[Dict] = []
    for suite in suites:
        suite_rows: List[Dict] = []
        for model_name in args.models:
            row = _evaluate_model_on_suite(
                model_name=model_name,
                suite=suite,
                config=config,
                loaded_models=loaded_models,
                vocab=vocab,
                thresholds=thresholds,
                ensemble_meta=ensemble_meta,
                device=device,
            )
            suite_rows.append(row)
            all_rows.append(row)

        _print_suite_table(suite.name, suite_rows)

        if args.show_ensemble_failures:
            ensemble_row = next((r for r in suite_rows if r["model"] == "ensemble"), None)
            if ensemble_row is not None:
                failures = [d for d in ensemble_row["details"] if not d["correct"]]
                if failures:
                    print("  Ensemble failures:")
                    for f in failures:
                        print(
                            "   - "
                            f"{f['file']}: expected={f['expected']} predicted={f['predicted']} "
                            f"prob={f['prob_vuln']:.4f} reason={f['veto_reason']}"
                        )
                else:
                    print("  Ensemble failures: none")

    _print_overall_table(all_rows, args.models)

    if args.save_json:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "suites": [str(s) for s in suites],
            "models": args.models,
            "rows": all_rows,
        }
        out = results_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSAVED_JSON: {out}")


if __name__ == "__main__":
    main()
