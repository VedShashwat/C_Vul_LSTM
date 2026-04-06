
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.dataset import get_dataloaders
from src.evaluate import (
    evaluate_ensemble,
    evaluate_model,
    generate_comparison_table,
    load_or_create_test_cwe_labels,
    per_cwe_analysis,
    plot_training_curves,
)
from src.models import build_model
from src.preprocess import preprocess_and_save
from src.train import train_model
from src.utils import load_checkpoint, load_config, resolve_device, seed_everything


BASELINE_MODELS = ["lstm", "bilstm", "bilstm_attn"]
NEW_MODELS = ["cnn_bilstm", "bilstm_multihead"]
ALL_MODELS = BASELINE_MODELS + NEW_MODELS
DEFAULT_MODELS = ALL_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM-based vulnerability detectors")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing stage")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=ALL_MODELS,
        help="Models to train/evaluate",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Load existing checkpoints for baseline models and train only new models.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load all existing checkpoints and run evaluation only with threshold tuning.",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    return parser.parse_args()


def _processed_artifacts_exist(processed_dir: Path) -> bool:
    required = [
        processed_dir / "train_data.pt",
        processed_dir / "val_data.pt",
        processed_dir / "test_data.pt",
        processed_dir / "vocab.json",
    ]
    return all(path.exists() for path in required)


def _build_model_from_config(model_name: str, config: Dict, vocab_size: int, pad_idx: int):
    return build_model(
        model_name=model_name,
        vocab_size=vocab_size,
        embed_dim=int(config["model"]["embed_dim"]),
        hidden_size=int(config["model"]["hidden_size"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        pad_idx=pad_idx,
        attention_dim=int(config["model"]["attention_dim"]),
        cnn_num_filters=int(config["model"].get("cnn_num_filters", 128)),
        cnn_kernel_sizes=config["model"].get("cnn_kernel_sizes", [3, 5, 7]),
        num_attention_heads=int(config["model"].get("num_attention_heads", 4)),
    )


def _ensure_baseline_checkpoints(results_dir: Path) -> None:
    missing = [name for name in BASELINE_MODELS if not (results_dir / f"{name}_best.pt").exists()]
    if missing:
        raise FileNotFoundError(f"Missing baseline checkpoints: {missing}")


def _ensure_checkpoints(results_dir: Path, model_names: List[str]) -> None:
    missing = [name for name in model_names if not (results_dir / f"{name}_best.pt").exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")


def _get_threshold_policy(config: Dict) -> Dict:
    evaluation_cfg = config.get("evaluation", {}) or {}
    return {
        "objective": str(evaluation_cfg.get("threshold_objective", "f1")),
        "beta": float(evaluation_cfg.get("threshold_beta", 2.0)),
        "min_recall": evaluation_cfg.get("threshold_min_recall", None),
        "threshold_start": float(evaluation_cfg.get("threshold_start", 0.10)),
        "threshold_end": float(evaluation_cfg.get("threshold_end", 0.90)),
        "threshold_step": float(evaluation_cfg.get("threshold_step", 0.02)),
        "ensemble_strategy": str(evaluation_cfg.get("ensemble_strategy", "auto")),
        "ensemble_quorum": evaluation_cfg.get("ensemble_quorum", None),
    }


def run_pipeline(
    config_path: str,
    skip_preprocess: bool,
    model_names: List[str],
    only_new: bool = False,
    eval_only: bool = False,
) -> str:
    config = load_config(config_path)
    seed_everything(int(config["training"]["seed"]))
    threshold_policy = _get_threshold_policy(config)

    processed_dir = Path(config["data"]["processed_dir"])
    results_dir = Path(config["results_dir"])
    has_processed = _processed_artifacts_exist(processed_dir)

    if not has_processed and skip_preprocess:
        raise FileNotFoundError("Processed data not found, but --skip-preprocess was set.")

    if not has_processed:
        print("Processed data not found. Running preprocessing...")
        preprocess_and_save(config_path=config_path)
    else:
        print("Processed data found. Skipping preprocessing.")

    train_loader, val_loader, test_loader = get_dataloaders(config)

    with open(processed_dir / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    pad_idx = vocab.get("<PAD>", 0)

    histories: Dict[str, Dict[str, List[float]]] = {}
    eval_results: Dict[str, Dict[str, float]] = {}
    loaded_models: Dict[str, torch.nn.Module] = {}
    selected_models = list(model_names)

    if eval_only:
        eval_models = [name for name in ALL_MODELS]
        _ensure_checkpoints(results_dir, eval_models)
        for model_name in eval_models:
            print(f"\n===== Evaluating checkpoint {model_name} =====")
            model = _build_model_from_config(model_name, config, vocab_size, pad_idx)
            load_checkpoint(
                model=model,
                checkpoint_path=str(results_dir / f"{model_name}_best.pt"),
                map_location=resolve_device(config["training"].get("device", "cpu")),
            )
            metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                model_name=model_name,
                results_dir=config["results_dir"],
                device_pref=config["training"].get("device", "cpu"),
                threshold=None,
                val_loader=val_loader,
                threshold_policy=threshold_policy,
            )
            eval_results[model_name] = metrics
            loaded_models[model_name] = model

    elif only_new:
        _ensure_baseline_checkpoints(results_dir)
        for model_name in BASELINE_MODELS:
            print(f"\n===== Loading baseline {model_name} =====")
            model = _build_model_from_config(model_name, config, vocab_size, pad_idx)
            load_checkpoint(
                model=model,
                checkpoint_path=str(results_dir / f"{model_name}_best.pt"),
                map_location=resolve_device(config["training"].get("device", "cpu")),
            )
            metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                model_name=model_name,
                results_dir=config["results_dir"],
                device_pref=config["training"].get("device", "cpu"),
                threshold=None,
                val_loader=val_loader,
                threshold_policy=threshold_policy,
            )
            eval_results[model_name] = metrics
            loaded_models[model_name] = model

    else:
        for model_name in [m for m in BASELINE_MODELS if m in selected_models]:
            print(f"\n===== Running {model_name} =====")
            model = _build_model_from_config(model_name, config, vocab_size, pad_idx)

            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                model_name=model_name,
                use_focal_loss=False,
                use_warmup=False,
            )
            histories[model_name] = history

            metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                model_name=model_name,
                results_dir=config["results_dir"],
                device_pref=config["training"].get("device", "cpu"),
                threshold=None,
                val_loader=val_loader,
                threshold_policy=threshold_policy,
            )
            eval_results[model_name] = metrics
            loaded_models[model_name] = model

    if eval_only:
        new_eval_models = []
    else:
        new_eval_models = [m for m in NEW_MODELS if m in selected_models]

    for model_name in new_eval_models:
        print(f"\n===== Running {model_name} =====")
        focal_flag = bool(config["training"].get("use_focal_loss", True))
        warmup_flag = bool(config["training"].get("use_warmup_cosine", True))

        attempted_batch_sizes = []
        for candidate_bs in [int(config["training"]["batch_size"]), 32, 16]:
            if candidate_bs in attempted_batch_sizes:
                continue
            attempted_batch_sizes.append(candidate_bs)

            config["training"]["batch_size"] = candidate_bs
            train_loader, val_loader, test_loader = get_dataloaders(config)

            model = _build_model_from_config(model_name, config, vocab_size, pad_idx)
            try:
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    model_name=model_name,
                    use_focal_loss=focal_flag,
                    use_warmup=warmup_flag,
                )
                histories[model_name] = history

                metrics = evaluate_model(
                    model=model,
                    test_loader=test_loader,
                    model_name=model_name,
                    results_dir=config["results_dir"],
                    device_pref=config["training"].get("device", "cpu"),
                    threshold=None,
                    val_loader=val_loader,
                    threshold_policy=threshold_policy,
                )
                eval_results[model_name] = metrics
                loaded_models[model_name] = model
                break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                    print(f"CUDA OOM for {model_name} at batch_size={candidate_bs}. Retrying with smaller batch.")
                    torch.cuda.empty_cache()
                    continue
                raise
        else:
            raise RuntimeError(f"Failed to train {model_name} after OOM retries.")

    ensemble_model_names = [name for name in ALL_MODELS if name in loaded_models]
    ensemble_models = [loaded_models[name] for name in ensemble_model_names]
    ensemble_results = evaluate_ensemble(
        model_list=ensemble_models,
        model_names=ensemble_model_names,
        test_loader=test_loader,
        val_loader=val_loader,
        results_dir=config["results_dir"],
        device=config["training"].get("device", "cpu"),
        threshold_policy=threshold_policy,
    )

    eval_results["ensemble"] = ensemble_results["ensemble"]

    single_model_names = [name for name in ALL_MODELS if name in eval_results]
    best_model_name = max(single_model_names, key=lambda n: eval_results[n]["f1"])
    expected_test_size = int(len(test_loader.dataset))  # type: ignore[arg-type]
    test_cwe_labels = load_or_create_test_cwe_labels(config, expected_size=expected_test_size)
    per_cwe_analysis(
        best_model=loaded_models[best_model_name],
        test_loader=test_loader,
        test_cwe_labels=test_cwe_labels,
        results_dir=config["results_dir"],
    )

    history_names = [name for name in ALL_MODELS if name in histories]
    if history_names:
        plot_training_curves([histories[name] for name in history_names], history_names, config["results_dir"])

    ordered_results: Dict[str, Dict[str, float]] = {}
    for name in ALL_MODELS:
        if name in eval_results:
            ordered_results[name] = eval_results[name]
    ordered_results["ensemble"] = eval_results["ensemble"]

    thresholds_out = {name: float(metrics.get("threshold", 0.5)) for name, metrics in ordered_results.items()}
    with open(results_dir / "optimal_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds_out, f, ensure_ascii=True, indent=2)

    table = generate_comparison_table(ordered_results, config["results_dir"])
    print(f"Best-2 ensemble models: {', '.join(ensemble_results['best2']['models'])}")

    print("\n===== Final Summary =====")
    print(table)

    return table


def main() -> None:
    args = parse_args()
    run_pipeline(
        args.config,
        args.skip_preprocess,
        args.models,
        only_new=args.only_new,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
