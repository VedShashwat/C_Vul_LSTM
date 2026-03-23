import argparse
from pathlib import Path
from typing import List

from src.dataset import get_dataloaders
from src.evaluate import evaluate_model, generate_comparison_table, plot_training_curves
from src.models import build_model
from src.preprocess import preprocess_and_save
from src.train import train_model
from src.utils import load_config, seed_everything


DEFAULT_MODELS = ["lstm", "bilstm", "bilstm_attn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM-based vulnerability detectors")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing stage")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Models to train/evaluate",
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


def run_pipeline(config_path: str, skip_preprocess: bool, model_names: List[str]) -> str:
    config = load_config(config_path)
    seed_everything(int(config["training"]["seed"]))

    processed_dir = Path(config["data"]["processed_dir"])
    has_processed = _processed_artifacts_exist(processed_dir)

    if not has_processed and skip_preprocess:
        raise FileNotFoundError("Processed data not found, but --skip-preprocess was set.")

    if not has_processed:
        print("Processed data not found. Running preprocessing...")
        preprocess_and_save(config_path=config_path)
    else:
        print("Processed data found. Skipping preprocessing.")

    train_loader, val_loader, test_loader = get_dataloaders(config)

    import json

    with open(processed_dir / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    pad_idx = vocab.get("<PAD>", 0)

    histories = []
    eval_results = {}

    for model_name in model_names:
        print(f"\n===== Running {model_name} =====")

        model = build_model(
            model_name=model_name,
            vocab_size=vocab_size,
            embed_dim=int(config["model"]["embed_dim"]),
            hidden_size=int(config["model"]["hidden_size"]),
            num_layers=int(config["model"]["num_layers"]),
            dropout=float(config["model"]["dropout"]),
            pad_idx=pad_idx,
            attention_dim=int(config["model"]["attention_dim"]),
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            model_name=model_name,
        )
        histories.append(history)

        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            model_name=model_name,
            results_dir=config["results_dir"],
            device_pref=config["training"].get("device", "cpu"),
        )
        eval_results[model_name] = metrics

    plot_training_curves(histories, model_names, config["results_dir"])
    table = generate_comparison_table(eval_results, config["results_dir"])

    print("\n===== Final Summary =====")
    print(table)

    return table


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, args.skip_preprocess, args.models)


if __name__ == "__main__":
    main()
