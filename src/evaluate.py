from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-not-found]
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from tqdm import tqdm

from src.utils import ensure_dir, resolve_device


def _extract_logits(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def evaluate_model(model, test_loader, model_name: str, results_dir: str, device_pref: str = "cpu") -> Dict[str, float]:
    ensure_dir(results_dir)
    device = resolve_device(device_pref)
    model = model.to(device)
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for sequences, labels in tqdm(test_loader, desc=f"Evaluating {model_name}", leave=False):
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = _extract_logits(model(sequences))
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        zero_division=0,
    )

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    report = str(classification_report(all_labels, all_preds, digits=4, zero_division=0))
    report += f"\nAccuracy: {accuracy:.4f}\nAUC: {roc_auc:.4f}\n"

    report_path = Path(results_dir) / f"{model_name}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(Path(results_dir) / f"{model_name}_confusion.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(Path(results_dir) / f"{model_name}_roc.png", dpi=200)
    plt.close()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc),
    }


def plot_training_curves(histories: List[Dict[str, List[float]]], model_names: List[str], results_dir: str) -> None:
    ensure_dir(results_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for history, model_name in zip(histories, model_names):
        epochs = np.arange(1, len(history["val_loss"]) + 1)
        axes[0].plot(epochs, history["val_loss"], label=model_name, linewidth=2)
        axes[1].plot(epochs, history["val_f1"], label=model_name, linewidth=2)

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(results_dir) / "training_curves.png", dpi=200)
    plt.close()


def generate_comparison_table(results_dict: Dict[str, Dict[str, float]], results_dir: str) -> str:
    ensure_dir(results_dir)

    header = "| Model | Accuracy | Precision | Recall | F1 | AUC |"
    divider = "|---|---:|---:|---:|---:|---:|"

    lines = [header, divider]
    for model_name, metrics in results_dict.items():
        line = (
            f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['auc']:.4f} |"
        )
        lines.append(line)

    table = "\n".join(lines)

    print(table)

    out_path = Path(results_dir) / "comparison_table.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")

    return table
