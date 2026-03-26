import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-not-found]
import torch
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils import ensure_dir, resolve_device


def _extract_logits(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def _find_best_threshold_from_probs(probs: Sequence[float], labels: Sequence[int]) -> Tuple[float, float]:
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in [i / 100 for i in range(10, 90, 2)]:
        preds = [1 if p >= thresh else 0 for p in probs]
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def find_optimal_threshold(model, val_loader, device):
    """
    Sweep thresholds from 0.1 to 0.9 on validation set.
    Return threshold that maximizes F1 score for class 1 (vulnerable).
    """

    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in val_loader:
            seqs, labels = batch[0].to(device), batch[1].to(device)
            out = model(seqs)
            logit = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logit).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    best_thresh, best_f1 = _find_best_threshold_from_probs(all_probs, all_labels)
    print(f"  Optimal threshold: {best_thresh:.2f} (val F1: {best_f1:.4f})")
    return best_thresh


def evaluate_model(
    model,
    test_loader,
    model_name: str,
    results_dir: str,
    device_pref: str = "cpu",
    threshold: Optional[float] = None,
    val_loader=None,
) -> Dict[str, float]:
    ensure_dir(results_dir)
    device = resolve_device(device_pref)
    model = model.to(device)
    model.eval()

    if threshold is None:
        if val_loader is None:
            raise ValueError("val_loader is required when threshold is None.")
        threshold = find_optimal_threshold(model, val_loader, device)

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for sequences, labels in tqdm(test_loader, desc=f"Evaluating {model_name}", leave=False):
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = _extract_logits(model(sequences))
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

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
    report += f"\nThreshold: {threshold:.2f}\nAccuracy: {accuracy:.4f}\nAUC: {roc_auc:.4f}\n"

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
        "threshold": float(threshold),
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

    header = "| Model | Threshold | Accuracy | Precision | Recall | F1 | AUC |"
    divider = "|---|---:|---:|---:|---:|---:|---:|"

    lines = [header, divider]
    for model_name, metrics in results_dict.items():
        threshold = metrics.get("threshold", 0.50)
        line = (
            f"| {model_name} | {threshold:.2f} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['auc']:.4f} |"
        )
        lines.append(line)

    table = "\n".join(lines)

    print(table)

    out_path = Path(results_dir) / "comparison_table.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")

    return table


def _metrics_from_probs(labels: Sequence[int], probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc),
        "threshold": float(threshold),
        "fpr": fpr,
        "tpr": tpr,
        "preds": preds,
    }


def _save_report_and_roc(
    labels: Sequence[int],
    preds: np.ndarray,
    probs: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    report_path: Path,
    roc_path: Path,
    title: str,
    threshold: float = None,
) -> None:
    report = str(classification_report(labels, preds, digits=4, zero_division=0))
    if threshold is not None:
        report += f"\nThreshold: {threshold:.2f}\n"
    report += f"Accuracy: {accuracy_score(labels, preds):.4f}\nAUC: {roc_auc:.4f}\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()


def evaluate_ensemble(model_list, model_names, test_loader, val_loader, results_dir, device):
    """
    Soft voting ensemble: average sigmoid probabilities from all models.
    model_list: list of loaded model objects (already on device, in eval mode)
    """

    ensure_dir(results_dir)
    if isinstance(device, str):
        device = resolve_device(device)

    labels: List[int] = []
    labels_val: List[int] = []
    probs_by_model: Dict[str, np.ndarray] = {}
    probs_by_model_val: Dict[str, np.ndarray] = {}

    for model, model_name in zip(model_list, model_names):
        model = model.to(device)
        model.eval()
        model_probs: List[float] = []
        model_probs_val: List[float] = []
        current_labels: List[int] = []
        current_labels_val: List[int] = []

        with torch.no_grad():
            for sequences, y in tqdm(val_loader, desc=f"Ensemble val pass: {model_name}", leave=False):
                sequences = sequences.to(device, non_blocking=True)
                logits = _extract_logits(model(sequences))
                probs = torch.sigmoid(logits)

                model_probs_val.extend(probs.cpu().numpy().tolist())
                current_labels_val.extend(y.cpu().numpy().tolist())

        with torch.no_grad():
            for sequences, y in tqdm(test_loader, desc=f"Ensemble pass: {model_name}", leave=False):
                sequences = sequences.to(device, non_blocking=True)
                logits = _extract_logits(model(sequences))
                probs = torch.sigmoid(logits)

                model_probs.extend(probs.cpu().numpy().tolist())
                current_labels.extend(y.cpu().numpy().tolist())

        if not labels:
            labels = current_labels
        if not labels_val:
            labels_val = current_labels_val

        probs_by_model[model_name] = np.asarray(model_probs, dtype=np.float32)
        probs_by_model_val[model_name] = np.asarray(model_probs_val, dtype=np.float32)

    stacked_val = np.vstack([probs_by_model_val[name] for name in model_names])
    ensemble_probs_val = stacked_val.mean(axis=0)
    ensemble_threshold, ensemble_val_f1 = _find_best_threshold_from_probs(ensemble_probs_val.tolist(), labels_val)
    print(f"  Ensemble optimal threshold: {ensemble_threshold:.2f} (val F1: {ensemble_val_f1:.4f})")

    stacked = np.vstack([probs_by_model[name] for name in model_names])
    ensemble_probs = stacked.mean(axis=0)
    ensemble_metrics = _metrics_from_probs(labels, ensemble_probs, threshold=ensemble_threshold)

    _save_report_and_roc(
        labels=labels,
        preds=ensemble_metrics["preds"],
        probs=ensemble_probs,
        fpr=ensemble_metrics["fpr"],
        tpr=ensemble_metrics["tpr"],
        roc_auc=ensemble_metrics["auc"],
        report_path=Path(results_dir) / "ensemble_report.txt",
        roc_path=Path(results_dir) / "ensemble_roc.png",
        title="Ensemble ROC Curve",
        threshold=ensemble_threshold,
    )

    per_model_thresholds = {}
    model_f1 = {
        name: (
            lambda t: f1_score(labels, (probs_by_model[name] >= t).astype(int), average="binary", zero_division=0)
        )(
            _find_best_threshold_from_probs(probs_by_model_val[name].tolist(), labels_val)[0]
        )
        for name in model_names
    }
    for name in model_names:
        t, _ = _find_best_threshold_from_probs(probs_by_model_val[name].tolist(), labels_val)
        per_model_thresholds[name] = t

    top2 = sorted(model_f1.items(), key=lambda x: x[1], reverse=True)[:2]
    top2_names = [x[0] for x in top2]

    top2_probs_val = np.vstack([probs_by_model_val[name] for name in top2_names]).mean(axis=0)
    top2_threshold, _ = _find_best_threshold_from_probs(top2_probs_val.tolist(), labels_val)

    top2_probs = np.vstack([probs_by_model[name] for name in top2_names]).mean(axis=0)
    top2_metrics = _metrics_from_probs(labels, top2_probs, threshold=top2_threshold)

    _save_report_and_roc(
        labels=labels,
        preds=top2_metrics["preds"],
        probs=top2_probs,
        fpr=top2_metrics["fpr"],
        tpr=top2_metrics["tpr"],
        roc_auc=top2_metrics["auc"],
        report_path=Path(results_dir) / "ensemble_best2_report.txt",
        roc_path=Path(results_dir) / "ensemble_best2_roc.png",
        title=f"Best-2 Ensemble ROC ({', '.join(top2_names)})",
        threshold=top2_threshold,
    )

    pred_matrix = []
    for name in model_names:
        pred_matrix.append((probs_by_model[name] >= per_model_thresholds[name]).astype(int))
    pred_matrix = np.vstack(pred_matrix)
    majority_preds = (pred_matrix.sum(axis=0) >= 2).astype(int)

    majority_report = str(classification_report(labels, majority_preds, digits=4, zero_division=0))
    majority_accuracy = accuracy_score(labels, majority_preds)
    majority_precision, majority_recall, majority_f1, _ = precision_recall_fscore_support(
        labels,
        majority_preds,
        average="binary",
        zero_division=0,
    )
    majority_report += (
        "\nVoting rule: vulnerable if >= 2 models vote vulnerable\n"
        f"Accuracy: {majority_accuracy:.4f}\n"
        f"Precision: {majority_precision:.4f}\n"
        f"Recall: {majority_recall:.4f}\n"
        f"F1: {majority_f1:.4f}\n"
    )
    with open(Path(results_dir) / "ensemble_majority_report.txt", "w", encoding="utf-8") as f:
        f.write(majority_report)

    return {
        "ensemble": {
            "accuracy": ensemble_metrics["accuracy"],
            "precision": ensemble_metrics["precision"],
            "recall": ensemble_metrics["recall"],
            "f1": ensemble_metrics["f1"],
            "auc": ensemble_metrics["auc"],
            "threshold": ensemble_metrics["threshold"],
        },
        "best2": {
            "accuracy": top2_metrics["accuracy"],
            "precision": top2_metrics["precision"],
            "recall": top2_metrics["recall"],
            "f1": top2_metrics["f1"],
            "auc": top2_metrics["auc"],
            "threshold": top2_metrics["threshold"],
            "models": top2_names,
        },
        "majority": {
            "accuracy": float(majority_accuracy),
            "precision": float(majority_precision),
            "recall": float(majority_recall),
            "f1": float(majority_f1),
        },
    }


def load_or_create_test_cwe_labels(config: Dict, expected_size: int) -> List[str]:
    processed_dir = Path(config["data"]["processed_dir"])
    cwe_path = processed_dir / "test_cwe.json"

    if cwe_path.exists():
        with open(cwe_path, "r", encoding="utf-8") as f:
            cwe_labels = json.load(f)
        if isinstance(cwe_labels, list) and len(cwe_labels) == expected_size:
            return cwe_labels

    raw_dir = config["data"]["raw_dir"]
    seed = int(config["training"]["seed"])
    train_ratio = float(config["data"]["train_ratio"])
    val_ratio = float(config["data"]["val_ratio"])
    test_ratio = float(config["data"]["test_ratio"])

    ds = load_from_disk(raw_dir)
    if isinstance(ds, DatasetDict):
        merged = concatenate_datasets([ds[split] for split in ds.keys()])
    else:
        merged = ds

    columns = set(merged.column_names)
    if "func" not in columns or "target" not in columns or "cwe" not in columns:
        cwe_labels = ["UNKNOWN"] * expected_size
        with open(cwe_path, "w", encoding="utf-8") as f:
            json.dump(cwe_labels, f, ensure_ascii=True, indent=2)
        return cwe_labels

    lang_column = None
    for candidate in ["lang", "language", "programming_language"]:
        if candidate in columns:
            lang_column = candidate
            break

    if lang_column is not None:
        merged = merged.filter(lambda x: str(x[lang_column]).strip().lower() == "c")

    funcs, labels, cwes = [], [], []
    for row in merged:
        func = row.get("func")
        target = row.get("target")
        cwe = row.get("cwe", "UNKNOWN")
        if isinstance(func, str) and str(target) in {"0", "1"}:
            funcs.append(func)
            labels.append(int(target))
            cwes.append(str(cwe) if cwe is not None else "UNKNOWN")

    target_size = 18000
    if len(funcs) > target_size:
        all_indices = list(range(len(funcs)))
        sampled_indices, _ = train_test_split(
            all_indices,
            train_size=target_size,
            stratify=labels,
            random_state=seed,
        )
        funcs = [funcs[i] for i in sampled_indices]
        labels = [labels[i] for i in sampled_indices]
        cwes = [cwes[i] for i in sampled_indices]

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    indices = list(range(len(funcs)))
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices,
        labels,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=seed,
    )

    val_portion = val_ratio / (val_ratio + test_ratio)
    _, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_labels,
        test_size=(1.0 - val_portion),
        stratify=temp_labels,
        random_state=seed,
    )

    test_cwe_labels = [cwes[i] for i in test_idx]
    if len(test_cwe_labels) != expected_size:
        test_cwe_labels = (test_cwe_labels + ["UNKNOWN"] * expected_size)[:expected_size]

    with open(cwe_path, "w", encoding="utf-8") as f:
        json.dump(test_cwe_labels, f, ensure_ascii=True, indent=2)

    return test_cwe_labels


def per_cwe_analysis(best_model, test_loader, test_cwe_labels, results_dir):
    """
    Compute F1 by CWE group for the selected best model.
    Reports only CWE groups with at least 20 samples.
    """

    ensure_dir(results_dir)
    device = next(best_model.parameters()).device
    best_model.eval()

    labels: List[int] = []
    preds: List[int] = []
    with torch.no_grad():
        for sequences, y in tqdm(test_loader, desc="Per-CWE inference", leave=False):
            sequences = sequences.to(device, non_blocking=True)
            logits = _extract_logits(best_model(sequences))
            p = torch.sigmoid(logits)
            pred = (p >= 0.5).long()
            labels.extend(y.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())

    n = min(len(labels), len(test_cwe_labels))
    labels = labels[:n]
    preds = preds[:n]
    test_cwe_labels = list(test_cwe_labels[:n])

    grouped: Dict[str, Tuple[List[int], List[int]]] = {}
    for y_true, y_pred, cwe in zip(labels, preds, test_cwe_labels):
        if cwe not in grouped:
            grouped[cwe] = ([], [])
        grouped[cwe][0].append(y_true)
        grouped[cwe][1].append(y_pred)

    rows = []
    for cwe, (y_true_list, y_pred_list) in grouped.items():
        if len(y_true_list) < 20:
            continue
        score = f1_score(y_true_list, y_pred_list, average="binary", zero_division=0)
        rows.append((cwe, len(y_true_list), float(score)))

    rows.sort(key=lambda x: x[2], reverse=True)

    csv_path = Path(results_dir) / "per_cwe_breakdown.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cwe", "count", "f1"])
        writer.writerows(rows)

    if not rows:
        return

    cwes = [x[0] for x in rows]
    f1_vals = [x[2] for x in rows]

    colors = []
    for score in f1_vals:
        if score > 0.80:
            colors.append("seagreen")
        elif score < 0.60:
            colors.append("indianred")
        else:
            colors.append("steelblue")

    plt.figure(figsize=(10, max(6, len(rows) * 0.35)))
    y_pos = np.arange(len(rows))
    plt.barh(y_pos, f1_vals, color=colors)
    plt.yticks(y_pos, cwes)
    plt.gca().invert_yaxis()
    plt.xlabel("F1 Score")
    plt.title("Per-CWE F1 (CWE groups with >= 20 samples)")
    plt.xlim(0.0, 1.0)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "per_cwe_f1.png", dpi=220)
    plt.close()
