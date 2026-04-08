import math
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.utils import AsymmetricLoss, FocalLoss, ensure_dir, resolve_device, save_checkpoint


def _extract_logits(model_output: torch.Tensor):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def _unpack_batch(batch, device):
    sequences = batch[0].to(device, non_blocking=True)
    labels = batch[1].float().to(device, non_blocking=True)
    vuln_features = None
    if len(batch) > 2:
        vuln_features = batch[2].float().to(device, non_blocking=True)
    return sequences, labels, vuln_features


def _forward_model(model, sequences: torch.Tensor, vuln_features: torch.Tensor | None):
    if vuln_features is not None:
        try:
            return model(sequences, vuln_features=vuln_features)
        except TypeError:
            return model(sequences)
    return model(sequences)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_model(
    model,
    train_loader,
    val_loader,
    config: Dict,
    model_name: str,
    use_focal_loss: bool = True,
    use_warmup: bool = False,
) -> Dict[str, list]:
    train_cfg = config["training"]
    results_dir = config["results_dir"]

    device = resolve_device(train_cfg.get("device", "cpu"))
    model = model.to(device)

    if bool(train_cfg.get("use_asymmetric_loss", True)):
        criterion = AsymmetricLoss(
            gamma_neg=float(train_cfg.get("asym_gamma_neg", 4.0)),
            gamma_pos=float(train_cfg.get("asym_gamma_pos", 0.0)),
            clip=float(train_cfg.get("asym_clip", 0.05)),
            reduction="mean",
        )
    elif use_focal_loss:
        criterion = FocalLoss(
            alpha=float(train_cfg.get("focal_alpha", 0.25)),
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            reduction="mean",
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = Adam(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=1e-5)

    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["patience"])
    grad_clip = float(train_cfg["grad_clip"])

    scheduler = None
    if use_warmup:
        num_training_steps = epochs * len(train_loader)
        num_warmup_steps = max(1, num_training_steps // 10)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=1e-6,
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_recall_vuln": [],
        "val_score": [],
    }

    best_val_loss = float("inf")
    best_val_score = float("-inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = -1
    patience_counter = 0

    ensure_dir(results_dir)
    checkpoint_path = str(Path(results_dir) / f"{model_name}_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"{model_name} | Epoch {epoch}/{epochs} [Train]", leave=False)
        for batch in train_bar:
            sequences, labels, vuln_features = _unpack_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)
            logits = _extract_logits(_forward_model(model, sequences, vuln_features))
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * sequences.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        val_bar = tqdm(val_loader, desc=f"{model_name} | Epoch {epoch}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                sequences, labels, vuln_features = _unpack_batch(batch, device)

                logits = _extract_logits(_forward_model(model, sequences, vuln_features))
                loss = criterion(logits, labels)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()

                val_running_loss += loss.item() * sequences.size(0)
                all_val_labels.extend(labels.long().cpu().numpy().tolist())
                all_val_preds.extend(preds.cpu().numpy().tolist())

                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds, average="binary", zero_division=0)
        _, val_recall_vuln, _, _ = precision_recall_fscore_support(
            all_val_labels,
            all_val_preds,
            average="binary",
            zero_division=0,
        )
        val_score = (0.4 * float(val_f1)) + (0.6 * float(val_recall_vuln))

        if not np.isfinite(val_loss):
            print(f"[{model_name}] Non-finite val_loss detected ({val_loss}); skipping checkpoint update.")
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{model_name}] Early stopping at epoch {epoch} due to non-finite validation loss.")
                break
            continue

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_accuracy))
        history["val_f1"].append(float(val_f1))
        history["val_recall_vuln"].append(float(val_recall_vuln))
        history["val_score"].append(float(val_score))

        print(
            f"[{model_name}] Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | val_f1={val_f1:.4f} | "
            f"val_recall_vuln={val_recall_vuln:.4f} | val_score={val_score:.4f}"
        )

        improved = (val_score > best_val_score + 1e-8) or (
            abs(val_score - best_val_score) <= 1e-8 and val_loss < best_val_loss
        )

        if improved:
            best_val_loss = val_loss
            best_val_score = val_score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_accuracy),
                    "val_f1": float(val_f1),
                    "val_recall_vuln": float(val_recall_vuln),
                    "val_score": float(val_score),
                },
                checkpoint_path=checkpoint_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[{model_name}] Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
            break

    model.load_state_dict(best_state)
    print(f"[{model_name}] Restored best model weights from epoch {best_epoch}.")

    return history
