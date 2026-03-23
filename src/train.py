from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
from tqdm import tqdm

from src.utils import ensure_dir, resolve_device, save_checkpoint


def _extract_logits(model_output: torch.Tensor):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def train_model(model, train_loader, val_loader, config: Dict, model_name: str) -> Dict[str, list]:
    train_cfg = config["training"]
    results_dir = config["results_dir"]

    device = resolve_device(train_cfg.get("device", "cpu"))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=1e-5)

    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["patience"])
    grad_clip = float(train_cfg["grad_clip"])

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = -1
    patience_counter = 0

    ensure_dir(results_dir)
    checkpoint_path = str(Path(results_dir) / f"{model_name}_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"{model_name} | Epoch {epoch}/{epochs} [Train]", leave=False)
        for sequences, labels in train_bar:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = _extract_logits(model(sequences))
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_loss += loss.item() * sequences.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        val_bar = tqdm(val_loader, desc=f"{model_name} | Epoch {epoch}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for sequences, labels in val_bar:
                sequences = sequences.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)

                logits = _extract_logits(model(sequences))
                loss = criterion(logits, labels)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()

                val_running_loss += loss.item() * sequences.size(0)
                all_val_labels.extend(labels.long().cpu().numpy().tolist())
                all_val_preds.extend(preds.cpu().numpy().tolist())

                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds, average="macro", zero_division=0)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_accuracy))
        history["val_f1"].append(float(val_f1))

        print(
            f"[{model_name}] Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
