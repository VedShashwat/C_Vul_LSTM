from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class VulnDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor) -> None:
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def _load_split(processed_dir: str, split: str) -> Dict[str, torch.Tensor]:
    split_path = Path(processed_dir) / f"{split}_data.pt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing processed split file: {split_path}")
    return torch.load(split_path)


def _build_weighted_sampler(labels: torch.Tensor, oversample_factor: float = 3.0) -> WeightedRandomSampler:
    labels = labels.long()
    train_labels = labels.cpu().numpy().tolist()
    n_safe = sum(1 for l in train_labels if l == 0)
    n_vuln = sum(1 for l in train_labels if l == 1)

    if n_vuln == 0:
        weight_vuln = 1.0
    else:
        weight_vuln = (n_safe / n_vuln) * oversample_factor
    weight_safe = 1.0
    sample_weights = [weight_vuln if l == 1 else weight_safe for l in train_labels]

    print(f"Class distribution — Safe: {n_safe}, Vulnerable: {n_vuln}")
    print(f"Effective sampling weight ratio vuln:safe = {weight_vuln:.1f}:1")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True,
    )
    return sampler


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    processed_dir = config["data"]["processed_dir"]
    batch_size = int(config["training"]["batch_size"])
    oversample_factor = float(config["training"].get("oversample_factor", 3.0))

    train_data = _load_split(processed_dir, "train")
    val_data = _load_split(processed_dir, "val")
    test_data = _load_split(processed_dir, "test")

    train_ds = VulnDataset(train_data["sequences"], train_data["labels"])
    val_ds = VulnDataset(val_data["sequences"], val_data["labels"])
    test_ds = VulnDataset(test_data["sequences"], test_data["labels"])

    train_sampler = _build_weighted_sampler(train_data["labels"], oversample_factor=oversample_factor)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
