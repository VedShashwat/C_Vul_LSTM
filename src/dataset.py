from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler


class VulnDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor, vuln_features: torch.Tensor) -> None:
        self.sequences = sequences
        self.labels = labels
        self.vuln_features = vuln_features

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx], self.vuln_features[idx]


class BalancedBatchSampler(Sampler[List[int]]):
    """
    Yield 50/50 vulnerable-safe batches with replacement.
    """

    def __init__(self, labels: torch.Tensor, batch_size: int) -> None:
        if batch_size % 2 != 0:
            raise ValueError("BalancedBatchSampler requires an even batch size.")

        self.batch_size = batch_size
        self.half = batch_size // 2
        labels_list = labels.long().cpu().numpy().tolist()

        self.safe_indices = [i for i, y in enumerate(labels_list) if y == 0]
        self.vuln_indices = [i for i, y in enumerate(labels_list) if y == 1]
        if len(self.safe_indices) == 0 or len(self.vuln_indices) == 0:
            raise ValueError("BalancedBatchSampler requires both classes in training labels.")

        self.num_batches = max(1, len(labels_list) // batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        safe_pool = torch.tensor(self.safe_indices, dtype=torch.long)
        vuln_pool = torch.tensor(self.vuln_indices, dtype=torch.long)

        for _ in range(self.num_batches):
            safe_pick = safe_pool[torch.randint(high=safe_pool.numel(), size=(self.half,))]
            vuln_pick = vuln_pool[torch.randint(high=vuln_pool.numel(), size=(self.half,))]
            batch = torch.cat([safe_pick, vuln_pick], dim=0)
            perm = torch.randperm(batch.numel())
            yield batch[perm].tolist()

    def __len__(self) -> int:
        return self.num_batches


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
    use_balanced_batch_sampler = bool(config["training"].get("use_balanced_batch_sampler", True))

    feature_dim = int(config["data"].get("vuln_feature_dim", 32))

    train_data = _load_split(processed_dir, "train")
    val_data = _load_split(processed_dir, "val")
    test_data = _load_split(processed_dir, "test")

    train_labels_list = train_data["labels"].long().cpu().numpy().tolist()
    n_safe = sum(1 for y in train_labels_list if y == 0)
    n_vuln = sum(1 for y in train_labels_list if y == 1)
    print(f"Class distribution — Safe: {n_safe}, Vulnerable: {n_vuln}")

    train_features = train_data.get("vuln_features", torch.zeros((train_data["labels"].size(0), feature_dim)))
    val_features = val_data.get("vuln_features", torch.zeros((val_data["labels"].size(0), feature_dim)))
    test_features = test_data.get("vuln_features", torch.zeros((test_data["labels"].size(0), feature_dim)))

    train_ds = VulnDataset(train_data["sequences"], train_data["labels"], train_features)
    val_ds = VulnDataset(val_data["sequences"], val_data["labels"], val_features)
    test_ds = VulnDataset(test_data["sequences"], test_data["labels"], test_features)

    if use_balanced_batch_sampler:
        print("Training sampler: BalancedBatchSampler (50/50 safe-vulnerable per batch)")
        train_batch_sampler = BalancedBatchSampler(train_data["labels"], batch_size=batch_size)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=0,
            pin_memory=True,
        )
    else:
        print("Training sampler: WeightedRandomSampler")
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
