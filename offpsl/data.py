from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class OfflineDataset:
    x_raw: np.ndarray
    y_raw: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray

    @property
    def x(self) -> np.ndarray:
        return ((self.x_raw - self.x_mean) / self.x_std).astype(np.float32)

    @property
    def y(self) -> np.ndarray:
        denom = np.maximum(self.y_max - self.y_min, 1e-8)
        return ((self.y_raw - self.y_min) / denom).astype(np.float32)

    @property
    def design_dim(self) -> int:
        return int(self.x_raw.shape[1])

    @property
    def objective_dim(self) -> int:
        return int(self.y_raw.shape[1])

    def denormalize_x(self, x: np.ndarray) -> np.ndarray:
        return (x * self.x_std + self.x_mean).astype(np.float32)

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.y_max - self.y_min, 1e-8)
        return ((y - self.y_min) / denom).astype(np.float32)


def load_offline_dataset(task: str, data_dir: str | Path) -> OfflineDataset:
    task_key = task.lower()
    task_dir = Path(data_dir) / task_key
    x_path = task_dir / f"{task_key}-x-0.npy"
    y_path = task_dir / f"{task_key}-y-0.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Expected dataset files at "
            f"{x_path.as_posix()} and {y_path.as_posix()}."
        )

    x_raw = np.load(x_path).astype(np.float32)
    y_raw = np.load(y_path).astype(np.float32)
    if x_raw.ndim != 2 or y_raw.ndim != 2:
        raise ValueError("OffPSL expects two-dimensional x and y arrays.")
    if x_raw.shape[0] != y_raw.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")

    x_mean = x_raw.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = np.maximum(x_raw.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    y_min = y_raw.min(axis=0, keepdims=True).astype(np.float32)
    y_max = y_raw.max(axis=0, keepdims=True).astype(np.float32)
    return OfflineDataset(x_raw, y_raw, x_mean, x_std, y_min, y_max)


def tensorize(dataset: OfflineDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(dataset.x).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(dataset.y).to(device=device, dtype=torch.float32)
    return x, y

