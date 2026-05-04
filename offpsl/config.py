from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OffPSLConfig:
    num_preferences: int = 256
    batch_size: int = 256
    total_steps: int = 5000
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip_norm: float = 1.0
    generator_hidden_dim: int = 256
    generator_depth: int = 3
    surrogate_hidden_dim: int = 256
    surrogate_depth: int = 3
    surrogate_ensemble_size: int = 5
    surrogate_epochs: int = 200
    surrogate_batch_size: int = 512
    surrogate_learning_rate: float = 0.001
    surrogate_weight_decay: float = 0.00001
    conservative_std_weight: float = 1.0
    smooth_temperature: float = 0.05
    scalarizer_ideal_epsilon: float = 0.2
    lambda_clip: float = 0.02
    support_weight: float = 0.1
    support_covariance_weight: float = 0.05
    support_sigma_min: float = 0.05
    support_sigma_max: float = 0.20
    support_perturbations: int = 4
    support_batch_size: int = 1024
    diversity_weight: float = 0.05
    diversity_bandwidth: float = 0.05
