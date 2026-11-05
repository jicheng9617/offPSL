"""Small, explicit training configuration; step counts are optimizer updates."""
from dataclasses import dataclass, fields
import json
import math


@dataclass(frozen=True)
class Config:
    hidden_dim: int = 128
    depth: int = 3
    ensemble_size: int = 5
    surrogate_steps: int = 2000
    denoiser_steps: int = 2000
    map_steps: int = 2000
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    beta: float = 1.0
    temperature: float = 0.05
    ideal_offset: float = 0.2
    support_weight: float = 0.1
    covariance_weight: float = 0.05
    diversity_weight: float = 0.01
    sigma_min: float = 0.05
    sigma_max: float = 0.2
    perturbations: int = 4
    preference_floor: float = 0.01
    bounded_map: bool = True

    def __post_init__(self):
        for key in ("hidden_dim", "depth", "ensemble_size", "surrogate_steps",
                    "denoiser_steps", "map_steps", "batch_size", "perturbations"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        for key in ("beta", "support_weight", "covariance_weight", "diversity_weight", "weight_decay"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.learning_rate <= 0 or self.temperature <= 0 or self.ideal_offset <= 0:
            raise ValueError("learning_rate, temperature and ideal_offset must be positive")
        if not 0 < self.sigma_min <= self.sigma_max:
            raise ValueError("require 0 < sigma_min <= sigma_max")
        if not 0 <= self.preference_floor < 1:
            raise ValueError("preference_floor must lie in [0, 1)")
        if self.covariance_weight > 0 and self.perturbations < 2:
            raise ValueError("covariance regularization requires at least two perturbations")
        if type(self.bounded_map) is not bool:
            raise ValueError("bounded_map must be boolean")

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as handle:
            return cls(**json.load(handle))
