"""Array-only interface. No benchmark imports, downloads, or oracle calls."""
from dataclasses import dataclass
import numpy as np


def matrix(value, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty, finite 2D numeric array")
    return array


@dataclass
class OfflineData:
    x: np.ndarray
    y: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None

    def __post_init__(self):
        self.x, self.y = matrix(self.x, "X"), matrix(self.y, "Y")
        if len(self.x) != len(self.y) or len(self.x) < 2 or self.y.shape[1] < 2:
            raise ValueError("X/Y need matching rows, at least two samples and two objectives")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("provide both lower and upper, or neither")
        if self.lower is None:
            self.lower, self.upper = self.x.min(0), self.x.max(0)
        self.lower, self.upper = np.asarray(self.lower, np.float32), np.asarray(self.upper, np.float32)
        for bound in (self.lower, self.upper):
            if bound.shape != (self.x.shape[1],) or not np.isfinite(bound).all():
                raise ValueError("bounds must be finite vectors of length design_dim")
        if np.any(self.upper < self.lower):
            raise ValueError("upper bounds must be >= lower bounds")
        if np.any(self.x < self.lower) or np.any(self.x > self.upper):
            raise ValueError("offline X lies outside supplied bounds")

    def statistics(self):
        return dict(x_mean=self.x.mean(0), x_scale=np.maximum(self.x.std(0), 1e-6),
                    y_min=self.y.min(0), y_scale=np.maximum(np.ptp(self.y, axis=0), 1e-6),
                    lower=self.lower, upper=self.upper)

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["X"], data["Y"],
                       data["lower"] if "lower" in data else None,
                       data["upper"] if "upper" in data else None)


def preferences(count, objectives, seed=0, floor=0.01):
    """Two-objective grid, otherwise seeded Dirichlet queries; all rows sum to one."""
    if count < 1 or objectives < 2 or not 0 <= floor < 1 / objectives:
        raise ValueError("invalid count, objective count, or preference floor")
    if objectives == 2:
        t = np.linspace(0, 1, count) if count > 1 else np.array([0.5])
        values = np.column_stack((t, 1 - t))
    else:
        values = np.random.default_rng(seed).dirichlet(np.ones(objectives), count)
    return (floor + (1 - objectives * floor) * values).astype(np.float32)
