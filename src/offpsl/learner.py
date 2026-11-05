"""Offline-only training and portable checkpoint inference."""
from dataclasses import asdict
import platform
import random
import time
import numpy as np
import torch
from .config import Config
from .data import matrix
from .models import Denoiser, Ensemble, ParetoMap, freeze
from .losses import scalarize, sample_sigmas, support_losses, diversity_penalty


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class FittedModel:
    def __init__(self, cfg, method, stats, device="cpu"):
        self.config, self.method = cfg, method
        self.device = torch.device(device)
        self.stats = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in stats.items()}
        s = self.stats
        self.d, self.m = len(s["x_mean"]), len(s["y_min"])
        lower = (s["lower"] - s["x_mean"]) / s["x_scale"]
        upper = (s["upper"] - s["x_mean"]) / s["x_scale"]
        self.map = ParetoMap(self.d, self.m, cfg, lower, upper).to(self.device)
        size = cfg.ensemble_size if method == "offpsl" else 1
        self.surrogate = Ensemble(self.d, self.m, cfg, size).to(self.device)
        self.denoiser = Denoiser(self.d, cfg).to(self.device) if method == "offpsl" else None
        self.report = {}

    @torch.no_grad()
    def predict(self, weights):
        w = matrix(weights, "preferences")
        if w.shape[1] != self.m or np.any(w < 0) or not np.allclose(w.sum(1), 1, atol=1e-5):
            raise ValueError("preferences must be nonnegative rows summing to one, with objective_dim columns")
        self.map.eval()
        self.surrogate.eval()
        s = self.stats
        raw = self.map(torch.as_tensor(w, device=self.device)) * s["x_scale"] + s["x_mean"]
        feasible = raw.clamp(s["lower"], s["upper"])
        mean, std = self.surrogate((feasible - s["x_mean"]) / s["x_scale"])
        result = dict(X=feasible, X_raw=raw, preferences=torch.as_tensor(w, device=self.device),
                      Y_pred=mean * s["y_scale"] + s["y_min"], Y_std=std * s["y_scale"])
        return {k: v.cpu().numpy() for k, v in result.items()}

    def save(self, path):
        torch.save(dict(format_version=1, config=asdict(self.config), method=self.method,
                        stats={k: v.cpu() for k, v in self.stats.items()},
                        map=self.map.state_dict(), surrogate=self.surrogate.state_dict(),
                        denoiser=self.denoiser.state_dict() if self.denoiser else None,
                        report=self.report), path)


def load(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload["format_version"] != 1:
        raise ValueError("unsupported checkpoint version")
    model = FittedModel(Config(**payload["config"]), payload["method"], payload["stats"], device)
    model.map.load_state_dict(payload["map"])
    model.surrogate.load_state_dict(payload["surrogate"])
    freeze(model.surrogate)
    if model.denoiser:
        model.denoiser.load_state_dict(payload["denoiser"])
        freeze(model.denoiser)
    model.report = payload["report"]
    return model


def fit(data, config=None, method="offpsl", seed=42, device="cpu", verbose=True):
    """Train from fixed X/Y only; no evaluation function is accepted or queried."""
    if method not in ("offpsl", "stch-psl"):
        raise ValueError("method must be offpsl or stch-psl")
    cfg = config or Config()
    if cfg.preference_floor >= 1 / data.y.shape[1]:
        raise ValueError("preference_floor must be < 1 / objective_dim")
    seed_all(seed)
    model = FittedModel(cfg, method, data.statistics(), device)
    s, dev = model.stats, model.device
    x = (torch.as_tensor(data.x, device=dev) - s["x_mean"]) / s["x_scale"]
    y = (torch.as_tensor(data.y, device=dev) - s["y_min"]) / s["y_scale"]
    n, batch = len(x), min(cfg.batch_size, len(x))
    timings, history = {}, []
    sync(dev)
    start = time.perf_counter()

    def optimize(loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss; inspect data scaling and configuration")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True)
        optimizer.step()

    # Independent initializations and fixed bootstrap samples for ensemble diversity.
    for member in model.surrogate.members:
        bootstrap = torch.randint(n, (n,), device=dev) if method == "offpsl" else torch.arange(n, device=dev)
        optimizer = torch.optim.AdamW(member.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        for _ in range(cfg.surrogate_steps):
            idx = bootstrap[torch.randint(n, (batch,), device=dev)]
            loss = (member(x[idx]) - y[idx]).square().mean()
            optimize(loss, optimizer, member.parameters())
    freeze(model.surrogate)
    sync(dev)
    timings["surrogate_seconds"] = time.perf_counter() - start
    if verbose:
        print(f"{method}: surrogate trained ({timings['surrogate_seconds']:.1f}s)", flush=True)

    start = time.perf_counter()
    if model.denoiser:
        optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        for _ in range(cfg.denoiser_steps):
            clean = x[torch.randint(n, (batch,), device=dev)]
            sigma = sample_sigmas((batch, 1), cfg, dev)
            loss = (model.denoiser(clean + sigma * torch.randn_like(clean), sigma) - clean).square().sum(-1).mean()
            optimize(loss, optimizer, model.denoiser.parameters())
        freeze(model.denoiser)
    sync(dev)
    timings["denoiser_seconds"] = time.perf_counter() - start
    if verbose and model.denoiser:
        print(f"{method}: denoiser trained ({timings['denoiser_seconds']:.1f}s)", flush=True)

    start = time.perf_counter()
    optimizer = torch.optim.AdamW(model.map.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    distribution = torch.distributions.Dirichlet(torch.ones(model.m, device=dev))
    for step in range(cfg.map_steps):
        w = cfg.preference_floor + (1 - model.m * cfg.preference_floor) * distribution.sample((batch,))
        generated = model.map(w)
        mean, std = model.surrogate(generated)
        conservative = mean + cfg.beta * std if method == "offpsl" else mean
        scalar = scalarize(conservative, w, cfg.temperature, cfg.ideal_offset).mean()
        loss = scalar
        support, covariance, diversity = [generated.new_zeros(()) for _ in range(3)]
        if method == "offpsl":
            support, covariance = support_losses(generated, model.denoiser, cfg)
            diversity = diversity_penalty(conservative) if cfg.diversity_weight else diversity
            loss = loss + cfg.support_weight * support + cfg.covariance_weight * covariance + cfg.diversity_weight * diversity
        optimize(loss, optimizer, model.map.parameters())
        if step % 100 == 0 or step == cfg.map_steps - 1:
            history.append(dict(step=step + 1, loss=float(loss.detach()), scalar=float(scalar.detach()),
                                support=float(support.detach()), covariance=float(covariance.detach()),
                                diversity=float(diversity.detach())))
    model.map.eval()
    sync(dev)
    timings["map_seconds"] = time.perf_counter() - start
    with torch.no_grad():
        mean, _ = model.surrogate(x)
        training_mse = float((mean - y).square().mean())
    model.report = dict(method=method, seed=int(seed), config=asdict(cfg), n_samples=n,
                        design_dim=model.d, objective_dim=model.m, timings=timings, history=history,
                        surrogate_training_mse_minmax=training_mse,
                        environment=dict(python=platform.python_version(), numpy=np.__version__,
                                         torch=str(torch.__version__), device=str(dev),
                                         threads=torch.get_num_threads()))
    if verbose:
        print(f"{method}: map trained ({timings['map_seconds']:.1f}s)", flush=True)
    return model
