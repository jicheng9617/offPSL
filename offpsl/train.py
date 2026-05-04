from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from .config import OffPSLConfig
from .data import load_offline_dataset, tensorize
from .losses import (
    conservative_prediction,
    diversity_reward,
    preference_grid,
    sample_preferences,
    smooth_tchebycheff,
    support_regularizer,
)
from .models import OffPSLGenerator, SurrogateEnsemble


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_surrogates(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: OffPSLConfig,
) -> SurrogateEnsemble:
    model = SurrogateEnsemble(
        design_dim=x.shape[1],
        objective_dim=y.shape[1],
        ensemble_size=cfg.surrogate_ensemble_size,
        hidden_dim=cfg.surrogate_hidden_dim,
        depth=cfg.surrogate_depth,
    ).to(x.device)
    loss_fn = torch.nn.MSELoss()
    for member in model.members:
        optimizer = torch.optim.AdamW(
            member.parameters(),
            lr=cfg.surrogate_learning_rate,
            weight_decay=cfg.surrogate_weight_decay,
        )
        for _ in range(cfg.surrogate_epochs):
            order = torch.randperm(x.shape[0], device=x.device)
            for start in range(0, x.shape[0], cfg.surrogate_batch_size):
                idx = order[start : start + cfg.surrogate_batch_size]
                prediction = member(x[idx])
                loss = loss_fn(prediction, y[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    model.freeze()
    return model


def train_offpsl(
    task: str,
    data_dir: str | Path,
    output_dir: str | Path,
    seed: int,
    device_name: str,
) -> None:
    cfg = OffPSLConfig()
    set_seed(seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )

    dataset = load_offline_dataset(task, data_dir)
    x, y = tensorize(dataset, device)
    surrogate = train_surrogates(x, y, cfg)

    lower = x.min(dim=0).values
    upper = x.max(dim=0).values
    generator = OffPSLGenerator(
        preference_dim=dataset.objective_dim,
        design_dim=dataset.design_dim,
        hidden_dim=cfg.generator_hidden_dim,
        depth=cfg.generator_depth,
        lower_bound=lower,
        upper_bound=upper,
    ).to(device)
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    ideal = torch.full(
        (dataset.objective_dim,),
        -cfg.scalarizer_ideal_epsilon,
        device=device,
        dtype=torch.float32,
    )

    history: list[dict[str, float]] = []
    iterator = trange(cfg.total_steps, desc="Training OffPSL")
    for step in iterator:
        preferences = sample_preferences(
            cfg.batch_size,
            dataset.objective_dim,
            cfg.lambda_clip,
            device,
        )
        generated_x = generator(preferences)
        y_mean, y_std = surrogate.mean_and_std(generated_x)
        y_conservative = conservative_prediction(
            y_mean,
            y_std,
            cfg.conservative_std_weight,
        )
        scalar_loss = smooth_tchebycheff(
            y_conservative,
            preferences,
            ideal,
            cfg.smooth_temperature,
        ).mean()

        support_idx = torch.randint(0, x.shape[0], (cfg.support_batch_size,), device=device)
        support_loss, support_covariance = support_regularizer(
            generated_x,
            x[support_idx],
            cfg.support_sigma_min,
            cfg.support_sigma_max,
            cfg.support_perturbations,
        )
        diversity = diversity_reward(y_mean, cfg.diversity_bandwidth)
        loss = (
            scalar_loss
            + cfg.support_weight * support_loss
            + cfg.support_covariance_weight * support_covariance
            - cfg.diversity_weight * diversity
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        if step % 100 == 0 or step == cfg.total_steps - 1:
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "scalar_loss": float(scalar_loss.detach().cpu()),
                "support_loss": float(support_loss.detach().cpu()),
                "support_covariance": float(support_covariance.detach().cpu()),
                "diversity_reward": float(diversity.detach().cpu()),
            }
            history.append(row)
            iterator.set_postfix(loss=row["loss"])

    eval_preferences = preference_grid(
        cfg.num_preferences,
        dataset.objective_dim,
        cfg.lambda_clip,
        device,
    )
    generator.eval()
    with torch.no_grad():
        generated_x = generator(eval_preferences)
        y_mean, y_std = surrogate.mean_and_std(generated_x)
    designs = dataset.denormalize_x(generated_x.cpu().numpy())
    objectives = y_mean.cpu().numpy()

    task_output = Path(output_dir) / task.lower()
    task_output.mkdir(parents=True, exist_ok=True)
    np.save(task_output / "offpsl_designs.npy", designs)
    np.save(task_output / "offpsl_objectives_minmax.npy", objectives)
    torch.save(
        {
            "generator": generator.state_dict(),
            "surrogate": surrogate.state_dict(),
            "config": asdict(cfg),
            "x_mean": dataset.x_mean,
            "x_std": dataset.x_std,
            "y_min": dataset.y_min,
            "y_max": dataset.y_max,
            "task": task.lower(),
            "seed": seed,
        },
        task_output / "offpsl_checkpoint.pt",
    )
    with (task_output / "offpsl_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"history": history, "config": asdict(cfg)}, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train OffPSL on an offline dataset.")
    parser.add_argument("--task", required=True, help="Task name matching the dataset folder.")
    parser.add_argument("--data_dir", default="offline_moo/data", help="Root directory containing task data.")
    parser.add_argument("--output_dir", default="outputs", help="Directory for OffPSL outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_offpsl(
        task=args.task,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
