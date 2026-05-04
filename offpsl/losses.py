from __future__ import annotations

import torch


def sample_preferences(
    batch_size: int,
    objective_dim: int,
    lambda_clip: float,
    device: torch.device,
) -> torch.Tensor:
    alpha = torch.full((objective_dim,), 1.0, device=device)
    preferences = torch.distributions.Dirichlet(alpha).sample((batch_size,))
    preferences = preferences.clamp_min(lambda_clip)
    return preferences / preferences.sum(dim=-1, keepdim=True)


def preference_grid(
    num_preferences: int,
    objective_dim: int,
    lambda_clip: float,
    device: torch.device,
) -> torch.Tensor:
    if objective_dim == 2:
        values = torch.linspace(0.0, 1.0, num_preferences, device=device)
        grid = torch.stack([values, 1.0 - values], dim=-1)
    else:
        alpha = torch.ones(objective_dim, device=device)
        grid = torch.distributions.Dirichlet(alpha).sample((num_preferences,))
    grid = grid.clamp_min(lambda_clip)
    return grid / grid.sum(dim=-1, keepdim=True)


def smooth_tchebycheff(
    y: torch.Tensor,
    preferences: torch.Tensor,
    ideal: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    distances = torch.abs(y - ideal.reshape(1, -1))
    weighted = preferences * distances
    tau = max(float(temperature), 1e-8)
    return tau * torch.logsumexp(weighted / tau, dim=-1)


def conservative_prediction(
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    std_weight: float,
) -> torch.Tensor:
    return y_mean + float(std_weight) * y_std


def support_regularizer(
    generated_x: torch.Tensor,
    support_x: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    perturbations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    perturbations = max(int(perturbations), 1)
    batch_size = generated_x.shape[0]
    noisy_targets = []
    penalties = []
    for _ in range(perturbations):
        sigma = torch.empty(batch_size, 1, device=generated_x.device).uniform_(
            float(sigma_min),
            float(sigma_max),
        )
        noised = generated_x + sigma * torch.randn_like(generated_x)
        distances = torch.cdist(noised, support_x, p=2).pow(2)
        nearest = distances.argmin(dim=1)
        target = support_x[nearest]
        noisy_targets.append(target)
        penalties.append(((generated_x - target).pow(2).mean(dim=1) / sigma.squeeze(1).pow(2)).mean())

    support_loss = torch.stack(penalties).mean()
    stacked_targets = torch.stack(noisy_targets, dim=0)
    if perturbations == 1:
        covariance_loss = generated_x.new_zeros(())
    else:
        covariance_loss = stacked_targets.var(dim=0, unbiased=False).sum(dim=1).mean()
    return support_loss, covariance_loss


def diversity_reward(
    y: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    if y.shape[0] < 2:
        return y.new_zeros(())
    distances = torch.cdist(y, y, p=2).pow(2)
    mask = ~torch.eye(y.shape[0], dtype=torch.bool, device=y.device)
    pairwise = distances[mask]
    return 1.0 - torch.exp(-pairwise / max(float(bandwidth), 1e-8)).mean()
