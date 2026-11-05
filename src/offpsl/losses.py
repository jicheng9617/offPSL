"""Current manuscript objectives, plus STCH scalarization (Lin et al., ICML 2024).

The offline STCH adaptation retains the local absolute-distance convention.
No upstream implementation is copied. See docs/PROVENANCE.md.
"""
import torch


def scalarize(y, weights, temperature, ideal_offset):
    return temperature * torch.logsumexp(weights * (y + ideal_offset).abs() / temperature, dim=-1)


def sample_sigmas(shape, cfg, device):
    # Uniform noise scales in normalized design coordinates.
    return torch.empty(shape, device=device).uniform_(cfg.sigma_min, cfg.sigma_max)


def support_losses(x, denoiser, cfg):
    p, b, d = cfg.perturbations, *x.shape
    sigma = sample_sigmas((p * b, 1), cfg, x.device)
    repeat = x.unsqueeze(0).expand(p, b, d).reshape(p * b, d)
    denoised = denoiser(repeat + sigma * torch.randn_like(repeat), sigma).reshape(p, b, d)
    residual = x.unsqueeze(0) - denoised
    # Manuscript R_sup: squared Euclidean norm, no sigma^-2 or dimension scaling.
    support = residual.square().sum(-1).mean()
    # trace(unbiased sample covariance) / d.
    covariance = residual.var(0, unbiased=True).mean() if p > 1 else x.new_zeros(())
    return support, covariance


def diversity_penalty(y):
    if len(y) < 2:
        return y.new_zeros(())
    distance = torch.pdist(y).square()
    bandwidth = distance.detach().median().clamp_min(1e-8)
    return torch.exp(-distance / bandwidth).mean()
