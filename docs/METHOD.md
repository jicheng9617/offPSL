# Implementation notes

## Objective and coordinates

All objectives are minimized. Designs are standardized with the offline mean and
population standard deviation (floor `1e-6`); objective coordinates use offline
min/max scaling (range floor `1e-6`). No oracle-derived normalization or reference
point enters training.

For a normalized generated design `x`, the ensemble gives coordinate-wise mean
`mu(x)` and population standard deviation `std(x)`. OffPSL scalarizes
`mu(x) + beta * std(x)` against `z = -ideal_offset`, using
`tau * logsumexp(lambda * abs(y - z) / tau)`.

The denoiser is a small SiLU MLP taking `[noisy_x, log(sigma)]`. It learns clean
designs from Gaussian perturbations with sigma uniformly sampled from
`[sigma_min, sigma_max]`. It is trained from scratch on the same offline designs.
It performs a single denoising step and needs no generative sampler or pretrained
model. Surrogate and denoiser weights remain frozen during map training, while
input gradients propagate through both models.

For `P` independently sampled perturbations per design, `r_p = x - d(x+s_p*xi_p,s_p)`:

| Term | Definition used here |
|:--|:--|
| Support | Mean over perturbations and batch of `sum_d r_p**2` |
| Covariance | Mean over batch and coordinates of unbiased variance over perturbations (`P-1` denominator) |
| Diversity | Mean off-diagonal `exp(-||u_i-u_j||² / eta)`, with conservative objective vectors `u` |

The diversity bandwidth is the detached median squared pair distance, floored at
`1e-8`. Minimizing the positive diversity penalty encourages separation. The full
loss adds the three weighted terms to the scalarization. This exactly identifies
the reductions and scale convention: the support norm is **not divided by sigma²
or design dimension**. The covariance coefficient is zero-compatible when `P=1`.

## Training and inference choices

Each OffPSL surrogate has its own random initialization and a fixed bootstrap sample
of the offline rows. Every member receives `surrogate_steps` optimizer updates;
the denoiser receives `denoiser_steps`; the preference map receives `map_steps`.
All are AdamW-trained. There is no online acquisition, oracle-based early stopping,
hidden checkpoint, or true-front supervision. The reported surrogate training MSE
is an in-sample diagnostic, not a held-out accuracy estimate.

During training, Dirichlet preferences are transformed to the interior simplex:
`floor + (1 - M*floor) * lambda`. This gives an actual lower bound per coordinate.
Inference accepts any nonnegative simplex rows, including vertices; near-vertex
behavior depends on training coverage. All config fields and defaults are in
[`config.py`](../src/offpsl/config.py).

The default `bounded_map=true` applies a sigmoid between normalized legal bounds.
Set it to `false` for unconstrained training outputs with final legal-box projection.
Predictions are always computed at the returned feasible `X`; `X_raw` preserves
the pre-projection design. Dataset ranges are fallback bounds, not a substitute
for task-specific feasibility constraints.

## STCH-PSL adaptation

The original STCH method is due to [Lin et al. (ICML 2024)](https://proceedings.mlr.press/v235/lin24y.html).
This local offline adaptation trains one objective MLP on **all** supplied rows,
freezes it, and trains a preference map with the same scalarizer and objective
normalization. It has no uncertainty, support, covariance, or diversity penalty.
The scalarizer retains the local research code's absolute-distance and outside-ideal
conventions. The original STCH formula can use signed differences when the
reference is a valid lower ideal; an estimated dataset offset need not satisfy that
condition globally.

This release independently implements the mathematical training pattern. It does
not import or copy the original authors' repository. The previous research runner
also offered top-set fitting and restarts; those experiment-specific options are
not part of this compact full-data adaptation. Its quick-start results should not
be read as a reproduction of the original STCH paper or the research benchmark table.

## Relation to the manuscript

The release implements the current manuscript's denoising support and covariance
equations. The older research pipeline used a pre-trained denoiser loaded through
additional project modules, including generative-model dependencies, and included
alternative residual scale conventions and task-specific settings. This release
replaces those dependencies with the directly trained single-step denoiser above.
It uses a compact MLP and a declared optimizer-step schedule; old research
checkpoints are not compatible. No benchmark numbers are transferred to this example.

The manuscript analyzes fixed-reference scalarized regret conditional on surrogate
error, surrogate suboptimality, feasible output membership, and support coverage.
The support-distance result is a small-noise limit requiring unique projection
and controlled candidate-centered denoiser approximation error. Finite-noise
training penalties are practical regularizers, not certificates that those
conditions hold for a particular learned model.
