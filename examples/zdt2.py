"""Reproducible ZDT2 illustration. No downloaded data or pretrained checkpoint.

Six design dimensions make the example compact; it is not the paper's benchmark
protocol. The analytic oracle labels the fixed dataset before training and is
called again only to evaluate final candidates. It never enters fit().
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from offpsl import Config, OfflineData, fit
from offpsl.data import preferences


def zdt2(x):
    x = np.asarray(x, np.float64)
    g = 1 + 9 * x[:, 1:].mean(1)
    return np.column_stack((x[:, 0], g - x[:, 0] ** 2 / g)).astype(np.float32)


def make_data(seed, n=4096, d=6):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, d)).astype(np.float32)
    # A declared mixture: 75% near-front-biased designs, 25% full-box exploration.
    # No exact Pareto solutions are inserted, and no oracle-based filtering occurs.
    near = rng.random(n) < 0.75
    x[near, 1:] *= (0.2 * rng.random((near.sum(), 1)) ** 3).astype(np.float32)
    return OfflineData(x, zdt2(x), np.zeros(d, np.float32), np.ones(d, np.float32))


def plot_run(run, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable

    run, destination = Path(run), Path(destination)
    with np.load(run / "offline.npz") as data:
        offline = data["Y"]
    results = {}
    for method in ("offpsl", "stch-psl"):
        with np.load(run / f"{method}.npz") as data:
            results[method] = {k: data[k].copy() for k in data.files}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(12, 7.4), facecolor="white", layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
    ax_data = fig.add_subplot(grid[0, 0])
    ax_off = fig.add_subplot(grid[0, 1])
    ax_stch = fig.add_subplot(grid[0, 2], sharex=ax_off, sharey=ax_off)
    ax_map = fig.add_subplot(grid[1, :2])
    ax_tail = fig.add_subplot(grid[1, 2])
    f1 = np.linspace(0, 1, 400)
    cmap, norm = plt.get_cmap("viridis"), Normalize(0, 1)
    ax_data.scatter(*offline.T, s=4, color="#a6adb4", alpha=0.35, rasterized=True)
    ax_data.plot(f1, 1 - f1**2, color="#202c36", lw=1.5)
    ax_data.set(title="a   Fixed offline dataset", xlabel="$f_1$  (minimize)", ylabel="$f_2$  (minimize)")
    # The zoom boundary is shown on the full dataset view, and stated in the caption.
    zoom_top = max(1.25, max(float(r["Y_true"][:, 1].max()) for r in results.values()) * 1.07)
    from matplotlib.patches import Rectangle
    ax_data.add_patch(Rectangle((0, -0.05), 1, zoom_top + 0.05, fill=False, edgecolor="#0072B2", lw=1, ls="--"))
    ax_data.text(0.04, 0.95, "4,096 designs · 6 variables", transform=ax_data.transAxes, va="top", fontsize=9)
    ax_data.legend(handles=[Line2D([], [], color="#202c36", lw=1.5, label="True Pareto front"),
                            Line2D([], [], color="#0072B2", ls="--", label="Zoom in b–c")],
                   loc="upper right", bbox_to_anchor=(1, 0.88), fontsize=8, frameon=False)
    for ax, key, title in ((ax_off, "offpsl", "b   OffPSL"), (ax_stch, "stch-psl", "c   STCH-PSL · offline adaptation")):
        r = results[key]
        ax.scatter(*offline.T, s=5, color="#b6bdc5", alpha=0.20, rasterized=True)
        ax.plot(f1, 1 - f1**2, color="#202c36", lw=1.5, zorder=2)
        ax.scatter(*r["Y_true"].T, c=r["preferences"][:, 0], cmap=cmap, norm=norm,
                   s=20, edgecolors="white", linewidths=0.25, zorder=3)
        ax.set(title=title, xlabel="$f_1$  (minimize)", xlim=(-0.04, 1.04), ylim=(-0.06, zoom_top))
        ax.grid(alpha=0.12)
        if np.ptp(r["X"][:, 0]) < 0.02:
            ax.text(0.05, 0.12, f"{len(r['X'])} query outputs overlap\nnear one design in this run",
                    transform=ax.transAxes, fontsize=8, color="#53616d")
    for key, label, marker, style in (("offpsl", "OffPSL", "o", "-"), ("stch-psl", "STCH-PSL", "^", "--")):
        r = results[key]
        w = r["preferences"][:, 0]
        ax_map.plot(w, r["X"][:, 0], style, color="#65727e", lw=1, alpha=0.5)
        ax_map.scatter(w, r["X"][:, 0], c=w, cmap=cmap, norm=norm, marker=marker, s=19, label=label)
        ax_tail.plot(w, r["X"][:, 1:].mean(1), style, color="#0072B2" if key == "offpsl" else "#D55E00", lw=2, label=label)
    ax_map.set(title="d   Query a preference, obtain a design", xlabel=r"Preference weight $\lambda_1$  (larger → prioritize $f_1$)", ylabel="Returned design coordinate $x_1$", xlim=(0, 1), ylim=(-0.04, 1.04))
    ax_map.legend(handles=[Line2D([], [], marker="o", color="#65727e", linestyle="-", label="OffPSL"),
                           Line2D([], [], marker="^", color="#65727e", linestyle="--", label="STCH-PSL")], loc="upper right", frameon=False)
    ax_tail.axhline(0, color="#202c36", lw=1, label="Pareto-set value")
    ax_tail.set(title="e   Remaining design coordinates", xlabel=r"Preference weight $\lambda_1$", ylabel=r"Mean of $x_2,\ldots,x_6$")
    ax_tail.legend(fontsize=8, frameon=False)
    for ax in (ax_map, ax_tail):
        ax.grid(alpha=0.15)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=[ax_off, ax_stch], location="bottom", shrink=0.8, pad=0.02, aspect=35)
    bar.set_label(r"Preference weight $\lambda_1$", fontsize=9)
    fig.suptitle("OffPSL  |  From offline observations to a queryable trade-off map", fontsize=16, fontweight="bold")
    fig.get_layout_engine().set(h_pad=0.14, w_pad=0.10)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(destination.with_suffix("." + ext), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/zdt2-seed42")
    parser.add_argument("--config", default="configs/zdt2.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.out)
    if args.plot_only:
        plot_run(output, output / "zdt2")
        return
    if output.exists() and any(output.iterdir()):
        parser.error("--out must be new or empty")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    cfg = Config.from_json(args.config)
    data = make_data(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "offline.npz", X=data.x, Y=data.y, lower=data.lower, upper=data.upper)
    weights = preferences(101, 2, args.seed, cfg.preference_floor)
    report = dict(seed=args.seed, example="ZDT2, six variables, declared offline mixture",
                  data_sha256=hashlib.sha256(data.x.tobytes() + data.y.tobytes()).hexdigest(), methods={})
    for method in ("offpsl", "stch-psl"):
        model = fit(data, cfg, method, args.seed, args.device)
        model.save(output / f"{method}.pt")
        candidates = model.predict(weights)
        # Evaluation only, after fitting and inference have finished.
        candidates["Y_true"] = zdt2(candidates["X"])
        np.savez_compressed(output / f"{method}.npz", **candidates)
        report["methods"][method] = model.report
        report["methods"][method]["evaluation"] = dict(
            objective_mae=float(np.abs(candidates["Y_pred"] - candidates["Y_true"]).mean()),
            x1_range=[float(candidates["X"][:, 0].min()), float(candidates["X"][:, 0].max())],
            mean_tail_coordinate=float(candidates["X"][:, 1:].mean()))
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_run(output, output / "zdt2")
    print(f"Example complete: {output}")


if __name__ == "__main__":
    main()
