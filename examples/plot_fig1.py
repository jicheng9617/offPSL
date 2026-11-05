"""Replot all Fig. 1 methods plus PCD with external original benchmark Y.

This script does not train a model. The bundled JSON contains generated method
results only; benchmark observations are neither bundled nor synthesized.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


def plot(offline_path, results_path, out):
    start = time.perf_counter()
    archive = json.loads(Path(results_path).read_text(encoding="utf-8"))
    digest = hashlib.sha256(Path(offline_path).read_bytes()).hexdigest()
    if digest != archive["offline"]["sha256"]:
        raise ValueError("Offline Y checksum differs from the Fig. 1 source. Use the documented dataset version.")
    offline = np.load(offline_path, allow_pickle=False)
    if offline.shape != (archive["offline"]["rows"], 2) or not np.isfinite(offline).all():
        raise ValueError("Expected finite two-objective ZDT2 observations")
    results = {}
    for name, entry in archive["methods"].items():
        results[name] = {key: np.asarray(entry[key], dtype=np.float64) for key in ("X", "Y_recorded", "preferences") if key in entry}
        r = results[name]
        n = len(r["Y_recorded"])
        if r["Y_recorded"].shape != (n, 2) or ("X" in r and r["X"].shape != (n, archive["design_dim"])):
            raise ValueError("Inconsistent archived candidate dimensions")
        if not all(np.isfinite(a).all() for a in r.values()):
            raise ValueError("Invalid archived values or preference rows")
        if "preferences" in r and (r["preferences"].shape != (n, 2) or np.any(r["preferences"] < 0) or not np.allclose(r["preferences"].sum(1), 1)):
            raise ValueError("Invalid archived preferences")
    # Same reproducible background subsampling as the original manuscript plotter.
    indices = np.random.default_rng(1000).choice(len(offline), 3500, replace=False)
    background = offline[indices]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(14, 10.5), facecolor="white", layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.88])
    data_ax = fig.add_subplot(grid[0, 0])
    specs = [("mh", "b   MultiHead", "#0072B2"), ("mobo", "c   MOBO", None),
             ("pcd", "d   PCD", "#CC79A7"), ("paretoflow", "e   ParetoFlow", "#009E73"),
             ("pgd_moo", "f   PGD-MOO", "#E69F00"), ("stch-psl", "g   STCH-PSL", None),
             ("offpsl", "h   OffPSL (Ours)", None)]
    method_axes = [fig.add_subplot(grid[(i + 1) // 4, (i + 1) % 4], sharex=data_ax, sharey=data_ax) for i in range(7)]
    map_ax = fig.add_subplot(grid[2, :2])
    tail_ax = fig.add_subplot(grid[2, 2:])
    f1 = np.linspace(0, 1, 400)
    norm, cmap = Normalize(0, 1), plt.get_cmap("viridis")
    upper = max(float(offline[:, 1].max()), max(r["Y_recorded"][:, 1].max() for r in results.values())) * 1.08
    lower = min(0., min(r["Y_recorded"][:, 1].min() for r in results.values())) - 0.22
    xmin = min(0., min(r["Y_recorded"][:, 0].min() for r in results.values())) - 0.04
    xmax = max(1., max(r["Y_recorded"][:, 0].max() for r in results.values())) + 0.04
    data_ax.scatter(*background.T, s=5, color="#9ba5ae", alpha=0.35, rasterized=True)
    data_ax.plot(f1, 1 - f1**2, color="#202c36", lw=1.5)
    data_ax.set(title="a   Original offline dataset", xlabel="$f_1$  (minimize)", ylabel="$f_2$  (minimize)",
                xlim=(xmin, xmax), ylim=(lower, upper))
    data_ax.text(0.04, 0.97, "60,000 observations · 30 variables\n3,500 shown · seed 1000", transform=data_ax.transAxes, va="top", fontsize=8)
    data_ax.legend(handles=[Line2D([], [], color="#202c36", lw=1.5, label="True Pareto front")],
                   loc="lower left", bbox_to_anchor=(0.02, 0.16), fontsize=8, frameon=False)
    for ax, (method, title, color) in zip(method_axes, specs):
        r = results[method]
        ax.scatter(*background.T, s=5, color="#b6bdc5", alpha=0.20, rasterized=True)
        ax.plot(f1, 1 - f1**2, color="#202c36", lw=1.5, zorder=2)
        style = dict(c=r["preferences"][:, 0], cmap=cmap, norm=norm) if color is None else dict(color=color)
        ax.scatter(*r["Y_recorded"].T, **style, s=18, edgecolors="white", linewidths=0.2, zorder=3)
        ax.set(title=title, xlabel="$f_1$  (minimize)")
        ax.text(0.96, 0.95, f"{len(r['Y_recorded'])} outputs", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#53616d")
        ax.grid(alpha=0.12)
        if method == "paretoflow":
            ax.set_ylabel("$f_2$  (minimize)")
        if method == "offpsl":
            ax.title.set_fontweight("bold")
    for ax in [data_ax] + method_axes:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 2, 4, 6, 8])
    for method, label, marker, style, color in [("offpsl", "OffPSL", "o", "-", "#0072B2"),
                                              ("stch-psl", "STCH-PSL", "^", "--", "#D55E00")]:
        r = results[method]
        order = np.argsort(r["preferences"][:, 0], kind="stable")
        w, x = r["preferences"][order, 0], r["X"][order]
        map_ax.plot(w, x[:, 0], style, color="#65727e", lw=1, alpha=0.45)
        map_ax.scatter(w, x[:, 0], c=w, cmap=cmap, norm=norm, marker=marker, s=16)
        tail_ax.plot(w, x[:, 1:].mean(1), style, color=color, lw=2, label=label)
    map_ax.set(title="i   Query a preference, obtain a design", xlabel=r"Preference weight $\lambda_1$  (larger → prioritize $f_1$)",
               ylabel="Saved design coordinate $x_1$", xlim=(0, 1), ylim=(-0.04, 1.04))
    map_ax.legend(handles=[Line2D([], [], marker="o", color="#65727e", linestyle="-", label="OffPSL"),
                           Line2D([], [], marker="^", color="#65727e", linestyle="--", label="STCH-PSL")], loc="upper right", frameon=False)
    tail_ax.axhline(0, color="#202c36", lw=1, label="Pareto-set value")
    tail_ax.set(title="j   Remaining design coordinates", xlabel=r"Preference weight $\lambda_1$", ylabel=r"Mean of $x_2,\ldots,x_{30}$")
    tail_ax.legend(fontsize=8, frameon=False)
    for ax in (map_ax, tail_ax):
        ax.grid(alpha=0.15)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=method_axes[3:], location="bottom", shrink=0.67, pad=0.025, aspect=55)
    bar.set_label(r"Recorded preference $\lambda_1$  ·  MOBO, STCH-PSL and OffPSL only", fontsize=9)
    fig.suptitle("OffPSL  |  ZDT2 with the original offline benchmark data", fontsize=16, fontweight="bold")
    fig.get_layout_engine().set(h_pad=0.14, w_pad=0.10)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out.with_suffix("." + suffix), dpi=180)
    plt.close(fig)
    report = dict(source="All six Fig. 1 methods plus archived PCD results", training="none; replot only", seed=archive["seed"],
                  offline_sha256=digest, offline_rows=len(offline), displayed_offline_rows=len(background),
                  background_sampling_seed=1000, candidate_counts={k: len(v["Y_recorded"]) for k, v in results.items()},
                  preference_colored=[k for k,v in results.items() if "preferences" in v],
                  shared_objective_ylim=[float(lower), float(upper)], shared_objective_xlim=[float(xmin), float(xmax)], numpy=np.__version__, matplotlib=matplotlib.__version__,
                  plot_seconds=time.perf_counter()-start)
    out.with_name(out.stem + "_plot.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-y", required=True, help="Externally downloaded original zdt2-y-0.npy")
    parser.add_argument("--results", default="assets/fig1_results.json")
    parser.add_argument("--out", default="runs/fig1/zdt2")
    args = parser.parse_args()
    plot(args.offline_y, args.results, args.out)
