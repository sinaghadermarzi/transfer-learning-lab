"""Plotting and analysis helpers.

Style notes: categorical series use config.PALETTE in fixed slot order (never
cycled past what a figure needs); magnitude heatmaps use a single-hue blue
ramp; identity is never carried by color alone (legends + direct annotation).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn.functional as F

from . import config, data

# Single-hue sequential ramp (light -> dark blue) for magnitude heatmaps.
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("tllab_blues", BLUES)


def savefig(name: str) -> None:
    """Save the current figure under artifacts/figures/ (and keep it inline)."""
    fig = plt.gcf()
    fig.tight_layout()
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"figure -> {path.relative_to(config.ROOT)}")


def plot_curves(df: pd.DataFrame, *, x: str, y: str, hue: str, ax=None,
                logx: bool = False, ylabel: str | None = None,
                title: str | None = None, marker: str = "o"):
    """One mean line per `hue` value with a min-max band across seeds."""
    ax = ax or plt.gca()
    for i, label in enumerate(dict.fromkeys(df[hue])):
        gdf = df[df[hue] == label]
        agg = gdf.groupby(x)[y].agg(["mean", "min", "max"])
        color = config.PALETTE[i % len(config.PALETTE)]
        ax.plot(agg.index, agg["mean"], marker=marker, label=str(label), color=color)
        ax.fill_between(agg.index, agg["min"], agg["max"], alpha=0.15,
                        color=color, linewidth=0)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_bars(labels, means, spreads=None, *, ax=None, ylabel="test accuracy",
              title=None, baseline=None, baseline_label="baseline",
              color_idx=None):
    """Bar chart with optional min-max whiskers and a reference line."""
    ax = ax or plt.gca()
    xs = np.arange(len(labels))
    idx = color_idx if color_idx is not None else range(len(labels))
    colors = [config.PALETTE[i % len(config.PALETTE)] for i in idx]
    yerr = None
    if spreads is not None:
        lo = [m - s[0] for m, s in zip(means, spreads)]
        hi = [s[1] - m for m, s in zip(means, spreads)]
        yerr = [lo, hi]
    ax.bar(xs, means, 0.62, color=colors, yerr=yerr, capsize=4,
           error_kw={"ecolor": "#52514e", "elinewidth": 1.2})
    for xi, m in zip(xs, means):
        ax.annotate(f"{m:.2f}", (xi, m), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, color="#0b0b0b")
    if baseline is not None:
        ax.axhline(baseline, color="#898781", linestyle="--", linewidth=1.2)
        ax.annotate(baseline_label, (len(labels) - 0.5, baseline),
                    textcoords="offset points", xytext=(0, 4), ha="right",
                    fontsize=9, color="#52514e")
    ax.set_xticks(xs, labels)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="x", visible=False)
    return ax


def plot_transfer_matrix(df: pd.DataFrame, *, source_col="source",
                         target_col="target", value_col="probe_acc",
                         sources=None, targets=None, ax=None,
                         title="Linear-probe transfer accuracy"):
    """Sources x targets heatmap with annotated cells (single-hue ramp)."""
    pivot = df.pivot_table(index=source_col, columns=target_col,
                           values=value_col, aggfunc="mean")
    if sources is not None:
        pivot = pivot.reindex(sources)
    if targets is not None:
        pivot = pivot.reindex(columns=targets)
    ax = ax or plt.gca()
    vals = pivot.to_numpy()
    im = ax.imshow(vals, cmap=SEQ_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = vals[r, c]
            ax.annotate(f"{v:.2f}", (c, r), ha="center", va="center", fontsize=9,
                        color="white" if v > 0.55 else "#0b0b0b")
    ax.set_xlabel("target task")
    ax.set_ylabel("source model")
    ax.set_title(title)
    ax.grid(visible=False)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=value_col)
    return ax


def leep(source_model, target_task: data.Task, n: int = 2000,
         seed: int = 0) -> float:
    """LEEP transferability estimate (Nguyen et al., ICML 2020).

    Uses the source model's softmax over its own label set as a "dummy" joint
    with the target labels; higher (less negative) = better expected transfer.
    """
    idx = data.stratified_indices(target_task.train_y,
                                  min(n, len(target_task.train_y)), seed)
    x, y = target_task.train_x[idx], target_task.train_y[idx]
    source_model.eval()
    with torch.no_grad():
        pz = torch.cat([F.softmax(source_model(x[i:i + 1024]), dim=1)
                        for i in range(0, len(x), 1024)])
    joint = torch.zeros(target_task.n_classes, pz.shape[1])
    joint.index_add_(0, y, pz)
    joint /= len(y)
    p_y_given_z = joint / joint.sum(dim=0, keepdim=True).clamp_min(1e-12)
    eep = (pz @ p_y_given_z.T)[torch.arange(len(y)), y]
    return float(torch.log(eep.clamp_min(1e-12)).mean())


def spearman(a, b) -> float:
    """Spearman rank correlation without scipy."""
    ra = pd.Series(a).rank()
    rb = pd.Series(b).rank()
    return float(ra.corr(rb))


def claim(name: str, ok: bool, detail: str = "", hard: bool = False) -> None:
    """Machine-greppable result check printed into the executed notebook.

    hard=True raises on failure (build-breaking); otherwise a WARN is printed
    for effects that are legitimately fragile at this experimental scale.
    """
    status = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"CLAIM CHECK [{status}] {name}" + (f" — {detail}" if detail else ""))
    if hard and not ok:
        raise AssertionError(f"claim failed: {name} — {detail}")
