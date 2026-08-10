"""Shared configuration for the transfer-learning lab.

QUICK mode (env var TLLAB_QUICK=1) shrinks every experiment ~10x and writes to
separate artifact directories, so smoke runs never pollute real results.
"""
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"

QUICK = bool(os.environ.get("TLLAB_QUICK"))
_SUFFIX = "_quick" if QUICK else ""
CKPT_DIR = ART_DIR / f"checkpoints{_SUFFIX}"
RESULTS_DIR = ART_DIR / f"results{_SUFFIX}"
FIG_DIR = ART_DIR / f"figures{_SUFFIX}"

SEEDS = [0, 1, 2]
DEFAULT_BATCH = 128
N_THREADS = 4

# Colorblind-validated categorical palette; order matters and is never cycled
# past what a figure needs (see analysis.py for the sequential ramp).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]


def q(n, quick_n):
    """Full-size value normally, small value in QUICK smoke mode."""
    return quick_n if QUICK else n


def seeds():
    """Seed list for multi-seed comparisons (single seed in QUICK mode)."""
    return [0] if QUICK else list(SEEDS)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup() -> None:
    """Call once per notebook: threads, determinism, directories, plot style."""
    import matplotlib.pyplot as plt
    import torch

    torch.set_num_threads(N_THREADS)
    torch.use_deterministic_algorithms(True)
    for d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "font.size": 11,
        "axes.titlesize": 12,
        "figure.dpi": 100,
        "savefig.dpi": 110,
        "text.color": "#0b0b0b",
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "legend.frameon": False,
    })
    print(f"tllab: torch {torch.__version__} | threads={torch.get_num_threads()}"
          f" | QUICK={QUICK} | root={ROOT}")
