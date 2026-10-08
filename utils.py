"""Small shared helpers (I/O, plotting style, reproducible bootstrap for scripts/notebooks)."""
from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CFG, Config
from .data import ingest, primary_market, simulate_raw


def setup_style() -> None:
    warnings.filterwarnings("ignore")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold", "axes.titlesize": 11,
        "font.size": 9.5, "legend.frameon": False})


def get_panel(cfg: Config = CFG, rebuild: bool = False) -> pd.DataFrame:
    """Return the analysis panel, simulating + ingesting raw data on first use."""
    f = cfg.processed_dir / "panel.csv"
    if rebuild or not f.exists():
        if rebuild or not (cfg.raw_dir / "market_transactions.csv").exists():
            simulate_raw(cfg)
        ingest(cfg)
    return pd.read_csv(f, parse_dates=["week_start"])


def save_json(obj: dict, name: str, cfg: Config = CFG) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / name).write_text(json.dumps(obj, indent=2, default=lambda o: float(o)
                                                   if isinstance(o, (np.floating, np.integer)) else str(o)))


def load_json(name: str, cfg: Config = CFG) -> dict:
    return json.loads((cfg.results_dir / name).read_text())


def savefig(fig, name: str, cfg: Config = CFG) -> None:
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.fig_dir / name, bbox_inches="tight")


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"
