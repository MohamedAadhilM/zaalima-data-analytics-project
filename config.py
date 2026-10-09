from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
@dataclass(frozen=True)
class Config:
    # ---- simulation / panel geometry -------------------------------------
    seed: int = 2026
    n_markets: int = 40
    n_weeks: int = 104
    start_date: str = "2023-01-02"
    n_treated: int = 12
    primary_adoption_week: int = 70          # first treated week of the focal market

    # ---- analysis windows (week index, 0-based) --------------------------
    train_end: int = 52                      # weeks [0, 52)  : fit synthetic control
    # weeks [52, T0)                         : blocked pre-treatment validation
    # weeks [T0, T)                          : post-treatment

    # ---- inference --------------------------------------------------------
    alpha: float = 0.05
    n_folds: int = 5

    # ---- paths -------------------------------------------------------------
    raw_dir: Path = field(default=ROOT / "data" / "raw")
    processed_dir: Path = field(default=ROOT / "data" / "processed")
    results_dir: Path = field(default=ROOT / "results")
    fig_dir: Path = field(default=ROOT / "reports" / "figures")
    artifact_dir: Path = field(default=ROOT / "artifacts")
    report_dir: Path = field(default=ROOT / "reports")

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.results_dir,
                  self.fig_dir, self.artifact_dir, self.report_dir):
            p.mkdir(parents=True, exist_ok=True)


CFG = Config()
