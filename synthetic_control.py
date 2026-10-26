"""Synthetic Control Method (Abadie-Diamond-Hainmueller) with
  * simplex-constrained weights solved as a convex QP (cvxpy, DPP-compiled once),
  * intercept-adjusted (demeaned) matching on the full weekly pre-treatment outcome path,
  * ridge-penalised weights (Abadie & L'Hour) whose strength is chosen by *blocked
    out-of-sample* validation MSPE (train window -> validation window), which prevents the
    single-donor over-fit that unpenalised SCM exhibits on short, noisy pre-periods,
  * spillover-aware donor screening (SUTVA),
  * placebo-in-space / placebo-in-time inference, leave-one-donor-out, rolling-origin CV.

All outcome modelling is done on log revenue; counterfactual revenue is exp() of the
synthetic log-path with a small Jensen correction from the pre-period residual variance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

STATIC = ["income_z", "urban_index", "log_pop", "comp_intensity"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def outcome_matrix(panel: pd.DataFrame, col: str = "log_revenue") -> pd.DataFrame:
    """weeks x markets matrix."""
    return panel.pivot(index="week_idx", columns="market_id", values=col).sort_index()


def screen_donors(panel: pd.DataFrame, primary: str, adjacency: pd.DataFrame,
                  exposure_cap: float = 0.5) -> tuple[list[str], list[str]]:
    """Return (kept, dropped) donors. Never-treated markets are dropped when they are
    adjacent to the focal market or heavily exposed to neighbours' adoption (SUTVA)."""
    never = panel.loc[panel.treated_unit == 0, "market_id"].unique()
    exposure = panel[panel.market_id.isin(never)].groupby("market_id")["neighbor_adoption"].max()
    adj_primary = set(adjacency.loc[adjacency.market_id == primary, "neighbor_id"])
    dropped = sorted(m for m in never if m in adj_primary or exposure[m] >= exposure_cap)
    kept = sorted(m for m in never if m not in dropped)
    return kept, dropped


def build_predictors(panel: pd.DataFrame, units: list[str], t_lo: int, t_hi: int,
                     n_blocks: int = 4, Y: pd.DataFrame | None = None) -> pd.DataFrame:
    """K x units predictor matrix built ONLY from weeks [t_lo, t_hi) (no look-ahead),
    z-scored across units so V is interpretable."""
    Y = (outcome_matrix(panel) if Y is None else Y)[units]
    G = outcome_matrix(panel, "comp_gap")[units]
    blocks = np.array_split(np.arange(t_lo, t_hi), n_blocks)
    rows = {f"log_rev_block{k + 1}": Y.loc[b].mean() for k, b in enumerate(blocks)}
    rows["comp_gap_mean"] = G.loc[t_lo:t_hi - 1].mean()
    static = panel.drop_duplicates("market_id").set_index("market_id").loc[units, STATIC]
    for c in STATIC:
        rows[c] = static[c]
    X = pd.DataFrame(rows).T
    return X.sub(X.mean(1), axis=0).div(X.std(1).replace(0, 1), axis=0)


class _WeightSolver:
    """Reusable DPP program: min mean((y1 - Y0 w)^2) + lam*||w||^2, w in simplex."""

    def __init__(self, n: int, j: int):
        self.y1, self.Y0 = cp.Parameter(n), cp.Parameter((n, j))
        self.lam = cp.Parameter(nonneg=True)
        self.w = cp.Variable(j, nonneg=True)
        obj = cp.sum_squares(self.y1 - self.Y0 @ self.w) / n + self.lam * cp.sum_squares(self.w)
        self.prob = cp.Problem(cp.Minimize(obj), [cp.sum(self.w) == 1])

    def solve(self, y1: np.ndarray, Y0: np.ndarray, lam: float) -> np.ndarray:
        self.y1.value, self.Y0.value, self.lam.value = y1, Y0, lam
        self.prob.solve(solver=cp.CLARABEL)
        w = np.clip(self.w.value, 0, None)
        return w / w.sum()


_SOLVERS: dict = {}


def solve_weights(y1: np.ndarray, Y0: np.ndarray, lam: float) -> np.ndarray:
    key = Y0.shape
    if key not in _SOLVERS:
        _SOLVERS[key] = _WeightSolver(*key)
    return _SOLVERS[key].solve(y1, Y0, lam)


def rmspe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


LAMBDA_GRID = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


# --------------------------------------------------------------------------
# main estimator
# --------------------------------------------------------------------------
@dataclass
class SCMResult:
    treated: str
    t0: int
    weights: pd.Series
    lam: float
    actual_log: pd.Series
    synth_log: pd.Series
    rmspe_val: float          # out-of-sample (blocked) pre-treatment fit
    rmspe_pre: float          # full pre-period fit
    rmspe_post: float
    jensen: float = 0.0
    donors: list[str] = field(default_factory=list)
    val_curve: dict = field(default_factory=dict)

    @property
    def gap(self) -> pd.Series:                      # log-points
        return self.actual_log - self.synth_log

    @property
    def ratio(self) -> float:                        # post/pre RMSPE (Abadie et al.)
        return self.rmspe_post / self.rmspe_pre

    @property
    def att_log(self) -> float:
        return float(self.gap.loc[self.t0:].mean())

    def revenue_paths(self) -> pd.DataFrame:
        actual = np.exp(self.actual_log)
        cf = np.exp(self.synth_log + self.jensen)
        return pd.DataFrame({"actual": actual, "counterfactual": cf, "incremental": actual - cf})

    @property
    def att_pct(self) -> float:
        p = self.revenue_paths().loc[self.t0:]
        return float(p.actual.sum() / p.counterfactual.sum() - 1)

    @property
    def cumulative_incremental(self) -> float:
        return float(self.revenue_paths().loc[self.t0:, "incremental"].sum())


def fit_scm(panel: pd.DataFrame, treated: str, donors: list[str], t0: int, train_end: int,
            lam: float | None = None) -> SCMResult:
    Yraw = outcome_matrix(panel)
    # stage 1: choose ridge strength by blocked validation (fit weeks < train_end, score on [train_end, t0))
    Yd = Yraw - Yraw.iloc[:train_end].mean()
    y_tr, Y_tr = Yd[treated].to_numpy(), Yd[donors].to_numpy()
    val = slice(train_end, t0)
    curve = {}
    for g in LAMBDA_GRID:
        w = solve_weights(y_tr[:train_end], Y_tr[:train_end], g)
        curve[g] = rmspe(y_tr[val], Y_tr[val] @ w)
    if lam is None:
        best = min(curve.values())
        lam = max(g for g, v in curve.items() if v <= best * 1.005)     # prefer more shrinkage on near-ties
    rm_val = curve.get(lam, np.nan) if lam in curve else rmspe(
        y_tr[val], Y_tr[val] @ solve_weights(y_tr[:train_end], Y_tr[:train_end], lam))

    # stage 2: final weights on the full pre-period, intercept-adjusted
    mu = Yraw.iloc[:t0].mean()
    Yf = Yraw - mu
    y1, Y0 = Yf[treated].to_numpy(), Yf[donors].to_numpy()
    w = solve_weights(y1[:t0], Y0[:t0], lam)
    synth = Y0 @ w + float(mu[treated])
    actual = y1 + float(mu[treated])
    resid = (actual - synth)[:t0]
    return SCMResult(
        treated=treated, t0=t0, weights=pd.Series(w, index=donors).sort_values(ascending=False),
        lam=float(lam), actual_log=pd.Series(actual, index=Yraw.index),
        synth_log=pd.Series(synth, index=Yraw.index), rmspe_val=rm_val,
        rmspe_pre=rmspe(actual[:t0], synth[:t0]), rmspe_post=rmspe(actual[t0:], synth[t0:]),
        jensen=0.5 * float(np.var(resid)), donors=donors, val_curve=curve)


def balance_table(panel: pd.DataFrame, res: SCMResult) -> pd.DataFrame:
    """Covariate balance: treated vs synthetic vs simple donor average (never used for fitting)."""
    st = panel.drop_duplicates("market_id").set_index("market_id")
    cg = panel[panel.week_idx < res.t0].groupby("market_id")["comp_gap"].mean()
    st = st.assign(comp_gap_pre=cg)
    cols = STATIC + ["comp_gap_pre"]
    w = res.weights
    return pd.DataFrame({
        "treated": st.loc[res.treated, cols],
        "synthetic": (st.loc[w.index, cols].mul(w, axis=0)).sum(),
        "donor_avg": st.loc[res.donors, cols].mean(),
    })


# --------------------------------------------------------------------------
# inference & robustness
# --------------------------------------------------------------------------
def placebo_in_space(panel: pd.DataFrame, base: SCMResult, train_end: int,
                     max_pre_ratio: float = 2.0) -> pd.DataFrame:
    """Permutation inference: re-assign 'treatment' to every donor (lambda held fixed)."""
    rows, gaps = [], {}
    for j in base.donors:
        pool = [d for d in base.donors if d != j]
        r = fit_scm(panel, j, pool, base.t0, train_end, lam=base.lam)
        rows.append({"market_id": j, "rmspe_pre": r.rmspe_pre, "rmspe_post": r.rmspe_post,
                     "ratio": r.ratio, "att_log": r.att_log})
        gaps[j] = r.gap
    tab = pd.DataFrame(rows)
    tab.attrs["gaps"] = pd.DataFrame(gaps)
    keep = tab[tab.rmspe_pre <= max_pre_ratio * base.rmspe_pre]
    tab.attrs["p_ratio"] = float((1 + (keep.ratio >= base.ratio).sum()) / (1 + len(keep)))
    tab.attrs["p_att_two_sided"] = float(
        (1 + (keep.att_log.abs() >= abs(base.att_log)).sum()) / (1 + len(keep)))
    tab.attrs["n_kept"] = len(keep)
    return tab


def placebo_in_time(panel: pd.DataFrame, base: SCMResult, shifts=(6, 10, 14)) -> pd.DataFrame:
    """Pretend adoption happened `s` weeks earlier; there should be no 'effect'."""
    rows = []
    for s in shifts:
        fake = base.t0 - s
        r = fit_scm(panel, base.treated, base.donors, fake, train_end=int(fake * 0.7), lam=base.lam)
        win = slice(fake, base.t0)
        rows.append({"shift_weeks": s, "fake_t0": fake,
                     "placebo_att_log": float(r.gap.loc[win].mean()),
                     "true_att_log": base.att_log})
    return pd.DataFrame(rows)


def leave_one_donor_out(panel: pd.DataFrame, base: SCMResult, train_end: int, top: int = 5) -> pd.DataFrame:
    rows = []
    for d in base.weights.index[:top]:
        pool = [x for x in base.donors if x != d]
        r = fit_scm(panel, base.treated, pool, base.t0, train_end, lam=base.lam)
        rows.append({"dropped_donor": d, "weight_dropped": base.weights[d],
                     "att_log": r.att_log, "att_pct": r.att_pct, "rmspe_pre": r.rmspe_pre})
    return pd.DataFrame(rows)


def rolling_origin_cv(panel: pd.DataFrame, treated: str, donors: list[str], t0: int, lam: float,
                      cutoffs=(36, 44, 52, 60), horizon: int = 8) -> pd.DataFrame:
    """Time-series cross-validation of the SC predictive fit: fit on weeks < c, forecast
    [c, c+h) - no future information leaks into the weights."""
    Y = outcome_matrix(panel)
    rows = []
    for c in cutoffs:
        if c + horizon > t0:
            continue
        Yd = Y - Y.iloc[:c].mean()
        w = solve_weights(Yd[treated].to_numpy()[:c], Yd[donors].to_numpy()[:c], lam)
        yhat = Yd[donors].to_numpy()[c:c + horizon] @ w
        rows.append({"cutoff_week": c, "horizon": horizon,
                     "oos_rmspe": rmspe(Yd[treated].to_numpy()[c:c + horizon], yhat)})
    return pd.DataFrame(rows)
