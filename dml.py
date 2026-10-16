"""Double / Debiased Machine Learning (Chernozhukov et al., 2018) for panel pricing data.

Week 2 Day 4 version:
  * `build_features`     - wide, treatment-unaffected control set (high-dimensional W)
  * `dml_plr`            - partially-linear DML, own implementation: market-grouped cross-fitting,
                           LightGBM nuisances, two-way (market, week) demeaning, market-clustered SE
  * `econml_linear_dml`  - EconML LinearDML cross-check
(Causal Forest CATE, imputation estimator and benchmarks arrive in Week 3.)
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

HET_VARS = ["income_z", "urban_index", "log_pop", "comp_intensity"]   # effect modifiers (X)


# --------------------------------------------------------------------------
# feature engineering - a deliberately wide control set (high-dimensional W)
# --------------------------------------------------------------------------
def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Controls that are NOT affected by the treatment (no post-treatment outcome lags)."""
    d = panel.sort_values(["market_id", "week_idx"]).copy()
    g = d.groupby("market_id")["comp_gap"]
    F = pd.DataFrame(index=d.index)
    F["comp_gap"] = d["comp_gap"]
    for k in (1, 2, 4):
        F[f"comp_gap_lag{k}"] = g.shift(k).bfill()
    F["comp_gap_roll4"] = g.transform(lambda s: s.rolling(4, min_periods=1).mean())
    F["comp_gap_roll8"] = g.transform(lambda s: s.rolling(8, min_periods=1).mean())
    F["comp_gap_mom"] = F["comp_gap_roll4"] - F["comp_gap_roll8"]
    F["comp_gap_dev"] = d["comp_gap"] - g.transform("mean")
    for k in (1, 2, 3):
        F[f"sin{k}"] = np.sin(2 * np.pi * k * d["week_idx"] / 52)
        F[f"cos{k}"] = np.cos(2 * np.pi * k * d["week_idx"] / 52)
    for c in HET_VARS + ["neighbor_adoption"]:
        F[c] = d[c]
    F["tier_A"] = (d["tier"] == "A").astype(float)
    F["tier_B"] = (d["tier"] == "B").astype(float)
    F["urban_x_sin1"] = d["urban_index"] * F["sin1"]
    F["income_x_sin1"] = d["income_z"] * F["sin1"]
    F["gap_x_intensity"] = d["comp_gap"] * d["comp_intensity"]
    F["gap_x_urban"] = d["comp_gap"] * d["urban_index"]
    F["trend"] = d["week_idx"] / d["week_idx"].max()
    return F


def two_way_demean(df: pd.DataFrame, col: str) -> pd.Series:
    """Balanced-panel within transformation (market and week fixed effects)."""
    p = df.pivot(index="market_id", columns="week_idx", values=col)
    r = p.sub(p.mean(axis=1), axis=0).sub(p.mean(axis=0), axis=1) + p.to_numpy().mean()
    return df.apply(lambda row: r.loc[row.market_id, row.week_idx], axis=1)


def _lgbm(seed=0, **kw):
    return lgb.LGBMRegressor(n_estimators=120, learning_rate=0.03, num_leaves=7,
                             min_child_samples=60, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                             verbose=-1, **kw)


# --------------------------------------------------------------------------
# DML
# --------------------------------------------------------------------------
def prepare(panel: pd.DataFrame) -> dict:
    d = panel.sort_values(["market_id", "week_idx"]).reset_index(drop=True)
    F = build_features(d).reset_index(drop=True)
    return {"df": d, "W": F, "Y": two_way_demean(d, "log_revenue").to_numpy(),
            "D": two_way_demean(d, "dynamic_pricing").to_numpy(),
            "groups": d["market_id"].to_numpy()}


def cross_fit(prep: dict, n_folds: int = 5, seed: int = 0) -> dict:
    """Out-of-fold nuisance predictions, folds grouped by market (respects serial dependence)."""
    W, Y, D, grp = prep["W"].to_numpy(), prep["Y"], prep["D"], prep["groups"]
    ghat, mhat = np.zeros_like(Y), np.zeros_like(D)
    for tr, te in GroupKFold(n_folds).split(W, Y, grp):
        ghat[te] = _lgbm(seed).fit(W[tr], Y[tr]).predict(W[te])
        mhat[te] = _lgbm(seed + 1).fit(W[tr], D[tr]).predict(W[te])
    r2 = lambda a, b: float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))
    return {"Yres": Y - ghat, "Dres": D - mhat, "r2_outcome": r2(Y, ghat), "r2_treatment": r2(D, mhat)}


def _cluster_se(psi_num: np.ndarray, denom: float, groups: np.ndarray) -> float:
    s = pd.Series(psi_num).groupby(groups).sum().to_numpy()
    return float(np.sqrt((s ** 2).sum()) / denom)


def dml_plr(prep: dict, n_folds: int = 5, seed: int = 0, alpha: float = 0.05) -> dict:
    cf = cross_fit(prep, n_folds, seed)
    Yr, Dr, g = cf["Yres"], cf["Dres"], prep["groups"]
    denom = float((Dr ** 2).sum())
    theta = float((Dr * Yr).sum() / denom)
    se = _cluster_se(Dr * (Yr - theta * Dr), denom, g)
    z = stats.norm.ppf(1 - alpha / 2)
    t = theta / se
    dof = int(len(Dr) - prep["W"].shape[1])
    return {"theta": theta, "se": se, "ci": (theta - z * se, theta + z * se), "t": t,
            "p": float(2 * (1 - stats.norm.cdf(abs(t)))), "dof": dof, **cf}


def econml_linear_dml(prep: dict, n_folds: int = 5, seed: int = 0) -> dict:
    from econml.dml import LinearDML
    est = LinearDML(model_y=_lgbm(seed), model_t=_lgbm(seed + 1), discrete_treatment=False,
                    cv=GroupKFold(n_folds), random_state=seed)
    est.fit(prep["Y"], prep["D"], X=None, W=prep["W"].to_numpy(), groups=prep["groups"])
    lo, hi = est.ate_interval(alpha=0.05)
    return {"theta": float(np.ravel(est.ate())[0]), "ci": (float(np.ravel(lo)[0]), float(np.ravel(hi)[0]))}
