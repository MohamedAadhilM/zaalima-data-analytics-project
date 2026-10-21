"""Double / Debiased Machine Learning (Chernozhukov et al., 2018) for panel pricing data.

Estimators provided (all return log-revenue effects of dynamic pricing):
  * `dml_plr`            - partially-linear DML, own implementation: market-grouped cross-fitting,
                           LightGBM nuisances, two-way (market, week) demeaning, market-clustered SE
  * `econml_linear_dml`  - EconML LinearDML cross-check
  * `causal_forest`      - EconML CausalForestDML for heterogeneous effects (CATE)
  * `bjs_imputation`     - heterogeneity-robust imputation estimator (Borusyak-Jaravel-Spiess)
                           - the safeguard against staggered-adoption bias in two-way FE
  * `twfe`, `naive_*`    - benchmarks that show *why* the advanced machinery is needed
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


def causal_forest(prep: dict, n_folds: int = 5, seed: int = 0, n_estimators: int = 300):
    """CausalForestDML: effect modifiers X = HET_VARS, remaining features as controls W."""
    from econml.dml import CausalForestDML
    Wcols = [c for c in prep["W"].columns if c not in HET_VARS]
    est = CausalForestDML(model_y=_lgbm(seed), model_t=_lgbm(seed + 1), discrete_treatment=False,
                          n_estimators=n_estimators, min_samples_leaf=40, max_depth=6,
                          cv=GroupKFold(n_folds), random_state=seed)
    est.fit(prep["Y"], prep["D"], X=prep["W"][HET_VARS].to_numpy(),
            W=prep["W"][Wcols].to_numpy(), groups=prep["groups"])
    return est


def cate_table(est, prep: dict, truth: pd.DataFrame) -> pd.DataFrame:
    """Unit-level CATE (post-adoption observations only) vs the simulation's true effect."""
    d = prep["df"]
    post = d["dynamic_pricing"].to_numpy() == 1
    X = prep["W"].loc[post, HET_VARS].to_numpy()
    cate = est.effect(X).ravel()
    lo, hi = est.effect_interval(X, alpha=0.05)
    out = d.loc[post, ["market_id", "tier", "income_z", "urban_index"]].copy()
    out["cate"], out["lo"], out["hi"] = cate, np.ravel(lo), np.ravel(hi)
    return out.merge(truth[["market_id", "true_tau_log"]], on="market_id")


# --------------------------------------------------------------------------
# heterogeneity-robust imputation estimator
# --------------------------------------------------------------------------
def bjs_imputation(panel: pd.DataFrame) -> dict:
    """Fit Y(0) = market FE + week FE + controls on *untreated* observations only,
    impute Y(0) for treated observations, average Y - Y0hat. Immune to the negative-weight
    problem of TWFE under staggered adoption and heterogeneous effects."""
    d = panel.sort_values(["market_id", "week_idx"]).reset_index(drop=True)
    X = pd.get_dummies(d[["market_id", "week_idx"]].astype(str), drop_first=True).astype(float)
    for c in ("comp_gap", "comp_gap_4w", "neighbor_adoption"):
        X[c] = d[c]
    X.insert(0, "const", 1.0)
    unt = d["dynamic_pricing"].to_numpy() == 0
    beta, *_ = np.linalg.lstsq(X.to_numpy()[unt], d["log_revenue"].to_numpy()[unt], rcond=None)
    y0 = X.to_numpy() @ beta
    d["tau_hat"] = d["log_revenue"] - y0
    tr = d[d.dynamic_pricing == 1]
    by_mkt = tr.groupby("market_id")["tau_hat"].mean()
    att = float(tr["tau_hat"].mean())
    se = float(by_mkt.std(ddof=1) / np.sqrt(len(by_mkt)))     # between-market SE (approximate)
    return {"theta": att, "se": se, "ci": (att - 1.96 * se, att + 1.96 * se), "by_market": by_mkt,
            "unit_effects": tr[["market_id", "week_idx", "tau_hat"]]}


def twfe(prep: dict) -> dict:
    import statsmodels.api as sm
    X = sm.add_constant(np.column_stack([prep["D"]]))
    m = sm.OLS(prep["Y"], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(prep["groups"])[0]})
    lo, hi = m.conf_int()[1]
    return {"theta": float(m.params[1]), "se": float(m.bse[1]), "ci": (float(lo), float(hi))}


def naive_estimates(panel: pd.DataFrame) -> dict:
    tr = panel[panel.treated_unit == 1]
    post = tr[tr.dynamic_pricing == 1]
    never_post = panel[(panel.treated_unit == 0) & (panel.week_idx >= post.week_idx.min())]
    pre_tr = tr[tr.dynamic_pricing == 0]
    return {
        "naive_treated_vs_never_post": float(post.log_revenue.mean() - never_post.log_revenue.mean()),
        "naive_before_after": float(post.log_revenue.mean() - pre_tr.log_revenue.mean()),
    }


def true_att(panel: pd.DataFrame, truth: pd.DataFrame) -> float:
    m = panel[panel.dynamic_pricing == 1].merge(truth[["market_id", "true_tau_log"]], on="market_id")
    return float(m["true_tau_log"].mean())
