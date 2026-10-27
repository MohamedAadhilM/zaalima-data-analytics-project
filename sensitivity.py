"""Robustness & sensitivity analysis for the causal estimates.

* `robustness_value`     - Cinelli & Hazlett (2020) robustness value: how strong (partial R^2 with
                           BOTH treatment and outcome) would an unobserved confounder have to be
                           to explain away the DML estimate?
* `omitted_variable_bias_grid` - bias contours for hypothetical confounder strengths
* `dowhy_refutations`    - DoWhy refuters (placebo treatment, random common cause, data subset,
                           unobserved common cause) on the identified back-door estimand
* `grouped_cv_report`    - market-grouped cross-validation of nuisance models and an R-loss
                           comparison of candidate CATE models
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

from . import dml as D


def robustness_value(theta: float, se: float, dof: int, q: float = 1.0, alpha: float = 0.05) -> dict:
    """RV_q: partial R^2 an unobserved confounder needs with treatment AND outcome to reduce the
    estimate by 100q%. RV_q,alpha: same, to make it statistically indistinguishable from zero."""
    t = abs(theta / se)
    f = q * t / np.sqrt(dof)
    rv = 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)
    f_alpha = stats.t.ppf(1 - alpha / 2, dof - 1) / np.sqrt(dof - 1)
    fa = f - f_alpha
    rv_a = 0.5 * (np.sqrt(fa ** 4 + 4 * fa ** 2) - fa ** 2) if fa > 0 else 0.0
    return {"t_stat": t, "partial_r2_treat_outcome": t ** 2 / (t ** 2 + dof), "RV_q": float(rv), "RV_q_alpha": float(rv_a)}


def omitted_variable_bias_grid(theta: float, se: float, dof: int,
                               strengths=(0.01, 0.02, 0.05, 0.10)) -> pd.DataFrame:
    """Adjusted estimate if a confounder explains r2_y of outcome and r2_d of treatment residual
    variance (worst-case sign). bias = se * sqrt(dof * r2y * r2d / (1 - r2d))."""
    rows = []
    for ry in strengths:
        for rd in strengths:
            bias = se * np.sqrt(dof * ry * rd / (1 - rd))
            rows.append({"R2_confounder_outcome": ry, "R2_confounder_treatment": rd,
                         "adjusted_theta": theta - np.sign(theta) * bias})
    return pd.DataFrame(rows).pivot(index="R2_confounder_outcome", columns="R2_confounder_treatment",
                                    values="adjusted_theta")


def dowhy_refutations(panel: pd.DataFrame, n_sim: int = 15, seed: int = 0) -> pd.DataFrame:
    from dowhy import CausalModel
    from . import dag
    g = dag.build_dag()
    cols = [n for n in g.nodes if n not in dag.UNOBSERVED]
    df = panel.sort_values(["market_id", "week_idx"]).reset_index(drop=True)
    # work on the two-way within-transformed outcome/treatment so market & week FE are respected
    df[dag.OUTCOME] = D.two_way_demean(df, dag.OUTCOME).to_numpy()
    df[dag.TREATMENT] = D.two_way_demean(df, dag.TREATMENT).to_numpy()
    df = df[cols]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CausalModel(data=df, treatment=dag.TREATMENT, outcome=dag.OUTCOME,
                            graph=dag.to_gml(g))
        est = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(est, method_name="backdoor.linear_regression")
        rows = [{"refuter": "baseline (linear back-door)", "original": float(estimate.value),
                 "new_effect": float(estimate.value), "p_value": np.nan}]
        specs = [
            ("placebo_treatment_refuter", dict(placebo_type="permute", num_simulations=n_sim)),
            ("random_common_cause", dict(num_simulations=n_sim)),
            ("data_subset_refuter", dict(subset_fraction=0.8, num_simulations=n_sim)),
            ("add_unobserved_common_cause", dict(confounders_effect_on_treatment="linear",
                                                 confounders_effect_on_outcome="linear",
                                                 effect_strength_on_treatment=0.02,
                                                 effect_strength_on_outcome=0.02)),
        ]
        for name, kw in specs:
            try:
                r = model.refute_estimate(est, estimate, method_name=name, random_seed=seed, **kw)
                p = getattr(r, "refutation_result", None)
                pv = p.get("p_value") if isinstance(p, dict) else np.nan
                rows.append({"refuter": name, "original": float(r.estimated_effect),
                             "new_effect": float(r.new_effect), "p_value": pv})
            except Exception as exc:                                 # keep the pipeline alive
                rows.append({"refuter": name, "original": np.nan, "new_effect": np.nan,
                             "p_value": np.nan, "note": type(exc).__name__})
    return pd.DataFrame(rows)


def grouped_cv_report(prep: dict, n_folds: int = 5, seed: int = 0) -> dict:
    """(a) fold-by-fold nuisance R^2 and effect stability; (b) out-of-fold R-loss of CATE models."""
    W, Y, Dd, grp = prep["W"].to_numpy(), prep["Y"], prep["D"], prep["groups"]
    het = prep["W"][D.HET_VARS].to_numpy()
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_folds).split(W, Y, grp)):
        gm = D._lgbm(seed).fit(W[tr], Y[tr]); mm = D._lgbm(seed + 1).fit(W[tr], Dd[tr])
        Yr, Dr = Y[te] - gm.predict(W[te]), Dd[te] - mm.predict(W[te])
        theta_k = float((Dr * Yr).sum() / (Dr ** 2).sum())
        # CATE candidates fitted on the training folds' residualised data (R-learner style)
        Yr_tr, Dr_tr = Y[tr] - gm.predict(W[tr]), Dd[tr] - mm.predict(W[tr])
        const = float((Dr_tr * Yr_tr).sum() / (Dr_tr ** 2).sum())
        A = np.column_stack([np.ones(len(tr)), het[tr]]) * Dr_tr[:, None]
        coef = np.linalg.lstsq(A, Yr_tr, rcond=None)[0]
        lin = np.column_stack([np.ones(len(te)), het[te]]) @ coef
        rloss = lambda tau: float(np.mean((Yr - tau * Dr) ** 2))
        folds.append({"fold": k, "r2_outcome": float(1 - np.mean(Yr ** 2) / np.var(Y[te])),
                      "theta_fold": theta_k, "rloss_zero": rloss(0.0),
                      "rloss_constant": rloss(const), "rloss_linear_cate": rloss(lin)})
    return {"folds": pd.DataFrame(folds)}
