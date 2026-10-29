# %% [markdown]
# # Week 4 · Day 1 — Visual Diagnostics: Actual Revenue vs Synthetic Counterfactual
# **Goal:** show product managers, in one picture, what revenue actually did against what it would have done
# without dynamic pricing — and export the artefacts the dashboard reads (Day 3).
#
# **Inputs (from Weeks 1-3):** `data/processed/panel.csv`, `data/raw/pricing_events.csv`, `results/w2_scm_revenue_paths.csv`,
# `results/w2_scm_weights.csv`, `results/w2_mspe_table.csv`, `results/w2_dml_results.csv`, `results/w3_*.csv`
# **Outputs:** `artifacts/scm_revenue_paths.csv`, `artifacts/placebo_gaps.csv`, `artifacts/scm_weights.csv`,
# `artifacts/cate_by_segment.csv`, `artifacts/summary.json`, `results/w4_dashboard_summary.csv`,
# `reports/figures/w4_actual_vs_counterfactual.png`

# %%
import _bootstrap  # noqa: F401
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from cpe.config import CFG
from cpe.utils import setup_style, savefig

setup_style(); CFG.ensure_dirs()
R, A = CFG.results_dir, CFG.artifact_dir
panel = pd.read_csv(CFG.processed_dir / "panel.csv", parse_dates=["week_start"])
events = pd.read_csv(CFG.raw_dir / "pricing_events.csv")
T0 = CFG.primary_adoption_week
primary = events.loc[events.go_live_week_idx == T0, "market_id"].iloc[0]
paths = pd.read_csv(R / "w2_scm_revenue_paths.csv", parse_dates=["week_start"])
print(f"Focal market {primary}; go-live week {T0}; {len(paths)} weeks of actual vs counterfactual revenue")

# %% [markdown]
# ## 1. The headline picture

# %%
post = paths[paths.week_idx >= T0]
inc_total = post.incremental.sum()
lift = post.actual.sum() / post.counterfactual.sum() - 1
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
ax[0].plot(paths.week_start, paths.actual, color="#d62728", lw=2, label="actual revenue")
ax[0].plot(paths.week_start, paths.counterfactual, "--", color="#1f77b4", lw=2, label="synthetic counterfactual (no dynamic pricing)")
ax[0].fill_between(post.week_start, post.counterfactual, post.actual, color="#2ca02c", alpha=0.25,
                   label=f"incremental revenue: {inc_total:,.0f}")
ax[0].axvline(paths.week_start[paths.week_idx == T0].iloc[0], color="k", ls=":"); ax[0].legend(loc="upper left")
ax[0].set_title(f"{primary}: actual vs counterfactual weekly revenue | lift = {lift:+.1%}")
ax[1].bar(paths.week_start, paths.incremental, color=np.where(paths.incremental >= 0, "#2ca02c", "#d62728"), width=5)
ax[1].axvline(paths.week_start[paths.week_idx == T0].iloc[0], color="k", ls=":")
ax[1].set_title("Weekly incremental revenue (actual - counterfactual)")
plt.tight_layout(); savefig(fig, "w4_actual_vs_counterfactual.png"); plt.show()

# %% [markdown]
# ## 2. Export dashboard artefacts
# The dashboard never refits a model; it only reads these files.

# %%
gaps = pd.read_csv(R / "w3_placebo_gaps.csv")
weights = pd.read_csv(R / "w2_scm_weights.csv"); weights = weights[weights.weight > 0.005]
seg = pd.read_csv(R / "w3_cate_by_segment.csv")
mspe = pd.read_csv(R / "w2_mspe_table.csv").set_index("window")["RMSPE"]
dml = pd.read_csv(R / "w2_dml_results.csv").iloc[0]
pls = pd.read_csv(R / "w3_placebo_summary.csv").set_index("metric")["value"]
rv = pd.read_csv(R / "w3_robustness_value.csv").set_index("metric")["value"]
ate = pd.read_csv(R / "w3_ate_table.csv")

se = (dml.ci_high - dml.ci_low) / (2 * 1.96)
tr = panel[panel.dynamic_pricing == 1]
delta_obs = float((tr.avg_price / tr.base_price - 1).mean())        # observed average price uplift
summary = {
    "primary": primary, "t0": T0, "delta_obs": delta_obs, "post_weeks": int(CFG.n_weeks - T0),
    "att_pct_scm": float(lift), "att_log_scm": float(np.log1p(lift)), "cumulative_incremental": float(inc_total),
    "p_att": float(pls["permutation p-value (|gap|)"]), "p_ratio": float(pls["permutation p-value (RMSPE ratio)"]),
    "rmspe_pre": float(mspe["pre-treatment (in-sample)"]), "rmspe_val": float(mspe["validation (blocked out-of-sample)"]),
    "ratio": float(mspe["post-treatment"] / mspe["pre-treatment (in-sample)"]),
    "dml_theta": float(dml.theta_log_points), "dml_ci": [float(dml.ci_low), float(dml.ci_high)],
    "dml_p": float(2 * (1 - stats.norm.cdf(abs(dml.theta_log_points / se)))),
    "bjs_theta": float(ate.loc[ate.estimator.str.startswith("Imputation"), "estimate_log_pts"].iloc[0]),
    "rv": float(rv["RV: confounder strength to erase the estimate"]),
    "rv_alpha": float(rv["RV (5%): strength to make it insignificant"])}

paths.drop(columns=["gap_log"], errors="ignore").to_csv(A / "scm_revenue_paths.csv", index=False)
gaps.to_csv(A / "placebo_gaps.csv", index=False)
weights.to_csv(A / "scm_weights.csv", index=False)
seg.to_csv(A / "cate_by_segment.csv", index=False)
(A / "summary.json").write_text(json.dumps(summary, indent=2))
pd.Series(summary).astype(str).rename("value").rename_axis("metric").to_csv(R / "w4_dashboard_summary.csv")
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}, indent=1))
print("\nFiles in artifacts/:", sorted(p.name for p in A.iterdir()))
