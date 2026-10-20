# %% [markdown]
# # Week 2 · Day 5 — Mid-Project Review (End of Week 2)
# **Goal:** check the three mid-project criteria from the project plan against the evidence produced in
# Days 1-4, and summarise where the project stands.
#
# 1. Causal DAGs mapped and validated
# 2. Synthetic control baselines successfully matching historical treated trajectories
# 3. Initial treatment-effect estimators functioning
#
# **Inputs (all produced earlier):** `results/w1_confounders.csv`, `results/w1_estimand.txt`,
# `results/w2_donor_pool.csv`, `results/w2_scm_fit_summary.csv`, `results/w2_mspe_table.csv`,
# `results/w2_scm_weights.csv`, `results/w2_scm_revenue_paths.csv`, `results/w2_dml_results.csv`,
# `results/w2_dml_nuisance_diagnostics.csv`
# Pass thresholds below are our own reasonable cut-offs (the project plan states the criteria, not the numbers).
# Needs `tabulate` for the Markdown table: `pip install tabulate`.
# **Outputs:** `results/w2_midproject_checklist.csv`, `reports/w2_midproject_review.md`,
# `reports/figures/w2_midproject_summary.png`

# %%
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpe.config import CFG
from cpe.utils import setup_style, savefig

setup_style(); CFG.ensure_dirs()
R = CFG.results_dir
conf = pd.read_csv(R / "w1_confounders.csv")
estimand = (R / "w1_estimand.txt").read_text()
pool = pd.read_csv(R / "w2_donor_pool.csv")
fit = pd.read_csv(R / "w2_scm_fit_summary.csv").set_index("metric")["value"]
mspe = pd.read_csv(R / "w2_mspe_table.csv").set_index("window")
weights = pd.read_csv(R / "w2_scm_weights.csv")
paths = pd.read_csv(R / "w2_scm_revenue_paths.csv")
dml = pd.read_csv(R / "w2_dml_results.csv")
nuis = pd.read_csv(R / "w2_dml_nuisance_diagnostics.csv").set_index("model")["out_of_fold_R2"]
T0 = CFG.primary_adoption_week

# %% [markdown]
# ## 1. Evaluate each criterion against an explicit pass rule

# %%
pre = mspe.loc["pre-treatment (in-sample)", "RMSPE"]
val = mspe.loc["validation (blocked out-of-sample)", "RMSPE"]
post = mspe.loc["post-treatment", "RMSPE"]
val_gap = paths.loc[(paths.week_idx >= CFG.train_end) & (paths.week_idx < T0), "gap_log"].mean()   # held-out weeks
post_gap = paths.loc[paths.week_idx >= T0, "gap_log"].mean()
dml_row = dml.iloc[0]
n_spread = int((weights.weight > 0.01).sum())

checks = [
    ("1. Causal DAG mapped and validated",
     "Estimand identified by DoWhy; confounders listed; mediator excluded from adjustment",
     bool(len(conf) > 0 and "Estimand type" in estimand and (conf.role.str.contains("mediator")).any()),
     f"{(conf.role.str.startswith('confounder')).sum()} confounder rows, "
     f"{(conf.role.str.contains('mediator')).sum()} mediator; back-door estimand identified"),
    ("2a. Synthetic control matches pre-treatment path (in-sample)",
     "pre-treatment RMSPE <= 0.05 (≈5% error)", bool(pre <= 0.05), f"RMSPE {pre:.4f}"),
    ("2b. Synthetic control matches on weeks it never saw",
     "validation RMSPE <= 0.05", bool(val <= 0.05), f"RMSPE {val:.4f}"),
    ("2c. No drift on held-out pre-treatment weeks",
     "average gap over the validation window within ±0.01 log-points (the full-pre average is zero by construction, so it is not used)",
     bool(abs(val_gap) <= 0.01), f"validation-window average gap {val_gap:+.4f}"),
    ("2d. Weights are spread across donors (not one-donor reliance)",
     ">= 5 donors with weight > 1%", bool(n_spread >= 5), f"{n_spread} donors > 1%"),
    ("3a. SCM effect estimator functioning",
     "finite post-treatment gap with the expected sign (positive)", bool(np.isfinite(post_gap) and post_gap > 0),
     f"gap {post_gap:+.4f} log-pts ({100 * np.expm1(post_gap):+.1f}%)"),
    ("3b. DML effect estimator functioning",
     "finite estimate, 95% interval excludes zero", bool(dml_row.ci_low > 0 or dml_row.ci_high < 0),
     f"θ {dml_row.theta_log_points:+.4f} (CI {dml_row.ci_low:+.3f} to {dml_row.ci_high:+.3f})"),
    ("3c. DML has usable overlap",
     "treatment-model out-of-fold R² < 0.9 (adoption not perfectly predictable)",
     bool(nuis["treatment model E[D|W]"] < 0.9), f"R² {nuis['treatment model E[D|W]']:.2f}"),
    ("3d. The two estimators agree on direction",
     "SCM gap and DML θ share the same sign", bool(np.sign(post_gap) == np.sign(dml_row.theta_log_points)),
     f"SCM {post_gap:+.3f} vs DML {dml_row.theta_log_points:+.3f}"),
]
chk = pd.DataFrame(checks, columns=["criterion", "pass_rule", "passed", "evidence"])
chk["status"] = np.where(chk.passed, "PASS", "REVIEW")
chk.to_csv(R / "w2_midproject_checklist.csv", index=False)
print(chk[["criterion", "status", "evidence"]].to_string(index=False))
print(f"\n{int(chk.passed.sum())} of {len(chk)} checks passed")

# %% [markdown]
# ## 2. One-page picture

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 3.9))
ax[0].plot(paths.week_idx, paths.actual, color="#d62728", label="actual")
ax[0].plot(paths.week_idx, paths.counterfactual, "--", color="#1f77b4", label="synthetic counterfactual")
ax[0].axvline(T0, color="k", ls=":"); ax[0].legend(); ax[0].set_title("Focal market: actual vs synthetic revenue")
lab = ["SCM (one market)", "DML (all adopters)"]
est = [100 * np.expm1(post_gap), dml_row["revenue_lift_%"]]
err_lo = [0, dml_row["revenue_lift_%"] - 100 * np.expm1(dml_row.ci_low)]
err_hi = [0, 100 * np.expm1(dml_row.ci_high) - dml_row["revenue_lift_%"]]
ax[1].bar(lab, est, color=["#1f77b4", "#2ca02c"]); ax[1].errorbar(lab, est, yerr=[err_lo, err_hi], fmt="none", c="k", capsize=5)
ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylabel("revenue lift %")
ax[1].set_title("Initial effect estimates (different samples, so not identical)")
plt.tight_layout(); savefig(fig, "w2_midproject_summary.png"); plt.show()

# %% [markdown]
# ## 3. Mid-project review note

# %%
note = f"""# Mid-Project Review — End of Week 2

**Focal market:** R35 (go-live week {T0}) · **Donor pool:** {int((pool.status == 'kept').sum())} kept, {int((pool.status == 'dropped').sum())} dropped for spillover risk

## Checklist
{chk[['criterion', 'status', 'evidence']].to_markdown(index=False)}

## Headline numbers
- Synthetic-control fit: RMSPE {pre:.3f} in-sample, {val:.3f} on held-out pre-treatment weeks, {post:.3f} after go-live (post/pre ratio {post / pre:.2f}).
- Focal-market lift (synthetic control): **{100 * np.expm1(post_gap):+.1f}%**.
- Average lift across adopters (Double ML): **{dml_row['revenue_lift_%']:+.1f}%** (95% CI {100 * np.expm1(dml_row.ci_low):+.1f}% to {100 * np.expm1(dml_row.ci_high):+.1f}%).

## Item flagged for review
{("- 2c: on the held-out pre-treatment weeks the focal market ran " + f"{val_gap:+.3f}" + " log-points relative to its synthetic twin, so the post-treatment lift of " + f"{100 * np.expm1(post_gap):+.1f}" + "% may be slightly understated or overstated by a similar amount. Week 3 sensitivity checks quantify this.") if abs(val_gap) > 0.01 else "- None."}

## What is NOT yet established
- Statistical significance of the single-market estimate (placebo tests are Week 3).
- Whether the effect differs by market segment (CATE, Week 3).
- Sensitivity to hidden confounding (Week 3).
- The two estimates differ because they cover different markets; neither is "the" answer yet.
"""
(CFG.report_dir / "w2_midproject_review.md").write_text(note, encoding="utf-8")
print(note)
