# %% [markdown]
# # Week 3 · Day 5 — Cross-Validation of the Causal Models & Week 3 Summary
# **Goal:** check that the causal models are *reliable out of sample* and wrap up Week 3.
#
# * **Market-grouped K-fold:** hold out whole markets; check the effect estimate is stable across folds and compare
#   candidate effect models (no effect / one constant effect / effect varying with market traits) by held-out R-loss.
# * **Rolling-origin validation of the synthetic control:** fit on early weeks, forecast the next 8 weeks, repeat.
#
# **Inputs:** `data/processed/panel.csv`, `data/raw/pricing_events.csv`, `results/w2_donor_pool.csv`, and the Day 1-4
# results `results/w3_*.csv` for the summary
# **Outputs:** `results/w3_cv_folds.csv`, `results/w3_scm_rolling_cv.csv`, `results/w3_week3_checklist.csv`,
# `reports/w3_week3_summary.md`, `reports/figures/w3_cross_validation.png`

# %%
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpe.config import CFG
from cpe import dml, sensitivity as sens, synthetic_control as sc
from cpe.utils import setup_style, savefig

setup_style(); CFG.ensure_dirs()
pd.options.display.float_format = "{:,.4f}".format
R = CFG.results_dir
panel = pd.read_csv(CFG.processed_dir / "panel.csv")
events = pd.read_csv(CFG.raw_dir / "pricing_events.csv")
pool = pd.read_csv(R / "w2_donor_pool.csv")
T0 = CFG.primary_adoption_week
primary = events.loc[events.go_live_week_idx == T0, "market_id"].iloc[0]
donors = pool.loc[pool.status == "kept", "market_id"].tolist()

# %% [markdown]
# ## 1. Market-grouped cross-validation of the DML model
# Each fold leaves out whole markets. `R-loss` scores how well a candidate effect model explains held-out
# revenue changes (lower is better; shown x10,000).

# %%
prep = dml.prepare(panel)
cv = sens.grouped_cv_report(prep, n_folds=CFG.n_folds)["folds"]
cv.to_csv(R / "w3_cv_folds.csv", index=False)
show = cv.copy()
for c in ["rloss_zero", "rloss_constant", "rloss_linear_cate"]:
    show[c] = show[c] * 1e4
print(show.round(3).to_string(index=False))
print(f"\nEffect estimate across folds: mean {cv.theta_fold.mean():+.4f}, sd {cv.theta_fold.std():.4f}, "
      f"min {cv.theta_fold.min():+.4f}, max {cv.theta_fold.max():+.4f}")
better = (cv.rloss_constant < cv.rloss_zero).sum()
print(f"A constant effect beats 'no effect' in {better} of {len(cv)} folds; "
      f"an effect varying with market traits beats a constant effect in {(cv.rloss_linear_cate < cv.rloss_constant).sum()} of {len(cv)}.")

# %% [markdown]
# ## 2. Rolling-origin validation of the synthetic control

# %%
scm = sc.fit_scm(panel, primary, donors, T0, CFG.train_end)
roll = sc.rolling_origin_cv(panel, primary, donors, T0, lam=scm.lam)
roll.to_csv(R / "w3_scm_rolling_cv.csv", index=False)
print(roll.round(4).to_string(index=False))
print(f"\nMean 8-week-ahead forecast error: {roll.oos_rmspe.mean():.4f} log-pts (compare with the post-launch gap {scm.att_log:+.4f})")

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 3.7))
ax[0].bar(cv.fold.astype(str), cv.theta_fold, color="#1f77b4"); ax[0].axhline(cv.theta_fold.mean(), color="k", ls="--")
ax[0].set_title("DML effect by held-out fold"); ax[0].set_xlabel("fold")
x = np.arange(len(cv)); w = 0.27
ax[1].bar(x - w, cv.rloss_zero * 1e4, w, label="no effect"); ax[1].bar(x, cv.rloss_constant * 1e4, w, label="constant")
ax[1].bar(x + w, cv.rloss_linear_cate * 1e4, w, label="varies with traits"); ax[1].legend(); ax[1].set_title("Held-out R-loss x1e4 (lower = better)")
ax[1].set_ylim(cv[["rloss_zero", "rloss_constant", "rloss_linear_cate"]].min().min() * 1e4 * 0.97, None)
ax[2].bar(roll.cutoff_week.astype(str), roll.oos_rmspe, color="#2ca02c"); ax[2].axhline(abs(scm.att_log), color="#d62728", ls="--", label="focal gap")
ax[2].legend(); ax[2].set_title("SCM 8-week-ahead forecast error"); ax[2].set_xlabel("forecast starts at week")
plt.tight_layout(); savefig(fig, "w3_cross_validation.png"); plt.show()

# %% [markdown]
# ## 3. Week 3 checklist (from the project plan) and summary
# Reads the Day 1-4 results. Pass rules are our own reasonable cut-offs.

# %%
ate = pd.read_csv(R / "w3_ate_table.csv")
seg = pd.read_csv(R / "w3_cate_by_segment.csv")
pls = pd.read_csv(R / "w3_placebo_summary.csv").set_index("metric")["value"]
pt = pd.read_csv(R / "w3_placebo_in_time.csv")
loo = pd.read_csv(R / "w3_leave_one_donor_out.csv")
rv = pd.read_csv(R / "w3_robustness_value.csv").set_index("metric")["value"]
ref = pd.read_csv(R / "w3_dowhy_refutations.csv")
causal = ate.iloc[2:6]

placebo_row = ref[ref.refuter == "placebo_treatment_refuter"]
checks = [
    ("Quantify impact: ATE", "causal estimators agree in sign and sit far from the naive read-out",
     bool((causal.estimate_log_pts > 0).all() and abs(ate.iloc[0].estimate_log_pts) > 5 * causal.estimate_log_pts.max()),
     f"causal ATE {causal.estimate_log_pts.min():+.3f} to {causal.estimate_log_pts.max():+.3f} vs naive {ate.iloc[0].estimate_log_pts:+.3f}"),
    ("Quantify impact: CATE", "segment effects differ and at least one segment excludes zero",
     bool(seg.cate.max() - seg.cate.min() > 0.02 and (seg.ci_low > 0).any()),
     "; ".join(f"{r.income_tercile} {100 * np.expm1(r.cate):+.1f}%" for r in seg.itertuples())),
    ("Sensitivity: placebo-in-time", "all fake-launch gaps smaller than the real gap",
     bool((pt.placebo_att_log.abs() < abs(pt.true_att_log)).all()),
     f"fake gaps {pt.placebo_att_log.min():+.3f} to {pt.placebo_att_log.max():+.3f}"),
    ("Sensitivity: leave-one-donor-out", "gap stays within ±0.01 of the full-pool gap",
     bool((loo.att_log - pls["focal gap (log-pts)"]).abs().max() <= 0.01),
     f"gap {loo.att_log.min():+.3f} to {loo.att_log.max():+.3f}"),
    ("Sensitivity: permutation test", "p-value <= 0.10", bool(pls["permutation p-value (|gap|)"] <= 0.10),
     f"p = {pls['permutation p-value (|gap|)']:.2f} ({int(pls['placebos kept (comparable pre-fit)'])} placebos)"),
    ("Sensitivity: placebo-treatment refuter", "shuffled treatment gives ~0 effect",
     bool(len(placebo_row) and abs(placebo_row.new_effect.iloc[0]) < 0.5 * abs(placebo_row.original.iloc[0])),
     f"{placebo_row.new_effect.iloc[0]:+.4f} vs {placebo_row.original.iloc[0]:+.4f}" if len(placebo_row) else "n/a"),
    ("Sensitivity: hidden-confounder strength", "robustness value >= 5%", bool(rv["RV: confounder strength to erase the estimate"] >= 0.05),
     f"RV {rv['RV: confounder strength to erase the estimate']:.1%}"),
    ("Cross-validation: DML folds", "effect positive in a clear majority of held-out folds",
     bool((cv.theta_fold > 0).mean() >= 0.8), f"{int((cv.theta_fold > 0).sum())} of {len(cv)} folds positive, sd {cv.theta_fold.std():.3f}"),
    ("Cross-validation: SCM rolling origin", "forecast error below the focal gap",
     bool(roll.oos_rmspe.mean() < abs(scm.att_log)), f"error {roll.oos_rmspe.mean():.3f} vs gap {abs(scm.att_log):.3f}"),
]
chk = pd.DataFrame(checks, columns=["criterion", "pass_rule", "passed", "evidence"])
chk["status"] = np.where(chk.passed, "PASS", "REVIEW")
chk.to_csv(R / "w3_week3_checklist.csv", index=False)
print(chk[["criterion", "status", "evidence"]].to_string(index=False))
print(f"\n{int(chk.passed.sum())} of {len(chk)} passed")

# %%
note = f"""# Week 3 Summary — Causal Impact, Sensitivity and Cross-Validation

## Checklist
{chk[['criterion', 'status', 'evidence']].to_markdown(index=False)}

## What we can say
- Causal estimators agree that dynamic pricing raised revenue, by roughly {100 * np.expm1(causal.estimate_log_pts.min()):.1f}% to {100 * np.expm1(causal.estimate_log_pts.max()):.1f}%, while the naive comparison reports {100 * np.expm1(ate.iloc[0].estimate_log_pts):.0f}%.
- The effect is not uniform: {'; '.join(f"{r.income_tercile} {100 * np.expm1(r.cate):+.1f}% ({r.decision})" for r in seg.itertuples())}.

## What we cannot say
- The robustness value is {rv['RV: confounder strength to erase the estimate']:.1%}: a modest hidden confounder could erase the average effect.
- The single-market p-value is the smallest possible with {int(pls['placebos kept (comparable pre-fit)'])} placebos, so it is suggestive, not conclusive.
- Cross-validation only weakly supports effect heterogeneity: a model with market-trait effects beat a constant effect in {int((cv.rloss_linear_cate < cv.rloss_constant).sum())} of {len(cv)} held-out folds, and the effect estimate varies across folds (sd {cv.theta_fold.std():.3f}).
- Segment intervals ignore clustering by market and rest on 3-5 markets per segment, so treat them as optimistic.
"""
(CFG.report_dir / "w3_week3_summary.md").write_text(note, encoding="utf-8")
print(note)
