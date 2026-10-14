# %% [markdown]
# # Week 2 · Day 2 — Synthetic Control Optimisation
# **Goal:** build the weighted average of donor markets that reproduces the treated market's behaviour
# *before* the intervention. That weighted average is the synthetic ("what-if-nothing-changed") market.
#
# **Method:** intercept-adjusted, ridge-penalised convex weights (non-negative, sum to 1) fitted on the
# weekly pre-treatment revenue path. The ridge strength is chosen on a *blocked validation window*.
#
# **Inputs:** `data/processed/panel.csv`, `data/raw/pricing_events.csv`, `results/w2_donor_pool.csv` (Day 1)
# **Outputs:** `results/w2_scm_weights.csv`, `results/w2_scm_balance.csv`, `results/w2_scm_fit_summary.csv`,
# `reports/figures/w2_scm_weights_fit.png`

# %%
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpe.config import CFG
from cpe import synthetic_control as sc
from cpe.utils import setup_style, savefig

setup_style(); CFG.ensure_dirs()
pd.options.display.float_format = "{:,.4f}".format
panel = pd.read_csv(CFG.processed_dir / "panel.csv")
events = pd.read_csv(CFG.raw_dir / "pricing_events.csv")
pool = pd.read_csv(CFG.results_dir / "w2_donor_pool.csv")

T0 = CFG.primary_adoption_week
primary = events.loc[events.go_live_week_idx == T0, "market_id"].iloc[0]
donors = pool.loc[pool.status == "kept", "market_id"].tolist()
print(f"Focal market {primary} | go-live week {T0} | donors: {len(donors)}")
print(f"Windows: train weeks 0-{CFG.train_end - 1}, validation {CFG.train_end}-{T0 - 1}, post {T0}+")

# %% [markdown]
# ## 1. Choose the ridge penalty λ by blocked validation
# For each λ: fit weights on the **training** weeks only, then score the error on the **validation** weeks that
# were never used for fitting. Lower is better; ties go to the more regularised (spread-out) solution.

# %%
res = sc.fit_scm(panel, primary, donors, T0, CFG.train_end)
curve = pd.DataFrame({"lambda": list(res.val_curve.keys()),
                      "validation_RMSPE": list(res.val_curve.values())})
print(curve.to_string(index=False))
print(f"\nSelected λ = {res.lam:g}")

# %% [markdown]
# ## 2. Donor weights
# Weights are ≥ 0 and sum to 1. A spread-out set of weights is more reliable than leaning on one donor.

# %%
weights = res.weights.rename("weight").rename_axis("market_id").reset_index()
weights.to_csv(CFG.results_dir / "w2_scm_weights.csv", index=False)
print(weights.head(10).to_string(index=False))
print(f"\nSum of weights = {weights.weight.sum():.4f} | donors with weight > 1%: {(weights.weight > 0.01).sum()}")

# %% [markdown]
# ## 3. Covariate balance
# Does the synthetic market *look like* the treated market on characteristics that were **not** used for fitting?
# Closer to the treated value than the plain donor average = better.

# %%
bal = sc.balance_table(panel, res)
bal["gap: synthetic"] = (bal.treated - bal.synthetic).abs()
bal["gap: donor avg"] = (bal.treated - bal.donor_avg).abs()
bal.round(4).rename_axis("covariate").reset_index().to_csv(CFG.results_dir / "w2_scm_balance.csv", index=False)
bal

# %% [markdown]
# ## 4. Fit summary and picture

# %%
summary = pd.DataFrame({"metric": ["selected lambda", "pre-treatment RMSPE (in-sample)",
                                   "validation RMSPE (out-of-sample)", "donors used", "donors with weight > 1%"],
                        "value": [res.lam, res.rmspe_pre, res.rmspe_val, len(donors), int((res.weights > 0.01).sum())]})
summary.to_csv(CFG.results_dir / "w2_scm_fit_summary.csv", index=False)
print(summary.to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(14, 3.7))
ax[0].plot([str(k) for k in res.val_curve], list(res.val_curve.values()), "o-")
ax[0].set_title("Validation RMSPE vs ridge λ"); ax[0].set_xlabel("λ"); ax[0].tick_params(axis="x", rotation=45)
w = res.weights[res.weights > 0.01]
ax[1].barh(w.index[::-1], w.values[::-1], color="#1f77b4"); ax[1].set_title("Donor weights (> 1%)")
ax[2].plot(res.actual_log, label=f"{primary} actual", color="#d62728")
ax[2].plot(res.synth_log, "--", label="synthetic", color="#1f77b4")
ax[2].axvline(T0, color="k", ls=":"); ax[2].set_title("Actual vs synthetic (log revenue)"); ax[2].legend()
plt.tight_layout(); savefig(fig, "w2_scm_weights_fit.png"); plt.show()
