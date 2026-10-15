# %% [markdown]
# # Week 2 · Day 3 — MSPE Validation of the Pre-Treatment Fit
# **Goal:** prove the synthetic market tracks the treated market *before* the intervention, so the gap
# *after* the intervention can be read as an effect.
#
# MSPE = mean squared prediction error; RMSPE = its square root (in log-revenue points, ≈ % error).
# Three numbers matter: **in-sample** pre-fit, **blocked out-of-sample** pre-fit (weeks never used for fitting),
# and the **post/pre ratio** (how much the gap grows once pricing changes).
#
# **Inputs:** `data/processed/panel.csv`, `data/raw/pricing_events.csv`, `results/w2_donor_pool.csv` (Day 1)
# **Outputs:** `results/w2_mspe_table.csv`, `results/w2_donor_pool_comparison.csv`,
# `results/w2_scm_revenue_paths.csv`, `reports/figures/w2_scm_actual_vs_synth.png`

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
res = sc.fit_scm(panel, primary, donors, T0, CFG.train_end)
print(f"Focal market {primary} | donors {len(donors)} | selected λ = {res.lam:g}")

# %% [markdown]
# ## 1. MSPE / RMSPE table

# %%
mspe = pd.DataFrame({
    "window": ["pre-treatment (in-sample)", "validation (blocked out-of-sample)", "post-treatment"],
    "weeks": [f"0-{T0 - 1}", f"{CFG.train_end}-{T0 - 1}", f"{T0}-{CFG.n_weeks - 1}"],
    "RMSPE": [res.rmspe_pre, res.rmspe_val, res.rmspe_post]})
mspe["MSPE"] = mspe.RMSPE ** 2
mspe.to_csv(CFG.results_dir / "w2_mspe_table.csv", index=False)
print(mspe.to_string(index=False))
print(f"\nPost / pre RMSPE ratio = {res.ratio:.2f}")
print("Reading: a post-treatment error well above the pre-treatment error means the actual market "
      "moved away from its synthetic twin after go-live.")

# %% [markdown]
# ## 2. Does the spillover screen (Day 1) matter?
# Same estimator, two donor pools: the screened pool vs **all** never-treated markets.

# %%
all_never = sorted(panel.loc[panel.treated_unit == 0, "market_id"].unique())
res_all = sc.fit_scm(panel, primary, all_never, T0, CFG.train_end)
cmp = pd.DataFrame({
    "donor pool": ["spillover-screened", "all never-treated"],
    "donors": [len(donors), len(all_never)],
    "pre RMSPE": [res.rmspe_pre, res_all.rmspe_pre],
    "validation RMSPE": [res.rmspe_val, res_all.rmspe_val],
    "post-period gap (log-pts)": [res.att_log, res_all.att_log],
    "revenue lift %": [100 * res.att_pct, 100 * res_all.att_pct]})
cmp.to_csv(CFG.results_dir / "w2_donor_pool_comparison.csv", index=False)
print(cmp.to_string(index=False))

# %% [markdown]
# ## 3. Actual vs synthetic counterfactual revenue

# %%
rp = res.revenue_paths()
dates = panel[panel.market_id == primary].set_index("week_idx")["week_start"]
out = rp.assign(week_start=dates, gap_log=res.gap).reset_index()
out.to_csv(CFG.results_dir / "w2_scm_revenue_paths.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].plot(rp.actual, color="#d62728", label="actual"); ax[0].plot(rp.counterfactual, "--", color="#1f77b4", label="synthetic")
ax[0].axvline(T0, color="k", ls=":"); ax[0].axvspan(CFG.train_end, T0, color="orange", alpha=0.08)
ax[0].legend(); ax[0].set_title("Revenue: actual vs synthetic (orange = validation window)")
ax[1].plot(res.gap, color="#d62728"); ax[1].axhline(0, color="k", lw=0.8); ax[1].axvline(T0, color="k", ls=":")
ax[1].set_title("Gap in log-points: near 0 before go-live, shifts after")
plt.tight_layout(); savefig(fig, "w2_scm_actual_vs_synth.png"); plt.show()

pre_gap = res.gap.loc[:T0 - 1].mean(); post_gap = res.gap.loc[T0:].mean()
print(f"Average gap before go-live: {pre_gap:+.4f} | after go-live: {post_gap:+.4f} log-points")
