# %% [markdown]
# # Week 3 · Day 3 — Synthetic Control Inference & Robustness
# **Goal:** decide whether the focal market's post-launch gap is a real effect or just noise.
#
# * **Placebo-in-space:** pretend each donor market got the treatment and measure the "effect" it would show.
#   If the focal market's effect is extreme compared with these fake ones, it is unlikely to be chance.
# * **Placebo-in-time:** pretend the launch happened earlier. There should be no effect before the real launch.
# * **Leave-one-donor-out:** drop the heaviest donors one at a time; the answer should not depend on any one of them.
#
# **Inputs:** `data/processed/panel.csv`, `data/raw/pricing_events.csv`, `results/w2_donor_pool.csv`
# **Outputs:** `results/w3_placebo_in_space.csv`, `results/w3_placebo_gaps.csv`, `results/w3_placebo_summary.csv`,
# `results/w3_placebo_in_time.csv`, `results/w3_leave_one_donor_out.csv`, `reports/figures/w3_placebo_in_space.png`

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
scm = sc.fit_scm(panel, primary, donors, T0, CFG.train_end)
print(f"Focal {primary}: gap after go-live {scm.att_log:+.4f} log-pts ({100 * scm.att_pct:+.1f}%), "
      f"post/pre RMSPE ratio {scm.ratio:.2f}")

# %% [markdown]
# ## 1. Placebo-in-space (permutation inference)
# Each donor is treated as if it were the launched market (the ridge penalty is held fixed). Placebos whose
# pre-launch fit is more than 2x worse than the focal fit are excluded, as is standard.

# %%
pl = sc.placebo_in_space(panel, scm, CFG.train_end)
pl.sort_values("ratio", ascending=False).to_csv(CFG.results_dir / "w3_placebo_in_space.csv", index=False)
pl.attrs["gaps"].assign(**{primary: scm.gap}).rename_axis("week_idx").reset_index().to_csv(
    CFG.results_dir / "w3_placebo_gaps.csv", index=False)
summary = pd.DataFrame({
    "metric": ["focal gap (log-pts)", "focal post/pre RMSPE ratio", "placebos kept (comparable pre-fit)",
               "permutation p-value (|gap|)", "permutation p-value (RMSPE ratio)"],
    "value": [scm.att_log, scm.ratio, pl.attrs["n_kept"], pl.attrs["p_att_two_sided"], pl.attrs["p_ratio"]]})
summary.to_csv(CFG.results_dir / "w3_placebo_summary.csv", index=False)
print(summary.to_string(index=False))

# %%
gaps = pl.attrs["gaps"]
fig, ax = plt.subplots(1, 2, figsize=(13, 3.9))
ax[0].plot(gaps, color="grey", alpha=0.35, lw=0.9); ax[0].plot(scm.gap, color="#d62728", lw=2.4, label=primary)
ax[0].axvline(T0, c="k", ls=":"); ax[0].axhline(0, c="k", lw=0.7); ax[0].set_ylim(-0.25, 0.25)
ax[0].set_title("Placebo-in-space: focal gap (red) vs donor placebos"); ax[0].legend()
ratios = pl.sort_values("ratio")["ratio"].to_numpy()
ax[1].bar(range(len(ratios)), ratios, color="grey"); ax[1].axhline(scm.ratio, color="#d62728", lw=2)
ax[1].set_title("Post/pre RMSPE ratios (red = focal market)"); ax[1].set_xlabel("donor placebos (sorted)")
plt.tight_layout(); savefig(fig, "w3_placebo_in_space.png"); plt.show()

# %% [markdown]
# ## 2. Placebo-in-time
# The pretend launch is moved 6, 10 and 14 weeks earlier. The "effect" measured in the fake window should be near zero.

# %%
pt = sc.placebo_in_time(panel, scm)
pt.to_csv(CFG.results_dir / "w3_placebo_in_time.csv", index=False)
print(pt.round(4).to_string(index=False))

# %% [markdown]
# ## 3. Leave-one-donor-out

# %%
loo = sc.leave_one_donor_out(panel, scm, CFG.train_end, top=5)
loo.to_csv(CFG.results_dir / "w3_leave_one_donor_out.csv", index=False)
print(loo.round(4).to_string(index=False))

# %% [markdown]
# ## 4. Honest reading

# %%
print(f"- Permutation p-value for the focal effect: {pl.attrs['p_att_two_sided']:.2f} "
      f"(smallest possible with {pl.attrs['n_kept']} placebos is {1 / (1 + pl.attrs['n_kept']):.2f}).")
print(f"- Placebo-in-time gaps range {pt.placebo_att_log.min():+.3f} to {pt.placebo_att_log.max():+.3f}; "
      f"the focal effect is {scm.att_log:+.3f}.")
print(f"- Dropping any of the 5 heaviest donors keeps the gap between {loo.att_log.min():+.3f} and {loo.att_log.max():+.3f}.")
print("- A single-market synthetic control is inherently noisy; treat the p-value as suggestive, not conclusive.")
