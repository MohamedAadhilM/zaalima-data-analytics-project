# %% [markdown]
# # Week 3 · Day 4 — Sensitivity Analysis: How Wrong Could We Be?
# **Goal:** test the claim "dynamic pricing raised revenue" against violations of the causal assumptions,
# above all *hidden confounding* (an unmeasured factor that drives both adoption and revenue).
#
# 1. **Robustness value (Cinelli-Hazlett):** how strong would a hidden confounder need to be to erase the effect?
# 2. **Bias grid:** the adjusted estimate for hypothetical confounder strengths.
# 3. **DoWhy refuters:** placebo treatment, random common cause, data subset, unobserved common cause.
#
# **Inputs:** `data/processed/panel.csv`
# **Outputs:** `results/w3_robustness_value.csv`, `results/w3_ovb_grid.csv`, `results/w3_dowhy_refutations.csv`,
# `reports/figures/w3_sensitivity_grid.png`

# %%
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpe.config import CFG
from cpe import dml, sensitivity as sens
from cpe.utils import setup_style, savefig

setup_style(); CFG.ensure_dirs()
pd.options.display.float_format = "{:,.4f}".format
panel = pd.read_csv(CFG.processed_dir / "panel.csv")
prep = dml.prepare(panel)
fit = dml.dml_plr(prep, n_folds=CFG.n_folds)
print(f"DML effect: {fit['theta']:+.4f} log-pts, SE {fit['se']:.4f}, t = {fit['t']:.2f}")

# %% [markdown]
# ## 1. Robustness value
# `RV` = the share of the *leftover* variation (after the controls) that a hidden confounder must explain in
# **both** adoption and revenue to reduce the estimate to zero. `RV (5%)` = strength needed to make it
# statistically indistinguishable from zero. Bigger = more robust.

# %%
rv = sens.robustness_value(fit["theta"], fit["se"], fit["dof"])
rv_tab = pd.DataFrame({"metric": ["t-statistic", "partial R² of treatment with outcome",
                                  "RV: confounder strength to erase the estimate",
                                  "RV (5%): strength to make it insignificant"],
                       "value": [rv["t_stat"], rv["partial_r2_treat_outcome"], rv["RV_q"], rv["RV_q_alpha"]]})
rv_tab.to_csv(CFG.results_dir / "w3_robustness_value.csv", index=False)
print(rv_tab.to_string(index=False))

# %% [markdown]
# ## 2. Bias grid
# Each cell: the estimate if a hidden confounder explained that share of the leftover variation in the outcome
# (rows) and in adoption (columns), in the worst direction.

# %%
grid = sens.omitted_variable_bias_grid(fit["theta"], fit["se"], fit["dof"])
grid.to_csv(CFG.results_dir / "w3_ovb_grid.csv")
fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.imshow(grid.to_numpy(), cmap="RdYlGn", vmin=-0.03, vmax=0.03, origin="lower")
ax.set_xticks(range(grid.shape[1])); ax.set_xticklabels([f"{c:.0%}" for c in grid.columns])
ax.set_yticks(range(grid.shape[0])); ax.set_yticklabels([f"{c:.0%}" for c in grid.index])
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(j, i, f"{grid.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=8)
ax.set_xlabel("confounder partial R² with adoption"); ax.set_ylabel("confounder partial R² with revenue")
ax.set_title("Adjusted effect under hidden confounding"); plt.tight_layout()
savefig(fig, "w3_sensitivity_grid.png"); plt.show()
print(grid.round(4).to_string())

# %% [markdown]
# ## 3. DoWhy refutation tests
# The effect is re-estimated after deliberately breaking or stressing the data:
# * **placebo treatment** - shuffle the treatment; the effect should collapse to ~0
# * **random common cause** - add a random confounder; the effect should not move
# * **data subset** - use 80% of the rows; the effect should stay similar
# * **unobserved common cause** - add a simulated hidden confounder; shows how much the effect shrinks

# %%
refut = sens.dowhy_refutations(panel, n_sim=15)
refut.to_csv(CFG.results_dir / "w3_dowhy_refutations.csv", index=False)
print(refut.round(4).to_string(index=False))

# %% [markdown]
# ## 4. Honest reading

# %%
print(f"- A hidden confounder would need partial R² of about {rv['RV_q']:.1%} with BOTH adoption and revenue to erase "
      f"the effect, and {rv['RV_q_alpha']:.1%} to make it insignificant.")
print("- These values are SMALL: the effect is real-looking but fragile to a modest hidden confounder (e.g. an unrecorded "
      "promotion or local event that coincided with go-live). Say so plainly.")
print("- Whether such a confounder exists cannot be tested from the data; it is a judgement about the business.")
print("- Placebo and random-cause refuters behaving as expected support the estimate, but do not prove it.")
