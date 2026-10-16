# %% [markdown]
# # Week 2 · Day 4 — Double Machine Learning with High-Dimensional Controls
# **Goal:** estimate the average effect of dynamic pricing on revenue across *all* adopting markets,
# adjusting for many confounders at once without assuming a simple linear relationship.
#
# **Model (partially linear):** after removing market and week fixed effects,
# `Y = θ·D + g(W) + ε`, where `Y` = log revenue, `D` = dynamic-pricing flag, `W` = controls.
# DML predicts `Y` and `D` from `W` with machine learning (LightGBM) on *held-out folds*, then regresses the
# leftover parts on each other. The slope θ is the effect, with market-clustered standard errors.
#
# **Inputs:** `data/processed/panel.csv`
# **Outputs:** `results/w2_dml_results.csv`, `results/w2_dml_features.csv`, `results/w2_dml_nuisance_diagnostics.csv`,
# `reports/figures/w2_dml_diagnostics.png`

# %%
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpe.config import CFG
from cpe import dml
from cpe.utils import setup_style, savefig

setup_style(); CFG.ensure_dirs()
pd.options.display.float_format = "{:,.4f}".format
panel = pd.read_csv(CFG.processed_dir / "panel.csv")

# %% [markdown]
# ## 1. Controls (high-dimensional W)
# Only variables **not affected by the treatment** are used: competitor-price lags and momentum, seasonality
# (Fourier terms), market structure, interactions and neighbour spillover exposure. Lagged revenue is deliberately
# excluded because after go-live it is partly caused by the treatment.

# %%
prep = dml.prepare(panel)
feat = pd.DataFrame({"feature": prep["W"].columns})
feat.to_csv(CFG.results_dir / "w2_dml_features.csv", index=False)
print(f"{prep['W'].shape[1]} control features | {len(prep['Y'])} market-week observations | "
      f"{pd.Series(prep['groups']).nunique()} markets")
print(", ".join(feat.feature))

# %% [markdown]
# ## 2. Cross-fitted DML estimate
# Folds are split **by market**, so a market's weeks never appear on both the training and testing side
# (weeks within a market are strongly correlated).

# %%
fit = dml.dml_plr(prep, n_folds=CFG.n_folds)
lin = dml.econml_linear_dml(prep, n_folds=CFG.n_folds)

# reference: the same regression with NO controls (only fixed effects)
theta_nocontrols = float((prep["D"] * prep["Y"]).sum() / (prep["D"] ** 2).sum())

res = pd.DataFrame({
    "method": ["DML-PLR (own implementation, clustered SE)", "EconML LinearDML (cross-check)",
               "Fixed effects only, no ML controls (reference)"],
    "theta_log_points": [fit["theta"], lin["theta"], theta_nocontrols],
    "ci_low": [fit["ci"][0], lin["ci"][0], np.nan], "ci_high": [fit["ci"][1], lin["ci"][1], np.nan]})
res["revenue_lift_%"] = 100 * np.expm1(res.theta_log_points)
res.to_csv(CFG.results_dir / "w2_dml_results.csv", index=False)
print(res.round(4).to_string(index=False))
print(f"\nDML-PLR: standard error {fit['se']:.4f} | t = {fit['t']:.2f} | p = {fit['p']:.3f}")

# %% [markdown]
# ## 3. Nuisance-model quality
# How well do the ML models predict the outcome and the treatment from the controls (out-of-fold R²)?

# %%
diag = pd.DataFrame({"model": ["outcome model E[Y|W]", "treatment model E[D|W]"],
                     "out_of_fold_R2": [fit["r2_outcome"], fit["r2_treatment"]]})
diag.to_csv(CFG.results_dir / "w2_dml_nuisance_diagnostics.csv", index=False)
print(diag.to_string(index=False))
print("\nA treatment R² well below 1 means adoption is not fully predictable from the controls, so there is "
      "leftover variation in D to learn the effect from (overlap).")

# %%
Yr, Dr = fit["Yres"], fit["Dres"]
bins = pd.qcut(Dr, 20, duplicates="drop")
bm = pd.DataFrame({"D": Dr, "Y": Yr}).groupby(bins, observed=True).mean()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].scatter(bm.D, bm.Y, color="#1f77b4")
xs = np.linspace(bm.D.min(), bm.D.max(), 50); ax[0].plot(xs, fit["theta"] * xs, color="#d62728")
ax[0].set_xlabel("treatment residual (D − Ê[D|W])"); ax[0].set_ylabel("outcome residual (Y − Ê[Y|W])")
ax[0].set_title(f"Residual-on-residual (binned): slope θ = {fit['theta']:.3f}")
ax[1].bar(["outcome R²", "treatment R²"], [fit["r2_outcome"], fit["r2_treatment"]], color=["#1f77b4", "#9ecae1"])
ax[1].set_ylim(0, 1); ax[1].set_title("Out-of-fold nuisance R²")
plt.tight_layout(); savefig(fig, "w2_dml_diagnostics.png"); plt.show()
