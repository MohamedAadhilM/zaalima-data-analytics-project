# %% [markdown]
# # Week 3 · Day 1 — Average Treatment Effect (ATE): Estimator Triangulation
# **Goal:** quantify the average causal impact of dynamic pricing with several *independent* methods and compare
# them with naive baselines. When independent methods agree, the evidence is stronger than any single number.
#
# Estimators: two naive baselines · two-way fixed effects · DML (own, clustered SE) · EconML LinearDML ·
# imputation estimator (heterogeneity-robust) · synthetic control (focal market only).
#
# **Inputs:** `data/processed/panel.csv`, `data/raw/pricing_events.csv`, `results/w2_donor_pool.csv`,
# optional `data/raw/_ground_truth.csv` (simulation-only: the TRUE effect, used to grade the methods)
# **Outputs:** `results/w3_ate_table.csv`, `reports/figures/w3_ate_forest.png`

# %%
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpe.config import CFG
from cpe import dml, synthetic_control as sc
from cpe.utils import setup_style, savefig

setup_style(); CFG.ensure_dirs()
pd.options.display.float_format = "{:,.4f}".format
panel = pd.read_csv(CFG.processed_dir / "panel.csv")
events = pd.read_csv(CFG.raw_dir / "pricing_events.csv")
pool = pd.read_csv(CFG.results_dir / "w2_donor_pool.csv")
gt_path = CFG.raw_dir / "_ground_truth.csv"
gt = pd.read_csv(gt_path) if gt_path.exists() else None
T0 = CFG.primary_adoption_week
primary = events.loc[events.go_live_week_idx == T0, "market_id"].iloc[0]
donors = pool.loc[pool.status == "kept", "market_id"].tolist()

# %% [markdown]
# ## 1. Run every estimator on the same panel

# %%
prep = dml.prepare(panel)
naive = dml.naive_estimates(panel)
tw = dml.twfe(prep)
fit = dml.dml_plr(prep, n_folds=CFG.n_folds)
lin = dml.econml_linear_dml(prep, n_folds=CFG.n_folds)
bjs = dml.bjs_imputation(panel)
scm = sc.fit_scm(panel, primary, donors, T0, CFG.train_end)

truth_all = dml.true_att(panel, gt) if gt is not None else np.nan
truth_primary = float(gt.loc[gt.is_primary, "true_tau_log"].iloc[0]) if gt is not None else np.nan

rows = [
    ("Naive: treated vs never-treated", naive["naive_treated_vs_never_post"], np.nan, np.nan, truth_all),
    ("Naive: before vs after (treated)", naive["naive_before_after"], np.nan, np.nan, truth_all),
    ("Two-way fixed effects", tw["theta"], *tw["ci"], truth_all),
    ("DML-PLR (LightGBM, clustered SE)", fit["theta"], *fit["ci"], truth_all),
    ("EconML LinearDML", lin["theta"], *lin["ci"], truth_all),
    ("Imputation (BJS, heterogeneity-robust)", bjs["theta"], *bjs["ci"], truth_all),
    (f"Synthetic control: {primary} only", scm.att_log, np.nan, np.nan, truth_primary),
]
ate = pd.DataFrame(rows, columns=["estimator", "estimate_log_pts", "ci_low", "ci_high", "truth_log_pts"])
ate["bias_vs_truth"] = ate.estimate_log_pts - ate.truth_log_pts
ate["revenue_lift_%"] = 100 * np.expm1(ate.estimate_log_pts)
ate.to_csv(CFG.results_dir / "w3_ate_table.csv", index=False)
ate

# %% [markdown]
# ## 2. Forest plot
# Black tick = the simulation's true effect (only known because the data are simulated). The first naive row is
# far off the chart and is clipped.

# %%
fig, ax = plt.subplots(figsize=(9, 4.2))
y = np.arange(len(ate))[::-1]
for yi, r in zip(y, ate.itertuples()):
    if not np.isnan(r.ci_low):
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color="#1f77b4", lw=3, alpha=0.5)
    ax.plot(r.estimate_log_pts, yi, "o", color="#d62728" if r.estimator.startswith("Naive") else "#1f77b4")
    if not np.isnan(r.truth_log_pts):
        ax.plot(r.truth_log_pts, yi, "k|", ms=14)
ax.set_yticks(y); ax.set_yticklabels(ate.estimator); ax.set_xlim(-0.05, 0.2)
ax.set_xlabel("effect on log revenue"); ax.set_title("ATE by estimator (black tick = simulation truth)")
plt.tight_layout(); savefig(fig, "w3_ate_forest.png"); plt.show()

# %% [markdown]
# ## 3. Reading the table

# %%
causal = ate.iloc[2:6]
print(f"Naive treated-vs-never-treated read-out: {ate.iloc[0].estimate_log_pts:+.3f} log-pts "
      f"({ate.iloc[0]['revenue_lift_%']:+.0f}%) - confounded by market size, urbanity and spillovers.")
print(f"Causal estimators range {causal.estimate_log_pts.min():+.3f} to {causal.estimate_log_pts.max():+.3f} log-pts "
      f"({causal['revenue_lift_%'].min():+.1f}% to {causal['revenue_lift_%'].max():+.1f}% revenue lift).")
if not np.isnan(truth_all):
    print(f"Simulation truth: {truth_all:+.3f} log-pts. Mean bias of the four causal estimators: "
          f"{causal.bias_vs_truth.mean():+.3f} (they understate the lift).")
