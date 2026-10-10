import _bootstrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpe.config import CFG
from cpe.data import simulate_raw, ingest, load_ground_truth, primary_market
from cpe import dag
from cpe.utils import setup_style, savefig, save_json

setup_style()
CFG.ensure_dirs()
pd.options.display.float_format = "{:,.4f}".format
sim = simulate_raw(CFG)             # remove this line when using real extracts
panel, report = ingest(CFG)
print("Ingestion report")
print(pd.Series(report.to_dict()).to_string())
panel.head()
assert report.panel_balanced, "panel must be balanced for the within transformation"
quality = pd.DataFrame({
    "check": ["balanced panel", "duplicate telemetry rows removed", "missing competitor prices (interpolated)",
              "spike outliers replaced", "markets", "weeks"],
    "value": [report.panel_balanced, report.duplicates_removed, report.missing_competitor_price,
              report.outliers_winsorised, report.n_markets, report.n_weeks]})
quality
treated = panel[panel.treated_unit == 1]
golive = treated.groupby("market_id")["go_live"].first().sort_values()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].barh(golive.index, golive.values, color="#1f77b4")
ax[0].axvline(CFG.primary_adoption_week, color="r", ls="--", lw=1)
ax[0].set_title("Staggered dynamic-pricing roll-out (go-live week)")
ax[0].set_xlabel("week index")
g = panel.groupby(["week_idx", "treated_unit"])["log_revenue"].mean().unstack()
ax[1].plot(g.index, g[0], label="never-treated markets")
ax[1].plot(g.index, g[1], label="eventually-treated markets")
ax[1].set_title("Mean log revenue: groups differ in *level* before any treatment")
ax[1].legend()
plt.tight_layout(); savefig(fig, "w1_rollout_and_levels.png"); plt.show()
static = panel.drop_duplicates("market_id").set_index("market_id")
cols = ["log_pop", "income_z", "urban_index", "comp_intensity"]
t, c = static[static.treated_unit == 1][cols], static[static.treated_unit == 0][cols]
smd = ((t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)).rename("SMD").to_frame()
smd["imbalanced (|SMD|>0.25)"] = smd.SMD.abs() > 0.25
smd

# %%
post = treated[treated.dynamic_pricing == 1]
never_post = panel[(panel.treated_unit == 0) & (panel.week_idx >= post.week_idx.min())]
naive = post.log_revenue.mean() - never_post.log_revenue.mean()
gt = load_ground_truth(CFG)
truth = panel[panel.dynamic_pricing == 1].merge(gt[["market_id", "true_tau_log"]], on="market_id").true_tau_log.mean()
print(f"Naive treated-vs-control 'lift'       : {naive:+.1%}  (log-points)")
print(f"True average effect (simulation only) : {truth:+.1%}")
print("→ the naive read-out is wildly biased because adopters are larger/urban markets.")
G = dag.build_dag()
fig = dag.plot_dag(G, CFG.fig_dir / "w1_causal_dag.png"); plt.show()
dag.confounder_report(G)
model, estimand, adjustment_set = dag.identify(panel, G)
print("Back-door adjustment set found by DoWhy:", adjustment_set)
print(estimand)
primary = primary_market(CFG)
T0 = CFG.primary_adoption_week
windows = {"train": (0, CFG.train_end - 1), "validation": (CFG.train_end, T0 - 1), "post": (T0, CFG.n_weeks - 1)}
print(f"Focal market: {primary}   |   windows (week idx): {windows}")

Y = panel.pivot(index="week_idx", columns="market_id", values="log_revenue")
never = [m for m in Y.columns if m not in set(treated.market_id)]
Yd = Y - Y.iloc[:T0].mean()                       # deviations from own pre-period level
band_lo, band_hi = Yd[never].quantile(0.1, axis=1), Yd[never].quantile(0.9, axis=1)

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.fill_between(Yd.index, band_lo, band_hi, color="grey", alpha=0.25, label="never-treated (10–90%)")
ax.plot(Yd.index, Yd[never].mean(1), color="grey", lw=1.5, label="never-treated mean")
ax.plot(Yd.index, Yd[primary], color="#d62728", lw=2, label=f"{primary} (focal)")
for k, (a, b) in windows.items():
    ax.axvspan(a, b + 1, alpha=0.05, color={"train": "b", "validation": "orange", "post": "r"}[k])
    ax.text((a + b) / 2, ax.get_ylim()[1] * 0.92, k, ha="center", fontsize=9)
ax.axvline(T0, color="k", ls="--"); ax.set_ylabel("log revenue, demeaned (pre-period)")
ax.set_title("Baseline trajectories: focal market vs donor pool"); ax.legend(loc="lower left")
savefig(fig, "w1_baseline_trajectories.png"); plt.show()
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

gap_pre = (Yd[primary] - Yd[never].mean(1)).iloc[:T0]
adf_stat, adf_p, *_ = adfuller(gap_pre, regression="c", autolag="AIC")
slope, intercept = np.polyfit(np.arange(T0), gap_pre.to_numpy(), 1)
stl = STL(Y[primary].iloc[:T0], period=52, robust=True).fit()
seas_strength = max(0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
diag = pd.Series({"ADF statistic (gap)": adf_stat, "ADF p-value": adf_p,
                  "pre-trend slope (log-pts / week)": slope,
                  "seasonal strength (STL, 0-1)": seas_strength})
print(diag.round(4).to_string())
save_json({"primary": primary, "t0": T0, "windows": windows, "never_treated": never,
           "treated": sorted(treated.market_id.unique()), "adjustment_set": adjustment_set,
           "ingest_report": report.to_dict(), "naive_lift_log": naive, "diagnostics": diag.to_dict()},
          "week1_summary.json")

checklist = pd.DataFrame({"Week-1 requirement": [
    "Ingest regional market transaction datasets & competitor pricing telemetry",
    "Construct DAG with DoWhy, map relationships, identify confounders",
    "Separate treatment period from pre-treatment learning period; baseline trajectories"],
    "Evidence": ["data/processed/panel.csv, results/week1_summary.json",
                 "reports/figures/w1_causal_dag.png + DoWhy adjustment set",
                 "reports/figures/w1_baseline_trajectories.png (train / validation / post)"],
    "Status": ["✅", "✅", "✅"]})
checklist
