# %% [markdown]
# # Week 3 · Day 2 — Conditional Average Treatment Effects (CATE)
# **Goal:** find *where* dynamic pricing works. A Causal Forest DML lets the effect vary with market income,
# urbanicity, size and competitive intensity, answering "which markets should adopt?"
#
# **Inputs:** `data/processed/panel.csv`; optional `data/raw/_ground_truth.csv` (simulation-only truth)
# **Outputs:** `results/w3_cate_by_segment.csv`, `results/w3_cate_by_tier.csv`, `results/w3_cate_by_market.csv`,
# `reports/figures/w3_cate_segments.png`

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
gt_path = CFG.raw_dir / "_ground_truth.csv"
gt = pd.read_csv(gt_path) if gt_path.exists() else pd.DataFrame(
    {"market_id": panel.market_id.unique(), "true_tau_log": np.nan})
has_truth = gt_path.exists()

# %% [markdown]
# ## 1. Fit the Causal Forest and predict effects for every post-adoption market-week

# %%
prep = dml.prepare(panel)
cf = dml.causal_forest(prep, n_folds=CFG.n_folds)
ct = dml.cate_table(cf, prep, gt)
print(f"{len(ct)} post-adoption observations across {ct.market_id.nunique()} adopting markets")
print(f"Average CATE: {ct.cate.mean():+.4f} log-pts ({100 * np.expm1(ct.cate.mean()):+.1f}% revenue)")

# %% [markdown]
# ## 2. Effects by income segment and market-size tier

# %%
ct["income_tercile"] = pd.qcut(ct.income_z, 3, labels=["low income", "mid income", "high income"])
seg = (ct.groupby("income_tercile", observed=True)
         .agg(n_markets=("market_id", "nunique"), cate=("cate", "mean"), ci_low=("lo", "mean"),
              ci_high=("hi", "mean"), truth=("true_tau_log", "mean")))
seg["revenue_lift_%"] = 100 * np.expm1(seg.cate)
seg["decision"] = np.where(seg.ci_low > 0, "Deploy", np.where(seg.ci_high < 0, "Do not deploy", "Hold / A-B test"))
tier = ct.groupby("tier").agg(n_markets=("market_id", "nunique"), cate=("cate", "mean"), truth=("true_tau_log", "mean"))
tier["revenue_lift_%"] = 100 * np.expm1(tier.cate)
seg.reset_index().to_csv(CFG.results_dir / "w3_cate_by_segment.csv", index=False)
tier.reset_index().to_csv(CFG.results_dir / "w3_cate_by_tier.csv", index=False)
print(seg.round(4).to_string()); print(); print(tier.round(4).to_string())

# %% [markdown]
# ## 3. Market-level effects and picture
# Each dot is one adopting market. If the truth is available, dots near the dashed line mean the forest recovered it.

# %%
bym = ct.groupby("market_id").agg(cate=("cate", "mean"), truth=("true_tau_log", "mean"),
                                  income_z=("income_z", "first"), tier=("tier", "first"))
bym.reset_index().to_csv(CFG.results_dir / "w3_cate_by_market.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
if has_truth:
    ax[0].scatter(bym.truth, bym.cate, c=bym.income_z, cmap="viridis", s=60, edgecolor="k")
    lim = [bym[["truth", "cate"]].min().min() - 0.01, bym[["truth", "cate"]].max().max() + 0.01]
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xlabel("true market effect")
    ax[0].set_title(f"CATE recovery (corr = {bym[['cate', 'truth']].corr().iloc[0, 1]:.2f}; colour = income)")
else:
    ax[0].scatter(bym.income_z, bym.cate, s=60, edgecolor="k"); ax[0].set_xlabel("market income (z-score)")
    ax[0].set_title("Estimated effect vs market income")
ax[0].set_ylabel("estimated CATE")
x = np.arange(len(seg))
ax[1].bar(x - 0.18 if has_truth else x, seg.cate, 0.36, label="estimated CATE", color="#1f77b4")
if has_truth:
    ax[1].bar(x + 0.18, seg.truth, 0.36, label="truth", color="#bbbbbb")
ax[1].errorbar(x - 0.18 if has_truth else x, seg.cate, yerr=[seg.cate - seg.ci_low, seg.ci_high - seg.cate], fmt="none", c="k")
ax[1].set_xticks(x); ax[1].set_xticklabels(seg.index); ax[1].axhline(0, c="k", lw=0.8); ax[1].legend()
ax[1].set_title("Effect by income segment")
plt.tight_layout(); savefig(fig, "w3_cate_segments.png"); plt.show()

# %% [markdown]
# ## 4. Takeaway

# %%
for name, r in seg.iterrows():
    print(f"{name:12s}: {r['revenue_lift_%']:+.1f}% revenue lift  ->  {r['decision']}")
