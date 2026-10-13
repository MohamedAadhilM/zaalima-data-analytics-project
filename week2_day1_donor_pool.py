# %% [markdown]
# # Week 2 · Day 1 — Spillover-Aware Donor Pool
# **Goal:** choose the markets that can safely serve as the "no dynamic pricing" comparison group.
# A donor is only valid if it was **not itself affected** by the treatment. In a connected marketplace,
# neighbours of adopting markets gain demand, which would contaminate the counterfactual.
#
# **Inputs:** `data/processed/panel.csv`, `data/raw/market_adjacency.csv`, `data/raw/pricing_events.csv`
# **Outputs:** `results/w2_donor_pool.csv`, `reports/figures/w2_donor_screening.png`

# %%
import _bootstrap  # noqa: F401
import pandas as pd
import matplotlib.pyplot as plt

from cpe.config import CFG
from cpe.donors import donor_table
from cpe.utils import setup_style, savefig

setup_style(); CFG.ensure_dirs()
panel = pd.read_csv(CFG.processed_dir / "panel.csv")
adjacency = pd.read_csv(CFG.raw_dir / "market_adjacency.csv")
events = pd.read_csv(CFG.raw_dir / "pricing_events.csv")
print(f"Panel: {panel.market_id.nunique()} markets x {panel.week_idx.nunique()} weeks")

# %% [markdown]
# ## 1. Focal (treated) market
# The focal market is the one that went live in week 70 (the adoption week set in `config.py`).

# %%
T0 = CFG.primary_adoption_week
primary = events.loc[events.go_live_week_idx == T0, "market_id"].iloc[0]
print(f"Focal market: {primary} | go-live week: {T0}")
print(f"Treated markets: {len(events)} | never-treated candidates: {(panel.groupby('market_id').treated_unit.first() == 0).sum()}")

# %% [markdown]
# ## 2. Screen the donor pool
# Rule: drop a never-treated market if it is **adjacent to the focal market** or if **at least 50% of its
# neighbours** went on to adopt dynamic pricing.

# %%
tbl = donor_table(panel, primary, adjacency, exposure_cap=0.5)
tbl.to_csv(CFG.results_dir / "w2_donor_pool.csv", index=False)
print(tbl.status.value_counts().to_string())
print("\nDropped donors:")
print(tbl[tbl.status == "dropped"].to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(11, 3.8))
t = tbl.sort_values("max_neighbor_exposure", ascending=False)
ax.bar(t.market_id, t.max_neighbor_exposure, color=t.status.map({"kept": "#2ca02c", "dropped": "#d62728"}))
ax.axhline(0.5, color="k", ls="--", lw=1); ax.text(len(t) - 0.5, 0.52, "exposure cap 50%", ha="right")
ax.set_ylabel("share of neighbours that adopted"); ax.tick_params(axis="x", rotation=90)
ax.set_title("Donor screening: green = kept, red = dropped (spillover risk)")
plt.tight_layout(); savefig(fig, "w2_donor_screening.png"); plt.show()

# %% [markdown]
# ## 3. Result
# The kept markets form the donor pool used by the synthetic control (Day 2).

# %%
kept = tbl.loc[tbl.status == "kept", "market_id"].tolist()
print(f"Donor pool: {len(kept)} markets kept, {(tbl.status == 'dropped').sum()} dropped")
print(kept)
