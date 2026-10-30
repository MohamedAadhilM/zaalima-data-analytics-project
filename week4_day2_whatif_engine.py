# %% [markdown]
# # Week 4 · Day 2 — What-If Scenario Engine
# **Goal:** let a product manager ask "what if we changed prices by X% in this segment from week W?" and see revenue
# against the synthetic counterfactual.
#
# **How it works:** the effect for a segment (Causal Forest CATE, Week 3) was estimated at the *observed* price change
# `delta_obs`. For another price change `delta` we scale it assuming constant elasticity:
# `tau(delta) = tau_obs * ln(1+delta) / ln(1+delta_obs)`. Scenario revenue = counterfactual x exp(tau).
# This is an extrapolation and is only trustworthy near `delta_obs`.
#
# **Inputs:** `artifacts/summary.json`, `artifacts/scm_revenue_paths.csv`, `artifacts/cate_by_segment.csv` (Day 1)
# **Outputs:** `results/w4_whatif_table.csv`, `results/w4_scenario_engine_checks.csv`, `reports/figures/w4_whatif_example.png`

# %%
import _bootstrap  # noqa: F401
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpe.config import CFG
from cpe import scenario
from cpe.utils import setup_style, savefig

setup_style(); CFG.ensure_dirs()
pd.options.display.float_format = "{:,.3f}".format
A = CFG.artifact_dir
S = json.loads((A / "summary.json").read_text())
paths = pd.read_csv(A / "scm_revenue_paths.csv").set_index("week_idx")
seg = pd.read_csv(A / "cate_by_segment.csv").set_index("income_tercile")
T0, d_obs = S["t0"], S["delta_obs"]
cf = paths.counterfactual
print(f"Observed average price uplift: {d_obs:.1%} | go-live week {T0}")

# %% [markdown]
# ## 1. Engine checks (the engine must behave sensibly before anyone relies on it)

# %%
tau0 = seg.cate.iloc[0]
base = scenario.simulate_scenario(cf, tau0, d_obs, d_obs, T0)
checks = pd.DataFrame([
    ("No price change -> no revenue change", np.allclose(scenario.simulate_scenario(cf, tau0, 0.0, d_obs, T0).incremental, 0)),
    ("Observed uplift reproduces the estimated effect", np.isclose(base.attrs["tau"], tau0)),
    ("Weeks before go-live are untouched", np.allclose(base.loc[:T0 - 1, "incremental"], 0)),
    ("Bigger uplift -> bigger effect (positive-effect segment)",
     scenario.scale_effect(0.05, 0.15, d_obs) > scenario.scale_effect(0.05, 0.05, d_obs)),
    ("Phase-in is slower than instant roll-out",
     scenario.simulate_scenario(cf, 0.05, d_obs, d_obs, T0, ramp_weeks=8).incremental.sum()
     < scenario.simulate_scenario(cf, 0.05, d_obs, d_obs, T0).incremental.sum())],
    columns=["check", "passed"])
checks.to_csv(CFG.results_dir / "w4_scenario_engine_checks.csv", index=False)
print(checks.to_string(index=False))
assert checks.passed.all(), "scenario engine failed a sanity check"

# %% [markdown]
# ## 2. Scenario table: every segment x three price changes

# %%
rows = []
for name, r in seg.iterrows():
    for d in (0.05, d_obs, 0.15):
        sim = scenario.simulate_scenario(cf.loc[T0:], r.cate, d, d_obs, T0)
        rows.append({"segment": name, "price_uplift": d, "effect_log_pts": sim.attrs["tau"],
                     "revenue_lift_%": 100 * np.expm1(sim.attrs["tau"]),
                     "incremental_revenue": sim.incremental.sum(),
                     "within_observed_range": abs(d - d_obs) <= 0.05})
table = pd.DataFrame(rows)
table.to_csv(CFG.results_dir / "w4_whatif_table.csv", index=False)
print(table.round(3).to_string(index=False))

# %% [markdown]
# ## 3. Example: high-income segment at three price changes

# %%
fig, ax = plt.subplots(figsize=(8.5, 3.8))
seg_name = "high income" if "high income" in seg.index else seg.cate.idxmax()
for d, c in zip((0.05, d_obs, 0.15), ("#9ecae1", "#3182bd", "#08519c")):
    s = scenario.simulate_scenario(cf.loc[T0:], seg.loc[seg_name, "cate"], d, d_obs, T0)
    ax.plot(s.index, s.scenario, color=c, label=f"+{d:.0%} price")
ax.plot(cf.loc[T0:], "k--", label="counterfactual (no change)")
ax.legend(); ax.set_title(f"What-if: {seg_name} segment, three price changes"); ax.set_xlabel("week")
plt.tight_layout(); savefig(fig, "w4_whatif_example.png"); plt.show()
print("\nReminder: scenarios far from the observed uplift are extrapolations, not evidence.")
