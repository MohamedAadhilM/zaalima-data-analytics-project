# %% [markdown]
# # Week 4 · Day 3 — Streamlit Dashboard: Build & Smoke Test
# **Goal:** deliver the interactive "Causal Pricing Simulator" (`app/streamlit_app.py`) and prove it works without
# opening a browser, using Streamlit's headless `AppTest` harness.
#
# The dashboard has six tabs: Overview · Counterfactual · What-if simulator · Robustness · Segments (CATE) · Method.
# It only reads the artefacts exported on Day 1 (no model refits at runtime).
#
# **Launch it yourself:** `streamlit run app/streamlit_app.py`
#
# **Inputs:** `app/streamlit_app.py`, `artifacts/*` (Day 1)
# **Outputs:** `results/w4_dashboard_smoke_test.csv`, `results/w4_dashboard_metrics.csv`

# %%
import _bootstrap
import pandas as pd
from streamlit.testing.v1 import AppTest

from cpe.config import CFG

CFG.ensure_dirs()
APP = _bootstrap.ROOT / "app" / "streamlit_app.py"
assert (CFG.artifact_dir / "summary.json").exists(), "run Week 4 Day 1 first"

# %% [markdown]
# ## 1. Default render: every tab loads without an exception

# %%
at = AppTest.from_file(str(APP), default_timeout=90).run()
metrics = pd.DataFrame([(m.label, m.value) for m in at.metric], columns=["metric", "value"])
metrics.to_csv(CFG.results_dir / "w4_dashboard_metrics.csv", index=False)
print("Exceptions:", [str(e.value) for e in at.exception] or "none")
print("Tabs:", [t.label for t in at.tabs])
print(metrics.to_string(index=False))

# %% [markdown]
# ## 2. Interaction tests: the simulator reacts correctly to the user

# %%
def effect(a):
    return next(m.value for m in a.metric if m.label == "Effect on revenue")

default_effect = effect(at)
at.slider[0].set_value(0.0).run()                       # price uplift -> 0%
zero_effect = effect(at)
at.slider[0].set_value(0.15).run()                      # large uplift
big_effect = effect(at)
at.selectbox[0].select("low income").run()              # low-income segment
low_effect = effect(at)

def num(s):
    return float(s.replace("%", "").replace("+", ""))

tests = [
    ("App renders with no exception", len(at.exception) == 0, "-"),
    ("Six tabs present", len(at.tabs) == 6, f"{len(at.tabs)} tabs"),
    ("Overview shows four headline metrics", sum(m.label in {"Revenue lift (synthetic control)", "Cumulative incremental revenue",
                                                             "DML average effect", "Placebo permutation p-value"} for m in at.metric) == 4, "-"),
    ("0% price change gives a 0% effect", abs(num(zero_effect)) < 0.05, f"effect = {zero_effect}"),
    ("Changing the price slider changes the effect", default_effect != big_effect, f"{default_effect} -> {big_effect}"),
    ("Switching segment changes the effect", big_effect != low_effect, f"{big_effect} -> {low_effect}"),
    ("No exception after interactions", len(at.exception) == 0, "-"),
]
out = pd.DataFrame(tests, columns=["test", "passed", "detail"])
out.to_csv(CFG.results_dir / "w4_dashboard_smoke_test.csv", index=False)
print(out.to_string(index=False))
assert out.passed.all(), "dashboard smoke test failed"
print("\nAll dashboard tests passed.")
