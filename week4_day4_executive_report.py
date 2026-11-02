# %% [markdown]
# # Week 4 · Day 4 — Executive Causal Analysis Report
# **Goal:** translate the ATE and CATE results into a plain-language report with actionable pricing strategy,
# written for decision-makers (not statisticians), including what we are *not* sure about.
#
# **Inputs:** `artifacts/summary.json`, `results/w1_estimand.txt`, `results/w2_donor_pool.csv`,
# `results/w3_*.csv`, `results/w4_whatif_table.csv`
# **Outputs:** `reports/executive_causal_report.md`, `reports/figures/*` (referenced by the report)

# %%
import _bootstrap  # noqa: F401
from cpe.config import CFG
from cpe.report import write_executive_report

CFG.ensure_dirs()
path = write_executive_report(CFG)
text = path.read_text(encoding="utf-8")
print(f"Report written: {path}  ({len(text.split())} words)")

# %% [markdown]
# ## Check: every figure the report links to exists

# %%
import re
figs = re.findall(r"\]\((figures/[^)]+)\)", text)
missing = [f for f in figs if not (CFG.report_dir / f).exists()]
print("Figures referenced:", figs)
assert not missing, f"missing figures: {missing}"
print("\n--- Report preview ---\n")
print(text[:2500])
