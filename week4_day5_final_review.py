# %% [markdown]
# # Week 4 · Day 5 — Final Project Review & Deliverables
# **Goal:** confirm the three final-review criteria of the project plan, verify every deliverable exists, and
# produce the overall project summary.
#
# 1. Causal pricing simulation dashboard deployed
# 2. Sensitivity and robustness checks validated
# 3. Final algorithmic pricing lift report delivered
#
# **Inputs:** everything produced in Weeks 1-4
# **Outputs:** `results/w4_final_checklist.csv`, `results/w4_deliverables_index.csv`, `reports/final_project_review.md`

# %%
import _bootstrap  # noqa: F401
import json
import numpy as np
import pandas as pd

from cpe.config import CFG

CFG.ensure_dirs()
R, A, REP = CFG.results_dir, CFG.artifact_dir, CFG.report_dir
S = json.loads((A / "summary.json").read_text())

# %% [markdown]
# ## 1. Deliverables index (every file the project promises)

# %%
ROOT = _bootstrap.ROOT
deliverables = [
    ("Week 1", "Cleaned, merged panel", "data/processed/panel.csv"),
    ("Week 1", "Data-quality / ingestion report", "data/processed/ingest_report.json"),
    ("Week 1", "Causal DAG figure", "reports/figures/w1_causal_dag.png"),
    ("Week 1", "Confounder / mediator table", "results/w1_confounders.csv"),
    ("Week 1", "DoWhy identified estimand", "results/w1_estimand.txt"),
    ("Week 1", "Baseline trajectories figure", "reports/figures/w1_baseline_trajectories.png"),
    ("Week 2", "Donor pool", "results/w2_donor_pool.csv"),
    ("Week 2", "Synthetic control weights", "results/w2_scm_weights.csv"),
    ("Week 2", "MSPE validation table", "results/w2_mspe_table.csv"),
    ("Week 2", "DML results", "results/w2_dml_results.csv"),
    ("Week 2", "Mid-project review", "reports/w2_midproject_review.md"),
    ("Week 3", "ATE estimator table", "results/w3_ate_table.csv"),
    ("Week 3", "CATE by segment", "results/w3_cate_by_segment.csv"),
    ("Week 3", "Placebo / robustness results", "results/w3_placebo_summary.csv"),
    ("Week 3", "Sensitivity (robustness value)", "results/w3_robustness_value.csv"),
    ("Week 3", "Cross-validation results", "results/w3_cv_folds.csv"),
    ("Week 3", "Week 3 summary", "reports/w3_week3_summary.md"),
    ("Week 4", "Counterfactual diagnostic figure", "reports/figures/w4_actual_vs_counterfactual.png"),
    ("Week 4", "What-if scenario table", "results/w4_whatif_table.csv"),
    ("Week 4", "Dashboard app", "app/streamlit_app.py"),
    ("Week 4", "Dashboard smoke-test results", "results/w4_dashboard_smoke_test.csv"),
    ("Week 4", "Executive causal analysis report", "reports/executive_causal_report.md"),
]
idx = pd.DataFrame(deliverables, columns=["week", "deliverable", "path"])
idx["exists"] = [(ROOT / p).exists() for p in idx.path]
idx.to_csv(R / "w4_deliverables_index.csv", index=False)
print(idx.to_string(index=False))
print(f"\n{int(idx.exists.sum())} of {len(idx)} deliverables present")

# %% [markdown]
# ## 2. Final-review checklist from the project plan

# %%
smoke = pd.read_csv(R / "w4_dashboard_smoke_test.csv")
w2 = pd.read_csv(R / "w2_midproject_checklist.csv")
w3 = pd.read_csv(R / "w3_week3_checklist.csv")
report_words = len((REP / "executive_causal_report.md").read_text(encoding="utf-8").split())

final = pd.DataFrame([
    ("Causal pricing simulation dashboard deployed", bool(smoke.passed.all()),
     f"{int(smoke.passed.sum())} of {len(smoke)} headless dashboard tests passed; launch: streamlit run app/streamlit_app.py"),
    ("Sensitivity and robustness checks validated", bool(len(w3) > 0 and w3.passed.mean() >= 0.8),
     f"{int(w3.passed.sum())} of {len(w3)} Week 3 checks pass; flagged: " +
     (", ".join(w3.loc[~w3.passed, "criterion"]) or "none")),
    ("Final algorithmic pricing lift report delivered", bool(report_words > 500),
     f"reports/executive_causal_report.md ({report_words} words)"),
    ("All promised deliverables present", bool(idx.exists.all()), f"{int(idx.exists.sum())} of {len(idx)} files"),
], columns=["criterion", "passed", "evidence"])
final["status"] = np.where(final.passed, "PASS", "REVIEW")
final.to_csv(R / "w4_final_checklist.csv", index=False)
print(final[["criterion", "status", "evidence"]].to_string(index=False))

# %% [markdown]
# ## 3. Overall project summary

# %%
flag_lines = []
for name, df in (("Week 2", w2), ("Week 3", w3)):
    for r in df[~df.passed].itertuples():
        flag_lines.append(f"- {name}: **{r.criterion}** - {r.evidence}")
note = f"""# Final Project Review - Causal Inference & Synthetic Control Engine for Algorithmic Pricing

## Final checklist
{final[['criterion', 'status', 'evidence']].to_markdown(index=False)}

## Results at a glance
| Question | Answer |
|---|---|
| Focal-market lift (synthetic control, {S['primary']}) | **{S['att_pct_scm']:+.1%}**, {S['cumulative_incremental']:,.0f} incremental revenue over {S['post_weeks']} weeks |
| Average lift, all adopters (Double ML) | **{np.expm1(S['dml_theta']):+.1%}** (95% CI {np.expm1(S['dml_ci'][0]):+.1%} to {np.expm1(S['dml_ci'][1]):+.1%}) |
| Pre-treatment fit | RMSPE {S['rmspe_pre']:.3f} in-sample, {S['rmspe_val']:.3f} held-out |
| Single-market permutation p-value | {S['p_att']:.2f} |
| Hidden-confounder robustness value | {S['rv']:.1%} (low) |

## Items flagged for review (be upfront about these)
{chr(10).join(flag_lines) or '- none'}

## How to reproduce everything
```bash
pip install -r requirements.txt
python run_all.py            # weeks 1-4, all days, tests, dashboard smoke test
streamlit run app/streamlit_app.py
```
"""
(REP / "final_project_review.md").write_text(note, encoding="utf-8")
print(note)
