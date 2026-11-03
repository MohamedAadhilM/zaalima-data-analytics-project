# Causal Inference & Synthetic Control Engine for Algorithmic Pricing

An end-to-end causal-inference engine that estimates the **true revenue impact (causal lift) of dynamic
pricing** in an interconnected marketplace, where classic A/B testing breaks down because of network
spillovers and non-random adoption.

> **Project 1 · Enterprise-Level Data Analytics · Zaalima Development**

## Why this is more than a notebook

| Concern | How the engine handles it |
|---|---|
| Non-random adoption (confounding) | DoWhy-validated DAG → back-door adjustment; DML with a 26-feature control set |
| Network spillovers (SUTVA) | Spillover-aware donor screening; neighbour-exposure control |
| Short, noisy pre-periods | Intercept-adjusted, **ridge-penalised** synthetic control, penalty chosen on a *blocked out-of-sample* window |
| Panel dependence | Market-grouped cross-fitting, market-clustered standard errors |
| Staggered adoption bias | Heterogeneity-robust imputation estimator alongside TWFE / DML |
| Heterogeneity | Causal Forest DML → segment-level CATE → deployment decisions |
| "Is it just noise / hidden confounding?" | Placebo-in-space/time, leave-one-donor-out, Cinelli–Hazlett robustness value, DoWhy refuters, grouped K-fold + rolling-origin CV |
| Decision support | Streamlit what-if simulator + auto-generated executive report |

## Repository layout

```
├── run_all.py                    # one command: weeks 1-4, every day, tests, run log
├── weeks/                        # one file per project day (.py: runs in IDLE, terminal or as Jupyter cells)
│   ├── week1_data_dag_baseline.py              Week 1 (Days 1-5): ingestion · DoWhy DAG · baselines
│   ├── week2_day1_donor_pool.py                Spillover-aware donor pool
│   ├── week2_day2_synthetic_control.py         Ridge-penalised synthetic control
│   ├── week2_day3_mspe_validation.py           MSPE / RMSPE validation
│   ├── week2_day4_double_ml.py                 Double ML with high-dimensional controls
│   ├── week2_day5_midproject_review.py         Mid-project review
│   ├── week3_day1_ate_estimators.py            ATE triangulation
│   ├── week3_day2_cate_segments.py             CATE by segment (Causal Forest)
│   ├── week3_day3_scm_inference.py             Placebo tests, leave-one-donor-out
│   ├── week3_day4_sensitivity.py               Robustness value, refuters
│   ├── week3_day5_cross_validation.py          Cross-validation + Week 3 summary
│   ├── week4_day1_counterfactual_diagnostics.py  Actual vs counterfactual + dashboard artefacts
│   ├── week4_day2_whatif_engine.py             What-if scenario engine
│   ├── week4_day3_dashboard_smoke_test.py      Dashboard headless test
│   ├── week4_day4_executive_report.py          Executive report
│   └── week4_day5_final_review.py              Final checklist & deliverables index
├── src/cpe/                      # tested, importable engine
│   └── data.py dag.py donors.py synthetic_control.py dml.py sensitivity.py scenario.py report.py utils.py config.py
├── app/streamlit_app.py          # interactive counterfactual simulator
├── tests/                        # pytest suite (run in CI)
├── data/{raw,processed}/         # generated on first run (replace raw CSVs with real extracts)
├── results/ artifacts/ reports/  # CSV/JSON outputs · dashboard inputs · figures + executive & final review reports
└── docs/COMMIT_PLAN.md           # daily commit schedule
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --fresh        # rebuild everything from scratch (about a minute)
streamlit run app/streamlit_app.py
pytest -q
```
Or run any single day, in order, e.g. `python weeks/week2_day3_mspe_validation.py`.

## Using real data
Replace the CSVs in `data/raw/` with extracts following the schema documented at the top of `src/cpe/data.py`
(`market_transactions`, `competitor_telemetry`, `pricing_events`, `market_attributes`, `market_adjacency`) and delete the
`simulate_raw(...)` line in Week 1. `_ground_truth.csv` is simulation-only and is used purely to validate the estimators.

## Reading the results honestly
The bundled data are **simulated with a known effect**, so every estimator is benchmarked against the truth.
Naive comparisons are off by an order of magnitude; the causal estimators land in the right neighbourhood but
sit slightly below truth (adoption timing and spillovers) — the report says so and bounds it with sensitivity analysis.
Single-market synthetic control is inherently noisy; portfolio-level DML is more precise. Re-seed in `config.py` to see sampling variability.
