# Causal Inference & Synthetic Control Engine for Algorithmic Pricing

Estimating the causal impact of pricing decisions using DAG-based causal modelling (DoWhy) and synthetic control methods.

## Project Overview
Algorithmic pricing changes are hard to evaluate because demand, competitor behaviour, region, and seasonality all move at the same time. This project builds a causal pipeline that:

1. Ingests regional market transaction data and competitor pricing telemetry
2. Maps assumed causal relationships and confounders with DAGs (DoWhy)
3. Separates pre-treatment learning periods from treatment periods to establish baselines
4. Builds a synthetic control to estimate the counterfactual and the treatment effect
5. Validates results with placebo tests and robustness checks

## Repository Structure
```
causal-pricing-engine/
├── data/
│   ├── raw/            # original datasets (not committed if large/sensitive)
│   └── processed/      # cleaned and merged data
├── notebooks/          # exploration and visualisation
├── src/                # pipeline code (one file per project week)
├── outputs/            # plots, tables, results
├── requirements.txt
└── README.md
```

## Project Plan
| Week | Focus | Status |
|------|-------|--------|
| 1 | Data ingestion, causal DAGs, pre/post split and baselines | In progress |
| 2 | Synthetic control construction and effect estimation | Planned |
| 3 | Placebo tests and robustness checks | Planned |
| 4 | Final report and packaging | Planned |

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Methods
- **Causal graphs:** DoWhy for DAG specification, identification, and confounder analysis
- **Synthetic control:** weighted donor-pool combination fitted on the pre-treatment period
- **Validation:** placebo-in-space tests, refutation checks, sensitivity analysis

## Author
Mohamed Aadhil M# zaalima-data-analytics-project
