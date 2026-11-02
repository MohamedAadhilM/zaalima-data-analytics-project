#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "Organizing repositories files into week/day folders..."

mkdir -p \
  code \
  source \
  output \
  week_1/day_1 \
  week_1/day_2 \
  week_1/day_3 \
  week_1/day_4 \
  week_1/day_5 \
  week_2/day_1 \
  week_2/day_2 \
  week_2/day_3 \
  week_2/day_4 \
  week_2/day_5 \
  week_3/day_1 \
  week_3/day_2 \
  week_3/day_3 \
  week_3/day_4 \
  week_3/day_5 \
  week_4/day_1 \
  week_4/day_2 \
  week_4/day_3 \
  week_4/day_4 \
  week_4/day_5 \
  week_1/output \
  week_2/output \
  week_3/output \
  week_4/output

# Move reusable code files
for f in __init__.py _bootstrap.py config.py data.py dag.py dml.py donors.py report.py scenario.py sensitivity.py streamlit_app.py synthetic_control.py utils.py; do
  if [ -f "$f" ]; then
    mv "$f" code/ 2>/dev/null || true
  fi
done

# Week 1
for f in week1_data_dag_baseline.py week1_data_dag_baseline.ipynb; do
  if [ -f "$f" ]; then
    mv "$f" week_1/day_1/ 2>/dev/null || true
  fi
done

# Week 2
for f in week2_day1_donor_pool.py week2_day2_synthetic_control.py week2_day3_mspe_validation.py week2_day4_double_ml.py week2_day5_midproject_review.py; do
  if [ -f "$f" ]; then
    case "$f" in
      week2_day1_donor_pool.py) mv "$f" week_2/day_1/ ;;
      week2_day2_synthetic_control.py) mv "$f" week_2/day_2/ ;;
      week2_day3_mspe_validation.py) mv "$f" week_2/day_3/ ;;
      week2_day4_double_ml.py) mv "$f" week_2/day_4/ ;;
      week2_day5_midproject_review.py) mv "$f" week_2/day_5/ ;;
    esac
  fi
done

# Week 3
for f in week3_day1_ate_estimators.py week3_day2_cate_segments.py week3_day3_scm_inference.py week3_day4_sensitivity.py week3_day5_cross_validation.py; do
  if [ -f "$f" ]; then
    case "$f" in
      week3_day1_ate_estimators.py) mv "$f" week_3/day_1/ ;;
      week3_day2_cate_segments.py) mv "$f" week_3/day_2/ ;;
      week3_day3_scm_inference.py) mv "$f" week_3/day_3/ ;;
      week3_day4_sensitivity.py) mv "$f" week_3/day_4/ ;;
      week3_day5_cross_validation.py) mv "$f" week_3/day_5/ ;;
    esac
  fi
done

# Week 4
for f in week4_day1_counterfactual_diagnostics.py week4_day2_whatif_engine.py week4_day3_dashboard_smoke_test.py week4_day4_executive_report.py; do
  if [ -f "$f" ]; then
    case "$f" in
      week4_day1_counterfactual_diagnostics.py) mv "$f" week_4/day_1/ ;;
      week4_day2_whatif_engine.py) mv "$f" week_4/day_2/ ;;
      week4_day3_dashboard_smoke_test.py) mv "$f" week_4/day_3/ ;;
      week4_day4_executive_report.py) mv "$f" week_4/day_4/ ;;
    esac
  fi
done

# Source data files
for f in panel.csv market_adjacency.csv pricing_events.csv _ground_truth.csv placebo_gaps.csv cate_by_segment.csv market_adjacency.csv; do
  if [ -f "$f" ]; then
    mv "$f" source/ 2>/dev/null || true
  fi
done

# Results / output files with patterns
for f in *.csv *.png *.json *.md *.txt; do
  if [ -f "$f" ]; then
    case "$f" in
      README.md|requirements.txt|.gitignore) ;;
      *.png|*.csv|*.json|*.md|*.txt)
        if [[ "$f" == w1_* ]]; then
          mv "$f" week_1/output/ 2>/dev/null || true
        elif [[ "$f" == w2_* ]]; then
          mv "$f" week_2/output/ 2>/dev/null || true
        elif [[ "$f" == w3_* ]]; then
          mv "$f" week_3/output/ 2>/dev/null || true
        elif [[ "$f" == w4_* ]]; then
          mv "$f" week_4/output/ 2>/dev/null || true
        else
          mv "$f" output/ 2>/dev/null || true
        fi
        ;;
    esac
  fi
done

echo "Folder organization complete."
echo "Structure created:"
find "$REPO_ROOT" -maxdepth 2 -type d | sort
