"""Builds the Power BI star-schema tables (CSV) from the project's results.

Run after `python run_all.py`:   python powerbi/build_powerbi_data.py
Writes to powerbi/data/. Every table name below is a table in the Power BI model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpe.config import CFG  # noqa: E402

R, A, OUT = CFG.results_dir, CFG.artifact_dir, ROOT / "powerbi" / "data"
OUT.mkdir(parents=True, exist_ok=True)
T0, TRAIN_END = CFG.primary_adoption_week, CFG.train_end

panel = pd.read_csv(CFG.processed_dir / "panel.csv", parse_dates=["week_start"])
events = pd.read_csv(CFG.raw_dir / "pricing_events.csv")
pool = pd.read_csv(R / "w2_donor_pool.csv")
weights = pd.read_csv(R / "w2_scm_weights.csv")
paths = pd.read_csv(R / "w2_scm_revenue_paths.csv", parse_dates=["week_start"])
gaps = pd.read_csv(R / "w3_placebo_gaps.csv")
seg = pd.read_csv(R / "w3_cate_by_segment.csv")
S = json.loads((A / "summary.json").read_text())
focal = S["primary"]


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:22s} {len(df):6d} rows x {df.shape[1]:2d} cols")


# ---------------------------------------------------------------- DimDate
d = panel[panel.market_id == focal][["week_idx", "week_start"]].drop_duplicates().sort_values("week_idx")
d["year"] = d.week_start.dt.year
d["month_num"] = d.week_start.dt.month
d["month_name"] = d.week_start.dt.strftime("%b")
d["year_month"] = d.week_start.dt.strftime("%Y-%m")
d["quarter"] = "Q" + d.week_start.dt.quarter.astype(str)
d["focal_period"] = np.where(d.week_idx < TRAIN_END, "1 Pre-treatment: training",
                     np.where(d.week_idx < T0, "2 Pre-treatment: validation", "3 Post go-live"))
d["is_post_focal"] = d.week_idx >= T0
save(d, "DimDate")

# -------------------------------------------------------------- DimMarket
st = panel.drop_duplicates("market_id")[["market_id", "tier", "income_z", "urban_index", "log_pop", "comp_intensity",
                                         "base_price", "treated_unit"]].copy()
st["population"] = np.exp(st.log_pop).round(0)
st = st.merge(events.rename(columns={"go_live_week_idx": "go_live_week"}), on="market_id", how="left")
st["market_group"] = np.where(st.market_id == focal, "Focal market",
                      np.where(st.treated_unit == 1, "Other adopter", "Never treated"))
obs = panel[panel.dynamic_pricing == 1]                      # same terciles as the CATE analysis
cut = pd.qcut(obs.income_z, 3, labels=["low income", "mid income", "high income"])
mapping = obs.assign(income_segment=cut).drop_duplicates("market_id").set_index("market_id").income_segment.astype(str)
st["income_segment"] = st.market_id.map(mapping).fillna("not an adopter")
st = st.merge(pool[["market_id", "status", "reason", "max_neighbor_exposure"]].rename(
    columns={"status": "donor_status", "reason": "donor_reason"}), on="market_id", how="left")
st["donor_status"] = st.donor_status.fillna("n/a (treated)")
st = st.merge(weights.rename(columns={"weight": "scm_weight"}), on="market_id", how="left")
st["scm_weight"] = st.scm_weight.fillna(0.0)
st["is_focal"] = st.market_id == focal
save(st, "DimMarket")

# -------------------------------------------------------- FactMarketWeek
f = panel[["market_id", "week_idx", "week_start", "units_sold", "avg_price", "gross_revenue", "competitor_avg_price",
           "comp_gap", "dynamic_pricing", "event_time", "neighbor_adoption", "log_revenue"]].copy()
save(f, "FactMarketWeek")

# ------------------------------------------------------ FactCounterfactual
fc = paths[["week_idx", "week_start", "actual", "counterfactual", "incremental", "gap_log"]].copy()
fc["market_id"] = focal
fc["is_post"] = fc.week_idx >= T0
save(fc, "FactCounterfactual")

# ------------------------------------------------------------ FactPlacebo
fp = gaps.melt(id_vars="week_idx", var_name="market_id", value_name="gap_log")
fp["is_focal"] = fp.market_id == focal
fp["is_post"] = fp.week_idx >= T0
save(fp, "FactPlacebo")

# ----------------------------------------------------------- SegmentEffect
se = seg.rename(columns={"revenue_lift_%": "lift_pct"}).copy()
se["lift_pct"] = 100 * np.expm1(se.cate)
se["lift_low_pct"] = 100 * np.expm1(se.ci_low)
se["lift_high_pct"] = 100 * np.expm1(se.ci_high)
save(se, "SegmentEffect")
save(pd.read_csv(R / "w3_cate_by_market.csv"), "CateMarket")

# ------------------------------------------------------------ Estimators
ate = pd.read_csv(R / "w3_ate_table.csv").rename(columns={"revenue_lift_%": "lift_pct"})
ate["method_type"] = np.where(ate.estimator.str.startswith("Naive"), "Naive (biased)",
                      np.where(ate.estimator.str.startswith("Synthetic"), "Synthetic control", "Causal panel"))
ate["sort_order"] = range(1, len(ate) + 1)
save(ate, "Estimators")

# ---------------------------------------------------------------- Summary
flat = {k: v for k, v in S.items() if not isinstance(v, list)}
flat["dml_ci_low"], flat["dml_ci_high"] = S["dml_ci"]
save(pd.DataFrame([flat]), "Summary")

# --------------------------------------------------------- Robustness set
save(pd.read_csv(R / "w2_mspe_table.csv"), "MspeFit")
save(pd.read_csv(R / "w3_placebo_in_time.csv"), "PlaceboInTime")
save(pd.read_csv(R / "w3_leave_one_donor_out.csv"), "LeaveOneOut")
save(pd.read_csv(R / "w3_dowhy_refutations.csv"), "Refuters")
save(pd.read_csv(R / "w3_cv_folds.csv"), "CvFolds")
save(pd.read_csv(R / "w3_scm_rolling_cv.csv"), "RollingCv")
rv = pd.read_csv(R / "w3_robustness_value.csv")
save(rv, "RobustnessValue")
g = pd.read_csv(R / "w3_ovb_grid.csv", index_col=0)
ovb = g.reset_index().melt(id_vars=g.index.name or "R2_confounder_outcome", var_name="r2_treatment",
                           value_name="adjusted_theta")
ovb.columns = ["r2_outcome", "r2_treatment", "adjusted_theta"]
ovb["adjusted_lift_pct"] = 100 * np.expm1(ovb.adjusted_theta)
ovb["r2_outcome_pct"] = (100 * ovb.r2_outcome).round(0).astype(int).astype(str) + "%"
ovb["r2_treatment_pct"] = (100 * ovb.r2_treatment.astype(float)).round(0).astype(int).astype(str) + "%"
save(ovb, "OvbGrid")

# ---------------------------------------------------- What-if + checklists
save(pd.read_csv(R / "w4_whatif_table.csv"), "WhatIfGrid")
chk = pd.concat([
    pd.read_csv(R / "w2_midproject_checklist.csv").assign(stage="Week 2 mid-project review"),
    pd.read_csv(R / "w3_week3_checklist.csv").assign(stage="Week 3 robustness"),
    pd.read_csv(R / "w4_final_checklist.csv").assign(stage="Week 4 final review")]
)[["stage", "criterion", "status", "evidence"]]
save(chk, "ReviewChecklist")
dq = pd.Series(json.loads((CFG.processed_dir / "ingest_report.json").read_text())).rename("value").rename_axis("metric").reset_index()
dq["value"] = dq.value.astype(str)
save(dq, "DataQuality")

# ------------------------------------------- expected values (to verify DAX)
post = fc[fc.is_post]
exp = pd.DataFrame([
    ("Total revenue, all markets, all weeks", f["gross_revenue"].sum()),
    ("Focal actual revenue, post go-live", post.actual.sum()),
    ("Focal counterfactual revenue, post go-live", post.counterfactual.sum()),
    ("Focal incremental revenue, post go-live", post.incremental.sum()),
    ("Focal revenue lift % (post)", 100 * (post.actual.sum() / post.counterfactual.sum() - 1)),
    ("DML average lift %", 100 * np.expm1(S["dml_theta"])),
    ("Observed average price uplift %", 100 * S["delta_obs"]),
    ("Markets total", st.shape[0]), ("Treated markets", int(st.treated_unit.sum())),
    ("Donors kept", int((st.donor_status == "kept").sum())), ("Donors dropped", int((st.donor_status == "dropped").sum())),
    ("Placebo series (excl. focal)", fp[~fp.is_focal].market_id.nunique()),
    ("What-if: high-income lift % at observed uplift", float(100 * np.expm1(seg.set_index("income_tercile").loc["high income", "cate"]))),
], columns=["quantity", "expected_value"])
exp.to_csv(OUT.parent / "expected_values.csv", index=False)
print("\n", exp.round(3).to_string(index=False))
