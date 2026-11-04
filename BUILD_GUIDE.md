# Power BI Dashboard — Build Guide (Causal Pricing Simulator)

**What you get:** a 6-page interactive report on top of the causal-inference results (synthetic control, Double ML,
CATE, robustness). Everything is a **ready-made data model** (20 CSV tables), **reference numbers checked in Python**
(`expected_values.csv`), **DAX measures**, a **Power Query loader** and a **theme**. Build time: about 60–90 minutes.

> **Be aware:** I could not run Power BI Desktop in my environment, so I could not generate or open a `.pbix`
> file myself. The tables were built and checked in Python, the DAX was written carefully but **has not been executed in
> Power BI** — use the checks in section 4 to confirm every number before you present it.

## 1. Prerequisites
- Power BI Desktop (Windows, free). On Mac: use the Power BI web service (upload the CSVs; you can create measures there).
- Folder `powerbi/data/` (20 CSVs). Regenerate any time with `python powerbi/build_powerbi_data.py` after `python run_all.py`.

## 2. Load and model (20 min)
1. **Get data:** follow `PowerQuery_load.m` (a `fnLoad` function + one query per table, folder set by a `DataFolder` parameter).
   Close & Apply.
2. **Model view → relationships** (all single direction, one-to-many):

| From (one) | To (many) |
|---|---|
| DimDate[week_idx] | FactMarketWeek[week_idx], FactCounterfactual[week_idx], FactPlacebo[week_idx] |
| DimMarket[market_id] | FactMarketWeek[market_id], FactPlacebo[market_id] |
| SegmentEffect[income_tercile] | DimMarket[income_segment] |

   Leave Estimators, Summary, MspeFit, PlaceboInTime, LeaveOneOut, Refuters, CvFolds, RollingCv, RobustnessValue,
   OvbGrid, WhatIfGrid, ReviewChecklist, DataQuality, CateMarket **unrelated** (stand-alone tables).
3. **Parameters:** Modeling → New table → paste the three `GENERATESERIES` lines (ParamUplift, ParamGoLive, ParamRamp);
   rename their value columns to `Uplift`, `Week`, `Weeks`.
4. **Measures:** create table `_Measures` (Home → Enter data, one column, then hide it). Paste every measure from
   `DAX_measures.dax`, setting the display folder named in each section header.
5. **Field parameter:** paste the `Metric Picker` table (or Modeling → New parameter → Fields: add Total Revenue, Total Units,
   Avg Price, Revenue Index).
6. **Sorting/format:** sort `DimDate[month_name]` by `month_num`; set `DimDate[focal_period]` summarise = none; hide key
   columns; format `% ` measures as percentage, revenue as `#,0`.
7. **Theme:** View → Themes → Browse for themes → `theme_causal_pricing.json`. Canvas size 1280 × 720 (16:9).

## 3. Pages (each visual: type · fields · formatting)
Add a **page-navigation bar** to every page (Insert → Buttons → Navigator → Page navigator) and a shared slicer panel
(sync slicers: View → Sync slicers) for *DimMarket[market_group]* and *DimDate[focal_period]*.

### Page 1 — Executive Overview
| Visual | Fields | Notes |
|---|---|---|
| Title text box | `[Headline]` (fx → field value) | Dynamic headline |
| 4 × Card | `[Revenue Lift %]`, `[Post Incremental]`, `[DML Lift %]`, `[Permutation p-value]` | Callout font from theme |
| Line + area combo | X `DimDate[week_start]`; lines `[Actual Revenue]`, `[Counterfactual Revenue]`; columns `[Incremental Revenue]` (secondary axis) | Red actual, blue dashed counterfactual; add a **constant line** at go-live (Analytics → X-axis constant = go-live date) |
| Clustered bar + error bars | Y `Estimators[estimator]` (sort by `sort_order`); X `[Estimator Lift %]`; Analytics → **Error bars**: upper `[Estimator High %]`, lower `[Estimator Low %]`; marker `[Estimator Truth %]` as a second series | Colour by `Estimators[method_type]` (naive = red). Naive row is huge — set axis max ≈ 12 |
| Text box | `[Honesty Note]` | Keeps the limits visible |

### Page 2 — Counterfactual & Model Fit
| Visual | Fields |
|---|---|
| Line chart | X `DimDate[week_idx]`, Y `[Gap (log-points)]`; shaded by `DimDate[focal_period]` (use a ribbon/column on secondary axis) |
| Line chart (cumulative) | X `week_idx`, Y `[Cumulative Incremental]` |
| Bar chart | Y `DimMarket[market_id]`, X `DimMarket[scm_weight]`, visual filter `scm_weight > 0.005`, sorted desc |
| Cards | `[RMSPE Pre]`, `[RMSPE Validation]`, `[RMSPE Ratio (post/pre)]` |
| Table | `MspeFit[window]`, `weeks`, `RMSPE`, `MSPE` |
| Card | `[Pre-period Gap (validation)]` — amber if |value| > 0.01 (conditional formatting) |

### Page 3 — Where to Price (CATE)
| Visual | Fields | Notes |
|---|---|---|
| Slicer (tile) | `SegmentEffect[income_tercile]` | Drives What-if page too |
| Clustered column + error bars | X `SegmentEffect[income_tercile]`; Y `[Segment Lift %]`; error bars `[Segment Lift High %]` / `[Segment Lift Low %]` | Title ← `[Segment Title]` |
| Matrix | Rows `income_tercile`; values `n_markets`, `[Segment Lift %]`, `[Segment Decision]` | Font colour ← `[Decision Colour]` (Conditional formatting → Field value) |
| Scatter | X `CateMarket[income_z]`, Y `[Market CATE %]`, details `CateMarket[market_id]`, size `DimMarket[population]` | Shows effect rising with income |
| Text | "Tier B and C each have one adopter — ignore tier rows" | Honest caveat |

### Page 4 — What-If Simulator (the interactive centrepiece)
| Visual | Fields |
|---|---|
| Slicer (slider) | `ParamUplift[Uplift]` (Between style, single value) — price change |
| Slicer (slider) | `ParamGoLive[Week]` — go-live week |
| Slicer (dropdown) | `ParamRamp[Weeks]` — phase-in |
| Slicer (tile) | `SegmentEffect[income_tercile]` — segment |
| Title | `[Scenario Title]` |
| Line chart | X `DimDate[week_start]`; `[Counterfactual Revenue]` (grey dashed), `[Scenario Revenue]` (green), `[Actual Revenue]` (red, faint) |
| Cards | `[Scenario Effect %]`, `[Scenario Incremental (from go-live)]`, `[Selected CATE]` |
| Line | X `week_idx`, Y `[Scenario Cumulative]` |
| Text (red) | `[Extrapolation Warning]` |
| Table | `WhatIfGrid` (segment × 5% / observed / 15%) for comparison |

**Bookmarks:** create "Reset to observed" (uplift = observed, go-live = week 70, ramp 0) and attach to a Reset button.

### Page 5 — Robustness & Sensitivity
| Visual | Fields |
|---|---|
| Line chart (placebo band) | X `week_idx`; `[Focal Placebo Gap]` (thick red), `[Placebo P10]`, `[Placebo P90]`, `[Placebo Median]` (grey) — focal gap must leave the band after go-live |
| Clustered bar | `LeaveOneOut[dropped_donor]` × `att_pct` |
| Column | `PlaceboInTime[shift_weeks]` × `placebo_att_log` with constant line at the real gap |
| Gauge | Value `[Robustness Value (RV)]`, target `[RV Gauge Target]` (0.05), max 0.15 — currently below target: say so |
| Matrix (heat-map) | Rows `OvbGrid[r2_outcome_pct]`, columns `OvbGrid[r2_treatment_pct]`, values `adjusted_lift_pct`; background colour scale red–white–green centred on 0 |
| Table | `Refuters[refuter]`, `original`, `new_effect` |
| Column | `CvFolds[fold]` × `theta_fold` (effect stability across held-out folds) |
| Matrix | `ReviewChecklist` stage/criterion/status/evidence; status font ← `[Check Colour]`; card `[Checks Passed]` of `[Checks Total]` |

### Page 6 — Market Explorer & Data Quality (drill-through)
| Visual | Fields |
|---|---|
| Line chart (small multiples) | X `week_idx`, Y `[Revenue Index]`, legend `DimMarket[market_group]` |
| Line chart with **field parameter** | Y `Metric Picker` fields — user switches Revenue / Units / Price / Index |
| Table | `DimMarket`: market, group, tier, income segment, go-live week, donor status, donor reason, `max_neighbor_exposure` |
| Cards | `DataQuality` rows (duplicates removed, missing prices filled, outliers fixed) |
| Matrix | Spillover: `[Neighbour Adoption Share]` by `week_idx` for never-treated markets |

**Drill-through page "Market detail":** add a new page, drag `DimMarket[market_id]` to Drill-through, show revenue, price,
competitor price gap, `[Neighbour Adoption Share]` by week. Right-click any market row on Page 6 → Drill through.
**Tooltip page:** a small page (Page settings → Tooltip) with a sparkline of `[Total Revenue]` by week; assign it to the
scatter on Page 3.

## 4. Verify before presenting (all must match `expected_values.csv`)
| Check | Expected |
|---|---|
| `[Post Incremental]` | 305,225 |
| `[Revenue Lift %]` | +3.54% |
| `[DML Lift %]` | +2.63% |
| `[Observed Uplift]` | 10.36% |
| Segment lift (high income) | +8.18% |
| Scenario at observed uplift, high-income, go-live 70, ramp 0 | effect +8.2% |
| Scenario at 0% uplift | exactly 0 incremental |
| High-income at +15% uplift, go-live 70, ramp 0 | effect +11.8%, incremental ≈ value in `expected_values.csv` |
| Donors kept / dropped | 19 / 9 |
| Placebo series (excl. focal) | 19 |

## 5. Publish and refresh
- Publish to the Power BI Service; schedule refresh by re-running the Python steps and replacing the CSVs
  (or put them in OneDrive/SharePoint and point `DataFolder` there).
- For **real data**, replace `data/raw` CSVs, run `python run_all.py --fresh` then `python powerbi/build_powerbi_data.py`.

## 6. What to say honestly when presenting
- The data are simulated; the "truth" markers are only available because of that.
- Single-market p = 0.05 is the smallest possible with 19 placebos; the hidden-confounder robustness value is low (4%).
- The what-if simulator extrapolates with constant elasticity — trust it only near the observed price change.

## 7. Troubleshooting
- *Numbers differ from expected_values:* check the relationships table; check `DataFolder` ends with `\`.
- *Dates not recognised:* the loader types `yyyy-MM-dd` text as Date — make sure Culture is en-US.
- *Placebo band blank:* `FactPlacebo[is_focal]` must load as True/False (logical), not text.
- *Scenario revenue blank:* the three Param tables must exist and have the renamed columns.
