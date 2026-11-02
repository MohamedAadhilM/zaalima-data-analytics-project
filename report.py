"""Executive causal analysis report generator (Markdown), built from the saved result CSVs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import CFG, Config


def _md(df: pd.DataFrame, fmt="{:,.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(
            ("-" if pd.isna(v) else fmt.format(v)) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def write_executive_report(cfg: Config = CFG):
    R, A = cfg.results_dir, cfg.artifact_dir
    s = json.loads((A / "summary.json").read_text())
    ate = pd.read_csv(R / "w3_ate_table.csv")
    seg = pd.read_csv(R / "w3_cate_by_segment.csv")
    pls = pd.read_csv(R / "w3_placebo_summary.csv").set_index("metric")["value"]
    loo = pd.read_csv(R / "w3_leave_one_donor_out.csv")
    ref = pd.read_csv(R / "w3_dowhy_refutations.csv")
    pool = pd.read_csv(R / "w2_donor_pool.csv")
    whatif = pd.read_csv(R / "w4_whatif_table.csv")
    adj = (R / "w1_estimand.txt").read_text().splitlines()[0].split(":", 1)[1].strip()
    n_treated = pd.read_csv(cfg.raw_dir / "pricing_events.csv").shape[0]

    seg["lift_%"] = 100 * np.expm1(seg.cate)
    seg["decision"] = np.where(seg.ci_low > 0, "Deploy", np.where(seg.ci_high < 0, "Do not deploy", "Hold / A-B test"))
    naive = float(ate.iloc[0].estimate_log_pts)
    causal = ate.iloc[2:6]
    best, worst = seg.loc[seg.cate.idxmax()], seg.loc[seg.cate.idxmin()]
    ate_show = ate[["estimator", "estimate_log_pts", "ci_low", "ci_high", "truth_log_pts", "bias_vs_truth"]]
    obs = whatif[whatif.within_observed_range & (whatif.price_uplift.round(3) == round(s["delta_obs"], 3))]
    p = cfg.report_dir / "executive_causal_report.md"

    txt = f"""# Executive Causal Analysis - Dynamic Pricing Revenue Lift

*Causal Inference & Synthetic Control Engine for Algorithmic Pricing - Project 1*

## 1. Headline

* Dynamic pricing raised revenue in focal market **{s['primary']}** by **{s['att_pct_scm']:+.1%}** against its synthetic
  counterfactual: **{s['cumulative_incremental']:,.0f}** units of incremental revenue over {s['post_weeks']} weeks
  (permutation p = {s['p_att']:.2f}, the smallest possible with {int(pls['placebos kept (comparable pre-fit)'])} placebos).
* Across all {n_treated} adopting markets, Double ML estimates **{np.expm1(s['dml_theta']):+.1%}**
  (95% CI {np.expm1(s['dml_ci'][0]):+.1%} to {np.expm1(s['dml_ci'][1]):+.1%}); an imputation estimator gives {np.expm1(s['bjs_theta']):+.1%}.
* The effect is **heterogeneous**: {best.income_tercile} markets {best['lift_%']:+.1f}%, {worst.income_tercile} markets {worst['lift_%']:+.1f}%.
* A naive treated-vs-untreated comparison reports {np.expm1(naive):+.0%}: adopters were larger, more urban markets, so
  A/B-style read-outs are heavily confounded.
* **Caution:** the robustness value is only {s['rv']:.1%} - a modest hidden confounder could erase the average effect.

![Actual vs counterfactual](figures/w4_actual_vs_counterfactual.png)

## 2. Method in one paragraph

A DoWhy-validated causal DAG identifies the effect by back-door adjustment (set: `{adj}`; the realised price is a mediator and is
deliberately not adjusted for). A spillover-screened donor pool ({int((pool.status == 'kept').sum())} markets kept,
{int((pool.status == 'dropped').sum())} dropped for network contamination) feeds an intercept-adjusted, ridge-penalised synthetic control
whose penalty is chosen on a blocked validation window. Double ML with market and week fixed effects, 26 controls, LightGBM nuisances,
market-grouped cross-fitting and market-clustered inference estimates the average effect; Causal Forest DML estimates segment effects.

## 3. Pre-treatment fit

| window | RMSPE |
|---|---|
| training (in-sample) | {s['rmspe_pre']:.4f} |
| blocked validation (out-of-sample) | {s['rmspe_val']:.4f} |
| post-treatment / pre ratio | {s['ratio']:.2f} |

![Placebo in space](figures/w3_placebo_in_space.png)

## 4. Average effect - estimator triangulation (log-points)

{_md(ate_show)}

*`truth` is known only because the data are simulated; on real data the evidence is agreement across independent methods.*

## 5. Where to price dynamically - effect by income segment

{_md(seg[['income_tercile', 'n_markets', 'cate', 'ci_low', 'ci_high', 'lift_%', 'decision']])}

![CATE](figures/w3_cate_segments.png)

## 6. Recommended pricing strategy

1. **Roll out** where the interval excludes zero ({', '.join(seg[seg.decision == 'Deploy'].income_tercile) or 'none'}).
2. **Hold or run a protected A/B holdout** where it spans zero ({', '.join(seg[seg.decision == 'Hold / A-B test'].income_tercile) or 'none'}).
3. **Keep the uplift near the observed {s['delta_obs']:.0%}.** The simulator scales effects with constant elasticity, which is unsupported far outside that range.
4. **Monitor** realised against counterfactual revenue with the dashboard after each roll-out wave.

What-if results at the observed uplift:

{_md(obs[['segment', 'price_uplift', 'revenue_lift_%', 'incremental_revenue']])}

## 7. Robustness

* **Placebo-in-space:** permutation p = {pls['permutation p-value (|gap|)']:.2f}.
* **Leave-one-donor-out:** the gap stays between {loo.att_log.min():+.3f} and {loo.att_log.max():+.3f} log-points.
* **Cinelli-Hazlett robustness value:** {s['rv']:.1%} of residual variance in both adoption and revenue erases the effect; {s['rv_alpha']:.1%} makes it insignificant.
* **DoWhy refuters:**

{_md(ref[['refuter', 'original', 'new_effect']], '{:,.4f}')}

![Sensitivity](figures/w3_sensitivity_grid.png)

## 8. Limitations

* A single-market synthetic control is noisy; the portfolio-level DML is more precise but relies on conditional ignorability.
* The low robustness value means hidden confounding cannot be ruled out.
* Segment intervals ignore clustering by market and rest on a few markets per segment; tier B and C each have a single adopter.
* On this simulated panel the causal estimators sit {abs(causal.bias_vs_truth.mean()):.3f} log-points below the known truth on average, so they understate the lift.
* Effects outside the observed price change are extrapolations.
"""
    p.write_text(txt, encoding="utf-8")
    return p
