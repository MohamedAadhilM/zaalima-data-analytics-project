# Executive Causal Analysis - Dynamic Pricing Revenue Lift

*Causal Inference & Synthetic Control Engine for Algorithmic Pricing - Project 1*

## 1. Headline

* Dynamic pricing raised revenue in focal market **R35** by **+3.5%** against its synthetic
  counterfactual: **305,225** units of incremental revenue over 34 weeks
  (permutation p = 0.05, the smallest possible with 19 placebos).
* Across all 12 adopting markets, Double ML estimates **+2.6%**
  (95% CI +0.6% to +4.7%); an imputation estimator gives +3.4%.
* The effect is **heterogeneous**: high income markets +8.2%, low income markets -0.8%.
* A naive treated-vs-untreated comparison reports +81%: adopters were larger, more urban markets, so
  A/B-style read-outs are heavily confounded.
* **Caution:** the robustness value is only 4.0% - a modest hidden confounder could erase the average effect.

![Actual vs counterfactual](figures/w4_actual_vs_counterfactual.png)

## 2. Method in one paragraph

A DoWhy-validated causal DAG identifies the effect by back-door adjustment (set: `comp_gap, comp_intensity, log_pop, urban_index`; the realised price is a mediator and is
deliberately not adjusted for). A spillover-screened donor pool (19 markets kept,
9 dropped for network contamination) feeds an intercept-adjusted, ridge-penalised synthetic control
whose penalty is chosen on a blocked validation window. Double ML with market and week fixed effects, 26 controls, LightGBM nuisances,
market-grouped cross-fitting and market-clustered inference estimates the average effect; Causal Forest DML estimates segment effects.

## 3. Pre-treatment fit

| window | RMSPE |
|---|---|
| training (in-sample) | 0.0209 |
| blocked validation (out-of-sample) | 0.0316 |
| post-treatment / pre ratio | 1.83 |

![Placebo in space](figures/w3_placebo_in_space.png)

## 4. Average effect - estimator triangulation (log-points)

| estimator | estimate_log_pts | ci_low | ci_high | truth_log_pts | bias_vs_truth |
|---|---|---|---|---|---|
| Naive: treated vs never-treated | 0.591 | - | - | 0.047 | 0.544 |
| Naive: before vs after (treated) | 0.051 | - | - | 0.047 | 0.004 |
| Two-way fixed effects | 0.031 | 0.011 | 0.051 | 0.047 | -0.016 |
| DML-PLR (LightGBM, clustered SE) | 0.026 | 0.006 | 0.045 | 0.047 | -0.021 |
| EconML LinearDML | 0.026 | 0.023 | 0.029 | 0.047 | -0.021 |
| Imputation (BJS, heterogeneity-robust) | 0.033 | 0.014 | 0.052 | 0.047 | -0.013 |
| Synthetic control: R35 only | 0.035 | - | - | 0.060 | -0.026 |

*`truth` is known only because the data are simulated; on real data the evidence is agreement across independent methods.*

## 5. Where to price dynamically - effect by income segment

| income_tercile | n_markets | cate | ci_low | ci_high | lift_% | decision |
|---|---|---|---|---|---|---|
| low income | 4 | -0.008 | -0.018 | 0.002 | -0.782 | Hold / A-B test |
| mid income | 5 | 0.035 | 0.023 | 0.047 | 3.533 | Deploy |
| high income | 3 | 0.079 | 0.069 | 0.088 | 8.177 | Deploy |

![CATE](figures/w3_cate_segments.png)

## 6. Recommended pricing strategy

1. **Roll out** where the interval excludes zero (mid income, high income).
2. **Hold or run a protected A/B holdout** where it spans zero (low income).
3. **Keep the uplift near the observed 10%.** The simulator scales effects with constant elasticity, which is unsupported far outside that range.
4. **Monitor** realised against counterfactual revenue with the dashboard after each roll-out wave.

What-if results at the observed uplift:

| segment | price_uplift | revenue_lift_% | incremental_revenue |
|---|---|---|---|
| low income | 0.104 | -0.782 | -67,398.817 |
| mid income | 0.104 | 3.533 | 304,650.789 |
| high income | 0.104 | 8.177 | 705,154.616 |

## 7. Robustness

* **Placebo-in-space:** permutation p = 0.05.
* **Leave-one-donor-out:** the gap stays between +0.033 and +0.037 log-points.
* **Cinelli-Hazlett robustness value:** 4.0% of residual variance in both adoption and revenue erases the effect; 1.0% makes it insignificant.
* **DoWhy refuters:**

| refuter | original | new_effect |
|---|---|---|
| baseline (linear back-door) | 0.0258 | 0.0258 |
| placebo_treatment_refuter | 0.0258 | -0.0001 |
| random_common_cause | 0.0258 | 0.0258 |
| data_subset_refuter | 0.0258 | 0.0256 |
| add_unobserved_common_cause | 0.0258 | 0.0144 |

![Sensitivity](figures/w3_sensitivity_grid.png)

## 8. Limitations

* A single-market synthetic control is noisy; the portfolio-level DML is more precise but relies on conditional ignorability.
* The low robustness value means hidden confounding cannot be ruled out.
* Segment intervals ignore clustering by market and rest on a few markets per segment; tier B and C each have a single adopter.
* On this simulated panel the causal estimators sit 0.018 log-points below the known truth on average, so they understate the lift.
* Effects outside the observed price change are extrapolations.
