# Mid-Project Review — End of Week 2

**Focal market:** R35 (go-live week 70) · **Donor pool:** 19 kept, 9 dropped for spillover risk

## Checklist
| criterion                                                     | status   | evidence                                                     |
|:--------------------------------------------------------------|:---------|:-------------------------------------------------------------|
| 1. Causal DAG mapped and validated                            | PASS     | 7 confounder rows, 1 mediator; back-door estimand identified |
| 2a. Synthetic control matches pre-treatment path (in-sample)  | PASS     | RMSPE 0.0209                                                 |
| 2b. Synthetic control matches on weeks it never saw           | PASS     | RMSPE 0.0316                                                 |
| 2c. No drift on held-out pre-treatment weeks                  | REVIEW   | validation-window average gap -0.0127                        |
| 2d. Weights are spread across donors (not one-donor reliance) | PASS     | 15 donors > 1%                                               |
| 3a. SCM effect estimator functioning                          | PASS     | gap +0.0349 log-pts (+3.5%)                                  |
| 3b. DML effect estimator functioning                          | PASS     | θ +0.0260 (CI +0.006 to +0.045)                              |
| 3c. DML has usable overlap                                    | PASS     | R² 0.24                                                      |
| 3d. The two estimators agree on direction                     | PASS     | SCM +0.035 vs DML +0.026                                     |

## Headline numbers
- Synthetic-control fit: RMSPE 0.021 in-sample, 0.032 on held-out pre-treatment weeks, 0.038 after go-live (post/pre ratio 1.83).
- Focal-market lift (synthetic control): **+3.5%**.
- Average lift across adopters (Double ML): **+2.6%** (95% CI +0.6% to +4.7%).

## Item flagged for review
- 2c: on the held-out pre-treatment weeks the focal market ran -0.013 log-points relative to its synthetic twin, so the post-treatment lift of +3.5% may be slightly understated or overstated by a similar amount. Week 3 sensitivity checks quantify this.

## What is NOT yet established
- Statistical significance of the single-market estimate (placebo tests are Week 3).
- Whether the effect differs by market segment (CATE, Week 3).
- Sensitivity to hidden confounding (Week 3).
- The two estimates differ because they cover different markets; neither is "the" answer yet.
