# Week 3 Summary — Causal Impact, Sensitivity and Cross-Validation

## Checklist
| criterion                               | status   | evidence                                              |
|:----------------------------------------|:---------|:------------------------------------------------------|
| Quantify impact: ATE                    | PASS     | causal ATE +0.026 to +0.033 vs naive +0.591           |
| Quantify impact: CATE                   | PASS     | low income -0.8%; mid income +3.5%; high income +8.2% |
| Sensitivity: placebo-in-time            | PASS     | fake gaps -0.013 to +0.007                            |
| Sensitivity: leave-one-donor-out        | PASS     | gap +0.033 to +0.037                                  |
| Sensitivity: permutation test           | PASS     | p = 0.05 (19 placebos)                                |
| Sensitivity: placebo-treatment refuter  | PASS     | -0.0001 vs +0.0254                                    |
| Sensitivity: hidden-confounder strength | REVIEW   | RV 4.0%                                               |
| Cross-validation: DML folds             | PASS     | 4 of 5 folds positive, sd 0.025                       |
| Cross-validation: SCM rolling origin    | PASS     | error 0.029 vs gap 0.035                              |

## What we can say
- Causal estimators agree that dynamic pricing raised revenue, by roughly 2.6% to 3.4%, while the naive comparison reports 81%.
- The effect is not uniform: low income -0.8% (Hold / A-B test); mid income +3.5% (Deploy); high income +8.2% (Deploy).

## What we cannot say
- The robustness value is 4.0%: a modest hidden confounder could erase the average effect.
- The single-market p-value is the smallest possible with 19 placebos, so it is suggestive, not conclusive.
- Cross-validation only weakly supports effect heterogeneity: a model with market-trait effects beat a constant effect in 2 of 5 held-out folds, and the effect estimate varies across folds (sd 0.025).
- Segment intervals ignore clustering by market and rest on 3-5 markets per segment, so treat them as optimistic.
