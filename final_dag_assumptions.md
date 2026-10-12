# Day 5 — Final DAG Documentation and Assumptions
| | |
|---|---|
| **Member** | Member 5 — Mohamed Aadhil M |
| **Branch** | `feature/causal-dag` |
| **Task (Detailed Plan, commit 29)** | Finalize DAG documentation and explain the rationale for major causal edges and adjustment variables |
| **Expected deliverable** | Final DAG + assumptions document |
| **Commit message** | `docs: finalize causal DAG and confounder assumptions` |

## 1. Summary
We estimate the causal effect of **dynamic pricing** on **revenue**. Adoption was not random, so a DAG records our assumptions and DoWhy identifies what to control for.

## 2. Final DAG
12 nodes, 19 arrows. Figure: `reports/figures/w1_causal_dag.png`.

## 3. Rationale for the major edges
| Edge | Rationale |
|---|---|
| `urban_index`, `log_pop`, `comp_intensity → dynamic_pricing` | Structural characteristics decide who adopts first |
| `urban_index`, `log_pop → log_revenue` | The same characteristics drive revenue |
| `local_demand → comp_gap`, `→ log_revenue` | Hidden demand moves competitor prices and revenue |
| `comp_gap → comp_gap_4w → dynamic_pricing` | Adoption is triggered by the smoothed competitor price |
| `neighbor_adoption → log_revenue` | Neighbours' adoption spills over onto demand |
| `dynamic_pricing → log_price → log_revenue` | The mechanism of the effect (mediator) |
| `dynamic_pricing → log_revenue` | The effect of interest |

## 4. Adjustment variables
| Adjust for | Do not adjust for |
|---|---|
| `urban_index`, `log_pop`, `comp_intensity`, `comp_gap` or `comp_gap_4w` | `log_price` (mediator), `local_demand` (unobservable) |

Optional precision controls: `income_z`, `neighbor_adoption`.

## 5. Assumptions and how they are handled
| Assumption | Handling |
|---|---|
| Unconfoundedness | Sensitivity analysis in Week 3 |
| No interference (SUTVA) | Spillover control now; donor screening in Week 2 |
| Overlap | Never-treated markets exist as comparisons |
| No anticipation | Pre-trend diagnostics (Member 6's baseline work) |
| Consistency | One definition of "dynamic pricing" for all markets |

## 6. Limitations
- The DAG is our assumption, not proven fact.
- The argument that competitor price blocks the hidden demand path cannot be tested from data.
- The arrows need domain review when real business data is used.

## 7. Handoff to Week 2
Adjustment set and treatment/outcome definitions go to the Synthetic Control and DML work.

## 8. How to explain it in a meeting
"I mapped what causes what. Large, urban, competitive markets adopted dynamic pricing first and also earn more, so comparing them directly misleads us. DoWhy read the map and told us what to control for. We deliberately do not control for the realised price, because that is how the pricing works. One factor, local demand, cannot be measured, so we state that and test it in Week 3."

## 9. Submit
- `docs/causal/day5_final_dag_assumptions.md`
- Tracker: Day 5 → Confirmed + link, then merge the pull request after review.
