# Final Project Review - Causal Inference & Synthetic Control Engine for Algorithmic Pricing

## Final checklist
| criterion                                       | status   | evidence                                                                           |
|:------------------------------------------------|:---------|:-----------------------------------------------------------------------------------|
| Causal pricing simulation dashboard deployed    | PASS     | 7 of 7 headless dashboard tests passed; launch: streamlit run app/streamlit_app.py |
| Sensitivity and robustness checks validated     | PASS     | 8 of 9 Week 3 checks pass; flagged: Sensitivity: hidden-confounder strength        |
| Final algorithmic pricing lift report delivered | PASS     | reports/executive_causal_report.md (768 words)                                     |
| All promised deliverables present               | PASS     | 22 of 22 files                                                                     |

## Results at a glance
| Question | Answer |
|---|---|
| Focal-market lift (synthetic control, R35) | **+3.5%**, 305,225 incremental revenue over 34 weeks |
| Average lift, all adopters (Double ML) | **+2.6%** (95% CI +0.6% to +4.7%) |
| Pre-treatment fit | RMSPE 0.021 in-sample, 0.032 held-out |
| Single-market permutation p-value | 0.05 |
| Hidden-confounder robustness value | 4.0% (low) |

## Items flagged for review (be upfront about these)
- Week 2: **2c. No drift on held-out pre-treatment weeks** - validation-window average gap -0.0127
- Week 3: **Sensitivity: hidden-confounder strength** - RV 4.0%

## How to reproduce everything
```bash
pip install -r requirements.txt
python run_all.py            # weeks 1-4, all days, tests, dashboard smoke test
streamlit run app/streamlit_app.py
```
