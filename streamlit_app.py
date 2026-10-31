"""Causal Pricing Simulator — Streamlit dashboard.

Run:  streamlit run app/streamlit_app.py
Reads pre-computed artefacts from ./artifacts (produced by weeks/week4_dashboard_report.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpe import scenario  # noqa: E402

ART = ROOT / "artifacts"
st.set_page_config(page_title="Causal Pricing Simulator", page_icon="📈", layout="wide")


@st.cache_data
def load():
    s = json.loads((ART / "summary.json").read_text())
    return (s, pd.read_csv(ART / "scm_revenue_paths.csv"), pd.read_csv(ART / "placebo_gaps.csv"),
            pd.read_csv(ART / "cate_by_segment.csv"), pd.read_csv(ART / "scm_weights.csv"))


if not (ART / "summary.json").exists():
    st.error("Artefacts not found. Run `python weeks/week4_dashboard_report.py` first.")
    st.stop()

S, paths, gaps, seg, weights = load()
T0, prim = S["t0"], S["primary"]
BLUE, RED, GREEN = "#1f77b4", "#d62728", "#2ca02c"

st.title("📈 Causal Pricing Simulator")
st.caption(f"Synthetic Control + Double ML engine · focal market **{prim}** · dynamic pricing live from week {T0}")

tab_over, tab_cf, tab_sim, tab_rob, tab_seg, tab_meth = st.tabs(
    ["Overview", "Counterfactual", "What-if simulator", "Robustness", "Segments (CATE)", "Method"])

# ---------------------------------------------------------------- overview
with tab_over:
    c = st.columns(4)
    c[0].metric("Revenue lift (synthetic control)", f"{S['att_pct_scm']:+.1%}")
    c[1].metric("Cumulative incremental revenue", f"{S['cumulative_incremental']:,.0f}")
    c[2].metric("DML average effect", f"{np.expm1(S['dml_theta']):+.1%}",
                help=f"95% CI on log scale: [{S['dml_ci'][0]:+.3f}, {S['dml_ci'][1]:+.3f}]")
    c[3].metric("Placebo permutation p-value", f"{S['p_att']:.2f}")
    st.markdown(
        f"Two independent estimators agree on direction and order of magnitude: synthetic control "
        f"({S['att_pct_scm']:+.1%} on {prim}) and DML across all adopters ({np.expm1(S['dml_theta']):+.1%}); "
        f"an imputation estimator gives {np.expm1(S['bjs_theta']):+.1%}. An unobserved confounder would need to explain "
        f"≈{S['rv']:.1%} of the residual variance of **both** price adoption and revenue to nullify the DML effect.")

# ---------------------------------------------------------- counterfactual
with tab_cf:
    fig = go.Figure()
    fig.add_scatter(x=paths.week_idx, y=paths.actual, name="Actual", line=dict(color=RED, width=3))
    fig.add_scatter(x=paths.week_idx, y=paths.counterfactual, name="Synthetic counterfactual",
                    line=dict(color=BLUE, dash="dash", width=3))
    post = paths[paths.week_idx >= T0]
    fig.add_scatter(x=post.week_idx.tolist() + post.week_idx.tolist()[::-1],
                    y=post.actual.tolist() + post.counterfactual.tolist()[::-1], fill="toself",
                    fillcolor="rgba(44,160,44,.25)", line=dict(width=0), name="Incremental revenue")
    fig.add_vline(x=T0, line_dash="dot")
    fig.update_layout(height=430, title="Actual revenue vs synthetic counterfactual", xaxis_title="week",
                      yaxis_title="weekly revenue", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    a.metric("Pre-treatment RMSPE (in-sample)", f"{S['rmspe_pre']:.3f}")
    b.metric("Blocked out-of-sample RMSPE", f"{S['rmspe_val']:.3f}")
    wf = go.Figure(go.Bar(x=weights.weight, y=weights.market_id, orientation="h", marker_color=BLUE))
    wf.update_layout(height=380, title="Donor-market weights in the synthetic control",
                     yaxis=dict(autorange="reversed"))
    st.plotly_chart(wf, width="stretch")

# ------------------------------------------------------------- simulator
with tab_sim:
    st.subheader("Counterfactual 'what-if' pricing scenarios")
    l, r = st.columns([1, 2])
    with l:
        segname = st.selectbox("Market segment", seg.income_tercile.tolist(),
                               index=seg.income_tercile.tolist().index("high income")
                               if "high income" in seg.income_tercile.tolist() else 0)
        delta = st.slider("Price uplift", 0.0, 0.20, float(round(S["delta_obs"], 2)), 0.01, format="%.2f")
        start = st.slider("Go-live week", int(paths.week_idx.min()), int(paths.week_idx.max()) - 4, T0)
        ramp = st.slider("Phase-in (weeks)", 0, 12, 0)
        horizon = paths.week_idx.max()
        base = paths.set_index("week_idx").counterfactual
        tau_obs = float(seg.set_index("income_tercile").loc[segname, "cate"])
        sim = scenario.simulate_scenario(base, tau_obs, delta, S["delta_obs"], start, ramp)
        win = sim.loc[start:]
        st.metric("Effect on revenue", f"{np.expm1(sim.attrs['tau']):+.1%}")
        st.metric("Incremental revenue (window)", f"{win.incremental.sum():,.0f}")
        if abs(delta - S["delta_obs"]) > 0.05:
            st.warning("Scenario is far from the observed price change — extrapolation is unsupported by data.")
    with r:
        f = go.Figure()
        f.add_scatter(x=sim.index, y=sim.counterfactual, name="Counterfactual (no change)",
                      line=dict(color="grey", dash="dash"))
        f.add_scatter(x=sim.index, y=sim.scenario, name="Scenario", line=dict(color=GREEN, width=3))
        f.add_vline(x=start, line_dash="dot")
        f.update_layout(height=420, xaxis_title="week", yaxis_title="weekly revenue", legend=dict(orientation="h"))
        st.plotly_chart(f, width="stretch")
    st.caption("Assumption: constant elasticity, τ(δ) = τ_obs · ln(1+δ)/ln(1+δ_obs), scaled from the estimated segment CATE.")

# ------------------------------------------------------------ robustness
with tab_rob:
    g = gaps.set_index("week_idx")
    f = go.Figure()
    for c in g.columns:
        if c != prim:
            f.add_scatter(x=g.index, y=g[c], line=dict(color="rgba(150,150,150,.4)", width=1), showlegend=False)
    f.add_scatter(x=g.index, y=g[prim], name=prim, line=dict(color=RED, width=3))
    f.add_vline(x=T0, line_dash="dot")
    f.update_layout(height=430, yaxis_range=[-0.25, 0.25], title="Placebo-in-space: focal gap vs donor placebos",
                    xaxis_title="week", yaxis_title="gap (log-points)")
    st.plotly_chart(f, width="stretch")
    st.markdown(f"Permutation p-value (|ATT|) **{S['p_att']:.2f}**, RMSPE-ratio p **{S['p_ratio']:.2f}**; "
                f"robustness value RV = **{S['rv']:.1%}** (RV at 5% significance = {S['rv_alpha']:.1%}).")

# --------------------------------------------------------------- segments
with tab_seg:
    f = go.Figure(go.Bar(x=seg.income_tercile, y=100 * np.expm1(seg.cate), marker_color=BLUE,
                         error_y=dict(type="data", symmetric=False,
                                      array=100 * (np.expm1(seg.ci_high) - np.expm1(seg.cate)),
                                      arrayminus=100 * (np.expm1(seg.cate) - np.expm1(seg.ci_low)))))
    f.update_layout(height=400, title="Estimated revenue lift by income segment (%, Causal Forest DML)",
                    yaxis_title="lift %")
    st.plotly_chart(f, width="stretch")
    show = seg.assign(**{"lift %": 100 * np.expm1(seg.cate)})[["income_tercile", "n_markets", "cate", "ci_low", "ci_high", "lift %"]]
    st.dataframe(show.round(4), hide_index=True, width="stretch")
    st.info("Deploy where the confidence interval excludes zero; keep guard-rails (or A/B holdouts) where it does not.")

# ----------------------------------------------------------------- method
with tab_meth:
    st.markdown("""
**Identification.** Back-door adjustment from a DoWhy-validated DAG; hidden local demand is blocked through the competitor-price signal.
**Synthetic control.** Intercept-adjusted, ridge-penalised convex weights on a spillover-screened donor pool; penalty chosen by blocked out-of-sample validation.
**Double ML.** Market & week fixed effects, 26 engineered controls, LightGBM nuisances, market-grouped cross-fitting, market-clustered standard errors; Causal Forest DML for CATE.
**Robustness.** Placebo-in-space/time, leave-one-donor-out, Cinelli–Hazlett robustness value, DoWhy refuters, grouped K-fold and rolling-origin CV.
""")
