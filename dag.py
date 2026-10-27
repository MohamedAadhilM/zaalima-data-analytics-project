"""Causal DAG for the dynamic-pricing problem + DoWhy identification.

Node names are the panel column names so the same graph can be handed straight to DoWhy.
`local_demand` is an *unobserved* node (observed="no" in GML) - DoWhy must identify the
effect without it, which is only possible because `comp_gap_4w` blocks the back-door path.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

TREATMENT = "dynamic_pricing"
OUTCOME = "log_revenue"

EDGES = [
    # structural market characteristics -> adoption decision and demand
    ("urban_index", TREATMENT), ("urban_index", OUTCOME),
    ("log_pop", TREATMENT), ("log_pop", OUTCOME),
    ("comp_intensity", TREATMENT), ("comp_intensity", "comp_gap"),
    ("income_z", OUTCOME),
    # competitor behaviour: observed telemetry driven by hidden local demand
    ("local_demand", "comp_gap"), ("local_demand", OUTCOME),
    ("comp_gap", "comp_gap_4w"), ("comp_gap_4w", TREATMENT), ("comp_gap", OUTCOME),
    # calendar
    ("seasonality", OUTCOME), ("seasonality", "comp_gap"),
    # network spillover: neighbours' adoption lifts our demand
    ("urban_index", "neighbor_adoption"), ("neighbor_adoption", OUTCOME),
    # mechanism: dynamic pricing works THROUGH the realised price (mediator - never adjust for it)
    (TREATMENT, "log_price"), ("log_price", OUTCOME),
    # the causal effect of interest
    (TREATMENT, OUTCOME),
]
UNOBSERVED = {"local_demand"}

POS = {  # hand-tuned layout for the figure
    "income_z": (0, 4), "log_pop": (0, 3), "urban_index": (0, 2), "comp_intensity": (0, 1),
    "seasonality": (0, 0), "neighbor_adoption": (2, 5), "local_demand": (2, -1),
    "comp_gap": (2, 1), "comp_gap_4w": (3.3, 1.8), TREATMENT: (4.6, 2.6), OUTCOME: (6.4, 2.6),
    "log_price": (5.5, 4.0),
}


def build_dag() -> nx.DiGraph:
    g = nx.DiGraph(EDGES)
    assert nx.is_directed_acyclic_graph(g), "graph must be acyclic"
    return g


def to_gml(g: nx.DiGraph) -> str:
    """GML string understood by DoWhy, with unobserved nodes flagged."""
    lines = ["graph [", "  directed 1"]
    idx = {n: i for i, n in enumerate(g.nodes)}
    for n, i in idx.items():
        obs = '"no"' if n in UNOBSERVED else '"yes"'
        lines.append(f'  node [ id {i} label "{n}" observed {obs} ]')
    for a, b in g.edges:
        lines.append(f"  edge [ source {idx[a]} target {idx[b]} ]")
    lines.append("]")
    return "\n".join(lines)


def plot_dag(g: nx.DiGraph, path=None):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    col = []
    for n in g.nodes:
        col.append("#d62728" if n == TREATMENT else "#1f77b4" if n == OUTCOME
                   else "#bbbbbb" if n in UNOBSERVED else "#9ecae1")
    nx.draw_networkx_nodes(g, POS, node_color=col, node_size=3400, edgecolors="k", ax=ax)
    nx.draw_networkx_nodes(g, POS, nodelist=list(UNOBSERVED), node_color="#dddddd",
                           node_size=3400, node_shape="s", edgecolors="k", ax=ax)
    nx.draw_networkx_labels(g, POS, font_size=7, ax=ax)
    nx.draw_networkx_edges(g, POS, arrows=True, arrowsize=14, node_size=3400,
                           connectionstyle="arc3,rad=0.08", edge_color="#555", ax=ax)
    nx.draw_networkx_edges(g, POS, edgelist=[(TREATMENT, OUTCOME)], width=3, edge_color="#d62728",
                           arrowsize=20, node_size=3400, ax=ax)
    ax.set_title("Assumed causal DAG - dynamic pricing -> revenue "
                 "(red = treatment, blue = outcome, grey square = unobserved)", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig


def identify(df: pd.DataFrame, g: nx.DiGraph | None = None):
    """Run DoWhy identification; returns (model, estimand, adjustment_set)."""
    from dowhy import CausalModel
    g = g or build_dag()
    cols = [n for n in g.nodes if n not in UNOBSERVED]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CausalModel(data=df[cols], treatment=TREATMENT, outcome=OUTCOME,
                            graph=to_gml(g), effect_modifiers=[])
        est = model.identify_effect(proceed_when_unidentifiable=True)
    bd = est.get_backdoor_variables()
    return model, est, sorted(bd)


def confounder_report(g: nx.DiGraph) -> pd.DataFrame:
    """Classify every variable by its causal role relative to treatment -> outcome.

    * confounder : causes treatment AND outcome, and is not downstream of treatment  -> ADJUST
    * mediator   : downstream of treatment and upstream of outcome                    -> DO NOT adjust
    * outcome-only / other                                                            -> optional precision control
    """
    desc_t = nx.descendants(g, TREATMENT)
    anc_t, anc_y = nx.ancestors(g, TREATMENT), nx.ancestors(g, OUTCOME)
    rows = []
    for n in g.nodes:
        if n in (TREATMENT, OUTCOME):
            continue
        if n in desc_t and n in anc_y:
            role = "mediator (do not adjust)"
        elif n in anc_t and n in anc_y:
            role = "confounder (adjust)" if n not in UNOBSERVED else "confounder (UNOBSERVED - cannot adjust)"
        elif n in anc_t:
            role = "treatment-only cause (instrument-like)"
        elif n in anc_y:
            role = "outcome-only cause (precision control)"
        else:
            role = "other"
        rows.append({"variable": n, "observed": n not in UNOBSERVED, "role": role,
                     "ancestor_of_treatment": n in anc_t, "ancestor_of_outcome": n in anc_y,
                     "descendant_of_treatment": n in desc_t})
    order = {"confounder (adjust)": 0, "confounder (UNOBSERVED - cannot adjust)": 1, "mediator (do not adjust)": 2}
    out = pd.DataFrame(rows)
    return out.sort_values("role", key=lambda c: c.map(order).fillna(2)).reset_index(drop=True)
