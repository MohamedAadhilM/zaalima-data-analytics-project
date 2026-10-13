"""Spillover-aware donor-pool screening (Week 2, Day 1)."""
from __future__ import annotations

import pandas as pd


def donor_table(panel: pd.DataFrame, primary: str, adjacency: pd.DataFrame,
                exposure_cap: float = 0.5) -> pd.DataFrame:
    """One row per never-treated market with the reason it is kept or dropped.

    A donor must not be affected by the treatment itself (SUTVA). It is dropped when it is
    adjacent to the focal market, or when at least `exposure_cap` of its neighbours adopted.
    """
    never = panel.loc[panel.treated_unit == 0, "market_id"].unique()
    exposure = panel[panel.market_id.isin(never)].groupby("market_id")["neighbor_adoption"].max()
    adj_primary = set(adjacency.loc[adjacency.market_id == primary, "neighbor_id"])
    rows = []
    for m in sorted(never):
        adj = m in adj_primary
        high = exposure[m] >= exposure_cap
        reason = ("adjacent to focal market" if adj else "") + (" + " if adj and high else "") + \
                 (f"neighbour exposure >= {exposure_cap:.0%}" if high else "")
        rows.append({"market_id": m, "adjacent_to_focal": adj,
                     "max_neighbor_exposure": round(float(exposure[m]), 3),
                     "status": "dropped" if (adj or high) else "kept", "reason": reason or "clean donor"})
    return pd.DataFrame(rows)
