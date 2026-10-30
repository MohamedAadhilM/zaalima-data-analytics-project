"""Counterfactual what-if engine used by the dashboard.

Assumption (stated explicitly in the app): within a segment the revenue response is
proportional to the log price change, tau(delta) = tau_obs * ln(1+delta) / ln(1+delta_obs)
- i.e. constant elasticity, which is exactly the structure (1+e)*ln(1+delta) of a constant-
elasticity demand curve. Extrapolating far outside the observed delta_obs is unsupported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def scale_effect(tau_obs: float, delta: float, delta_obs: float) -> float:
    return float(tau_obs * np.log1p(delta) / np.log1p(delta_obs))


def simulate_scenario(counterfactual: pd.Series, tau_obs: float, delta: float, delta_obs: float,
                      start_week: int, ramp_weeks: int = 0) -> pd.DataFrame:
    """Scenario revenue = synthetic-control counterfactual * exp(tau(delta)) from `start_week`
    (optionally phased in linearly over `ramp_weeks`)."""
    tau = scale_effect(tau_obs, delta, delta_obs)
    wk = counterfactual.index.to_numpy()
    ramp = np.clip((wk - start_week + 1) / max(ramp_weeks, 1), 0, 1) if ramp_weeks else (wk >= start_week).astype(float)
    scen = counterfactual.to_numpy() * np.exp(tau * ramp)
    out = pd.DataFrame({"counterfactual": counterfactual.to_numpy(), "scenario": scen}, index=counterfactual.index)
    out["incremental"] = out.scenario - out.counterfactual
    out.attrs["tau"] = tau
    return out
