import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpe import dag, dml, scenario, sensitivity as sens, synthetic_control as sc  # noqa: E402
from cpe.utils import get_panel  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return get_panel()


def test_panel_is_balanced_and_clean(panel):
    assert panel.groupby("market_id").size().nunique() == 1
    assert panel[["log_revenue", "comp_gap", "comp_gap_4w"]].notna().all().all()
    assert panel.dynamic_pricing.isin([0, 1]).all()


def test_dag_is_acyclic_and_identifiable(panel):
    g = dag.build_dag()
    assert dag.TREATMENT in g and dag.OUTCOME in g
    _, _, adj = dag.identify(panel, g)
    assert {"comp_gap", "comp_gap_4w"} & set(adj)   # DoWhy may return either equivalent minimal set


def test_scm_weights_on_simplex_and_recover_known_mixture():
    rng = np.random.default_rng(0)
    T, J = 80, 8
    Y0 = rng.normal(0, 1, (T, J)).cumsum(0) * 0.05
    w_true = np.array([.5, .3, .2, 0, 0, 0, 0, 0])
    y1 = Y0 @ w_true + rng.normal(0, 0.005, T)
    w = sc.solve_weights(y1 - y1.mean(), Y0 - Y0.mean(0), 0.0)
    assert np.isclose(w.sum(), 1) and (w >= -1e-9).all()
    assert np.abs(w - w_true).max() < 0.1


def test_scm_effect_close_to_truth(panel):
    from cpe.data import primary_market, load_ground_truth
    from cpe.config import CFG
    prim = primary_market()
    adj = pd.read_csv(CFG.raw_dir / "market_adjacency.csv")
    donors, dropped = sc.screen_donors(panel, prim, adj)
    assert not set(donors) & set(dropped)
    r = sc.fit_scm(panel, prim, donors, CFG.primary_adoption_week, CFG.train_end)
    truth = float(load_ground_truth().query("is_primary").true_tau_log.iloc[0])
    assert r.rmspe_pre < 0.05
    assert abs(r.att_log - truth) < 0.05          # within sampling noise of a single market


def test_dml_beats_naive_and_ci_sane(panel):
    from cpe.data import load_ground_truth
    prep = dml.prepare(panel)
    fit = dml.dml_plr(prep)
    truth = dml.true_att(panel, load_ground_truth())
    naive = dml.naive_estimates(panel)["naive_treated_vs_never_post"]
    assert abs(fit["theta"] - truth) < abs(naive - truth) / 5
    assert fit["ci"][0] < fit["theta"] < fit["ci"][1]


def test_robustness_value_monotone():
    a = sens.robustness_value(0.03, 0.01, 4000)
    b = sens.robustness_value(0.03, 0.02, 4000)
    assert a["RV_q"] > b["RV_q"] > 0


def test_scenario_engine():
    cf = pd.Series(100.0, index=range(10))
    s = scenario.simulate_scenario(cf, tau_obs=0.05, delta=0.10, delta_obs=0.10, start_week=5)
    assert np.isclose(s.attrs["tau"], 0.05) and (s.incremental[:5] == 0).all() and (s.incremental[5:] > 0).all()
    assert scenario.scale_effect(0.05, 0.0, 0.10) == 0.0
