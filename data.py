from __future__ import annotations
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import CFG, Config
# --------------------------------------------------------------------------
# 1. Simulator
# --------------------------------------------------------------------------
def _ar1(rng: np.random.Generator, n_units: int, n_t: int, rho: float, sd: float) -> np.ndarray:
    out = np.zeros((n_units, n_t))
    eps = rng.normal(0, sd, (n_units, n_t))
    out[:, 0] = eps[:, 0] / np.sqrt(1 - rho ** 2)
    for t in range(1, n_t):
        out[:, t] = rho * out[:, t - 1] + eps[:, t]
    return out
def simulate_raw(cfg: Config = CFG) -> dict:
    """Generate an interconnected-marketplace panel with a *known* causal effect.
    Design features that make naive A/B-style comparisons fail (and motivate the engine):
      * treatment adoption is confounded (urban, competitive, large markets adopt; timing
        is triggered by rising competitor prices),
      * heterogeneous effects (price elasticity varies with market income),
      * network spillovers (untreated neighbours of adopters gain demand => SUTVA violation).
    """
    cfg.ensure_dirs()
    rng = np.random.default_rng(cfg.seed)
    N, T = cfg.n_markets, cfg.n_weeks
    ids = [f"R{i:02d}" for i in range(N)]

    # ---- static attributes ------------------------------------------------
    log_pop = rng.normal(12.4, 0.7, N)
    income_z = rng.normal(0, 1, N)
    urban = 1 / (1 + np.exp(-(0.8 * income_z + 0.9 * (log_pop - 12.4) / 0.7 + rng.normal(0, 0.6, N))))
    comp_int = rng.uniform(0.2, 1.0, N)
    tier = pd.qcut(log_pop, 3, labels=["C", "B", "A"]).astype(str)
    base_price = rng.uniform(80, 140, N)

    elasticity = np.clip(-0.7 + 0.35 * income_z + rng.normal(0, 0.05, N), -1.6, -0.15)
    delta = 0.10 + rng.normal(0, 0.012, N)                       # dynamic-pricing uplift
    tau_unit = (1 + elasticity) * np.log1p(delta)                # true log-revenue effect

    # ---- spillover network: 3 nearest neighbours in a latent geography ------
    xy = rng.uniform(0, 1, (N, 2))
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nbrs = np.argsort(d, axis=1)[:, :3]
    A = np.zeros((N, N))
    for i in range(N):
        A[i, nbrs[i]] = 1
    A = np.maximum(A, A.T)                                       # symmetric adjacency
    deg = A.sum(1)

    # ---- time-varying drivers ----------------------------------------------
    wk = np.arange(T)
    season = 0.08 * np.sin(2 * np.pi * wk / 52 + 0.6) + 0.05 * ((wk % 52) >= 46)
    macro = _ar1(rng, 1, T, 0.9, 0.012)[0]
    local = _ar1(rng, N, T, 0.6, 0.012)                          # unobserved local demand shock
    cgap = np.zeros((N, T))
    for t in range(T):
        prev = cgap[:, t - 1] if t else 0
        cgap[:, t] = 0.5 * prev + 0.9 * local[:, t] + rng.normal(0, 0.008, N)
    comp_mu = rng.normal(0.03, 0.03, N)

    # ---- who adopts, and when ----------------------------------------------
    z = lambda v: (v - v.mean()) / v.std()
    score = 0.9 * z(urban) + 0.6 * z(comp_int) + 0.4 * z(log_pop) + rng.gumbel(0, 0.7, N)
    treated_idx = np.argsort(-score)[: cfg.n_treated]
    donors_like = [i for i in treated_idx if abs(log_pop[i] - log_pop.mean()) < 0.5]
    primary = int(donors_like[0] if donors_like else treated_idx[0])
    cg4 = pd.DataFrame(cgap.T).rolling(4, min_periods=1).mean().to_numpy().T
    adopt = np.full(N, np.inf)
    for i in treated_idx:
        if i == primary:
            adopt[i] = cfg.primary_adoption_week
            continue
        trig = np.where((cg4[i] > 0.012) & (wk >= 62))[0]         # competitor-price trigger
        adopt[i] = trig[0] if len(trig) else rng.integers(66, 92)
        adopt[i] = min(adopt[i], T - 14)
    D = (wk[None, :] >= adopt[:, None]).astype(float)

    # ---- outcomes ------------------------------------------------------------
    nbr_adopt = (A @ D) / np.maximum(deg, 1)[:, None] * (1 - D)  # exposure of *untreated* markets
    g = rng.normal(0.05, 0.01, N)
    price = base_price[:, None] * (1 + delta[:, None] * D) * np.exp(rng.normal(0, 0.01, (N, T)))
    logunits = (
        (log_pop - 5.0)[:, None] + g[:, None] * wk[None] / T + season[None] + macro[None]
        + local + 0.20 * cgap + elasticity[:, None] * np.log(price / base_price[:, None])
        + 0.06 * nbr_adopt + rng.normal(0, 0.008, (N, T))
    )
    units = np.exp(logunits)
    revenue = price * units
    comp_price = base_price[:, None] * np.exp(comp_mu[:, None] + cgap)

    # ---- write raw extracts (with realistic defects) ---------------------------
    dates = pd.date_range(cfg.start_date, periods=T, freq="W-MON")
    tx = pd.DataFrame({
        "week_start": np.tile(dates, N), "market_id": np.repeat(ids, T),
        "units_sold": units.ravel().round(1), "avg_price": price.ravel().round(2),
        "gross_revenue": revenue.ravel().round(2)})
    ct = pd.DataFrame({
        "week_start": np.tile(dates, N), "market_id": np.repeat(ids, T),
        "competitor_avg_price": comp_price.ravel().round(2)})
    # telemetry defects: dropouts, spikes, duplicated pings
    miss = rng.random(len(ct)) < 0.02
    ct.loc[miss, "competitor_avg_price"] = np.nan
    spike = rng.random(len(ct)) < 0.005
    ct.loc[spike, "competitor_avg_price"] *= rng.uniform(2.0, 3.0, spike.sum())
    ct = pd.concat([ct, ct.sample(25, random_state=cfg.seed)], ignore_index=True)

    attrs = pd.DataFrame({"market_id": ids, "log_pop": log_pop.round(4), "income_z": income_z.round(4),
                          "urban_index": urban.round(4), "comp_intensity": comp_int.round(4), "tier": tier})
    adj = pd.DataFrame([(ids[i], ids[j]) for i in range(N) for j in range(N) if A[i, j]],
                       columns=["market_id", "neighbor_id"])
    ev = pd.DataFrame({"market_id": [ids[i] for i in treated_idx],
                       "go_live_week_idx": [int(adopt[i]) for i in treated_idx]})
    truth = pd.DataFrame({"market_id": ids, "true_tau_log": tau_unit, "true_elasticity": elasticity,
                          "true_price_uplift": delta, "is_primary": [i == primary for i in range(N)]})

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    tx.to_csv(cfg.raw_dir / "market_transactions.csv", index=False)
    ct.to_csv(cfg.raw_dir / "competitor_telemetry.csv", index=False)
    attrs.to_csv(cfg.raw_dir / "market_attributes.csv", index=False)
    adj.to_csv(cfg.raw_dir / "market_adjacency.csv", index=False)
    ev.to_csv(cfg.raw_dir / "pricing_events.csv", index=False)
    truth.to_csv(cfg.raw_dir / "_ground_truth.csv", index=False)
    return {"primary": ids[primary], "treated": [ids[i] for i in treated_idx]}


# --------------------------------------------------------------------------
# 2. Ingestion & validation
# --------------------------------------------------------------------------
@dataclass
class IngestReport:
    rows_transactions: int
    rows_telemetry_raw: int
    duplicates_removed: int
    missing_competitor_price: int
    outliers_winsorised: int
    panel_balanced: bool
    n_markets: int
    n_weeks: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _hampel(s: pd.Series, window: int = 9, k: float = 4.0) -> tuple[pd.Series, int]:
    """Rolling-median / MAD outlier filter; returns cleaned series and #replaced."""
    med = s.rolling(window, center=True, min_periods=5).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=5).median() * 1.4826
    scale = np.maximum(mad, 0.03 * med.abs())          # floor: 3% of local level
    bad = (s - med).abs() > k * scale
    return s.mask(bad, med), int(bad.sum())


def ingest(cfg: Config = CFG, save: bool = True) -> tuple[pd.DataFrame, IngestReport]:
    """Load raw extracts, validate, clean, and build the analysis panel."""
    r = cfg.raw_dir
    tx = pd.read_csv(r / "market_transactions.csv", parse_dates=["week_start"])
    ct = pd.read_csv(r / "competitor_telemetry.csv", parse_dates=["week_start"])
    attrs = pd.read_csv(r / "market_attributes.csv")
    adj = pd.read_csv(r / "market_adjacency.csv")
    ev = pd.read_csv(r / "pricing_events.csv")

    # --- schema & sanity checks -------------------------------------------------
    assert not tx.duplicated(["market_id", "week_start"]).any(), "duplicate transaction keys"
    assert (tx[["units_sold", "avg_price", "gross_revenue"]] > 0).all().all(), "non-positive values"
    n_raw = len(ct)
    ct = ct.drop_duplicates(["market_id", "week_start"])
    dups = n_raw - len(ct)

    # --- telemetry cleaning: outliers -> interpolate dropouts ----------------------
    ct = ct.sort_values(["market_id", "week_start"]).reset_index(drop=True)
    n_missing = int(ct["competitor_avg_price"].isna().sum())
    n_out = 0
    cleaned = []
    for _, g in ct.groupby("market_id", sort=False):
        s, k = _hampel(g["competitor_avg_price"])
        n_out += k
        cleaned.append(s.interpolate(limit_direction="both"))
    ct["competitor_avg_price"] = pd.concat(cleaned)

    # --- merge & feature construction ---------------------------------------------
    df = tx.merge(ct, on=["market_id", "week_start"], how="left", validate="1:1")
    df = df.merge(attrs, on="market_id", how="left", validate="m:1")
    df = df.sort_values(["market_id", "week_start"]).reset_index(drop=True)
    df["week_idx"] = df.groupby("market_id").cumcount()

    golive = dict(zip(ev["market_id"], ev["go_live_week_idx"]))
    df["go_live"] = df["market_id"].map(golive)
    df["treated_unit"] = df["go_live"].notna().astype(int)
    df["dynamic_pricing"] = (df["week_idx"] >= df["go_live"].fillna(np.inf)).astype(int)
    df["event_time"] = df["week_idx"] - df["go_live"]

    base = df[df.week_idx < 52].groupby("market_id")["avg_price"].median().rename("base_price")
    df = df.merge(base, on="market_id")
    df["log_revenue"] = np.log(df["gross_revenue"])
    df["log_price"] = np.log(df["avg_price"])
    df["comp_gap"] = np.log(df["competitor_avg_price"] / df["base_price"])
    df["comp_gap_4w"] = df.groupby("market_id")["comp_gap"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["seasonality"] = np.sin(2 * np.pi * df["week_idx"] / 52 + 0.6)

    # spillover exposure: share of a market's neighbours already on dynamic pricing
    dp = df.pivot(index="market_id", columns="week_idx", values="dynamic_pricing")
    ids = list(dp.index)
    A = np.zeros((len(ids), len(ids)))
    pos = {m: i for i, m in enumerate(ids)}
    for a, b in zip(adj["market_id"], adj["neighbor_id"]):
        A[pos[a], pos[b]] = 1
    expo = pd.DataFrame((A @ dp.to_numpy()) / np.maximum(A.sum(1), 1)[:, None], index=ids, columns=dp.columns)
    df["neighbor_adoption"] = [expo.loc[m, w] for m, w in zip(df.market_id, df.week_idx)]
    df["adjacent_to_treated"] = df["market_id"].map(
        lambda m: int(any(n in golive for n in adj.loc[adj.market_id == m, "neighbor_id"])))

    balanced = bool(df.groupby("market_id").size().nunique() == 1)
    rep = IngestReport(len(tx), n_raw, dups, n_missing, n_out, balanced,
                       df.market_id.nunique(), df.week_idx.nunique())
    if save:
        cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.processed_dir / "panel.csv", index=False)
        (cfg.processed_dir / "ingest_report.json").write_text(json.dumps(rep.to_dict(), indent=2))
    return df, rep


def load_ground_truth(cfg: Config = CFG) -> pd.DataFrame:
    return pd.read_csv(cfg.raw_dir / "_ground_truth.csv")


def primary_market(cfg: Config = CFG) -> str:
    gt = load_ground_truth(cfg)
    return str(gt.loc[gt.is_primary, "market_id"].iloc[0])
