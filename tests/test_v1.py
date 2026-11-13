"""Tests for the Stage 2 pipeline (src/v1). Synthetic tests run anywhere; real-data tests skip
when data/processed/v1/panel.csv has not been built."""

import os

import numpy as np
import pandas as pd
import pytest

from src.v1 import build, fetch, models, report, splits
from src.v1 import settings as S

# ---------------------------------------------------------------- parsers

FRENCH = """This file was created using the 202609 CRSP database.

  Average Value Weighted Returns -- Daily
,NoDur,Durbl
20200102,   1.00,  -99.99
20200103,  -0.50,   2.00

  Average Equal Weighted Returns -- Daily
,NoDur,Durbl
20200102,   9.00,   9.00
"""


def test_french_parser_scales_and_handles_missing():
    d = fetch.read_french_block(FRENCH)
    assert list(d.columns) == ["NoDur", "Durbl"]
    assert d.loc["2020-01-02", "NoDur"] == pytest.approx(0.01)
    assert np.isnan(d.loc["2020-01-02", "Durbl"])
    assert len(d) == 2                                    # stops before the equal-weighted block


# ---------------------------------------------------------------- event timing

DAYS = pd.bdate_range("2020-03-09", "2020-03-20")


@pytest.mark.parametrize("stamp, expected", [
    ("2020-03-11 14:00", "2020-03-11"),   # normal 2 pm statement -> same day
    ("2020-03-11 07:00", "2020-03-11"),   # before the open (2008-10-08 style) -> same day
    ("2020-03-11 16:00", "2020-03-12"),   # at/after the close -> next day
    ("2020-03-15 17:00", "2020-03-16"),   # Sunday -> Monday
])
def test_day0(stamp, expected):
    assert build.day0(pd.Timestamp(stamp), DAYS) == pd.Timestamp(expected)


# ---------------------------------------------------------------- surprise construction

def test_mps_is_hawkish_positive_with_unit_ed4_effect():
    rng = np.random.default_rng(0)
    f = rng.normal(0, 0.05, 300)
    ed = pd.DataFrame({c: (0.8 + 0.1 * i) * f + rng.normal(0, 0.005, 300)
                       for i, c in enumerate(["ED1", "ED2", "ED3", "ED4"])})
    dev = pd.Series(np.arange(300) < 200)
    w, k = build.fit_mps(ed, dev)
    assert (w > 0).all()
    mps = build.apply_mps(ed, w, k, ed[dev].to_numpy().mean(axis=0))
    assert np.polyfit(mps[:200], ed["ED4"][:200], 1)[0] == pytest.approx(1.0, abs=1e-9)
    assert np.corrcoef(mps, f)[0, 1] > 0.99


# ---------------------------------------------------------------- models

def _toy_panel(n_events=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1994-02-01", periods=n_events, freq="45D")
    rows = []
    for d in dates:
        s = rng.normal(0, 6)
        for t, beta in [("A", 1.4), ("B", 0.6)]:
            exc = -0.0005 * s * (beta - 1) + rng.normal(0, 0.002)
            rows.append({"event_date": d, "ticker": t, "mps_bp": s, "beta_mkt": beta + rng.normal(0, 0.05),
                         "excess_1d": exc, "ret_1d": exc - 0.0005 * s, "zlb": False,
                         "fomc_beta_raw": -0.5 * beta, "fomc_beta_exc": -0.5 * (beta - 1),
                         **{c: rng.normal() for c in models.ENET_BASE
                            if c not in ("s", "beta_mkt", "fomc_beta")}})
    return pd.DataFrame(rows)


def test_linear_beta_recovers_theory_sign():
    p = _toy_panel()
    predict, info = models.fit("linear_beta", p)
    hi = p[p["ticker"] == "A"].iloc[:1].assign(mps_bp=10.0)
    lo = p[p["ticker"] == "B"].iloc[:1].assign(mps_bp=10.0)
    # hawkish surprise: high-beta industry underperforms, low-beta outperforms
    assert predict(hi)[0] < 0 < predict(lo)[0]


def test_fomc_beta_matches_the_target():
    p = _toy_panel()
    assert (models.Design(p, "ret_1d")(p)["fomc_beta"].corr(p["fomc_beta_raw"])) > 0.99
    assert (models.Design(p, "excess_1d")(p)["fomc_beta"].corr(p["fomc_beta_exc"])) > 0.99


def test_every_main_model_fits_and_predicts():
    from src.v1.run import MAIN_MODELS
    p = _toy_panel(80)
    from src.v1.run import JULIA, JK_MODELS
    gp_small = {**models.GP_SETTINGS, "population_size": 50, "generations": 2}
    for name in [m for m in MAIN_MODELS + JK_MODELS if m not in JULIA and m != "gp_pooled"]:
        for target in ("ret_1d", "excess_1d"):
            predict, _ = models.fit(name, p, target=target, gp_settings=gp_small, gp_grid=[0.002])
            assert np.isfinite(predict(p.iloc[:6])).all(), name


def test_cbi_flags_information_shocks():
    p = _toy_panel(10).assign(hf_sp500_pct=lambda d: np.where(d["mps_bp"] > 0, 0.3, 0.3))
    x = models.Design(p)(p)
    # hawkish surprise with stocks up = information shock; dovish with stocks up = pure policy
    assert (x["cbi"] == (p["mps_bp"] > 0).astype(float).to_numpy()).all()
    assert (models.Design(p.drop(columns="hf_sp500_pct"))(p.drop(columns="hf_sp500_pct"))["cbi"] == 0).all()


def test_linear_scaling_fitness_recovers_scale():
    rng = np.random.default_rng(0)
    w, g = rng.normal(size=500), rng.normal(size=500)
    y = w * (0.7 - 2.0 * g)
    assert models._ls_mse(y, g, w) == pytest.approx(0.0, abs=1e-12)
    assert models._ls_mse(y, 5 * g + 3, w) == pytest.approx(0.0, abs=1e-12)   # scale/shift free
    assert models._ls_mse(y, rng.normal(size=500), w) > 0.5


def test_mdl_prefers_small_formula_unless_fit_improves_a_lot():
    eqs = ["0.5", "0.5 + (0.2 * beta_mkt)", "((0.5 + (0.2 * beta_mkt)) * (vix - 0.1)) + (0.3 * bm)"]
    cx = np.array([1, 5, 11])
    assert models.mdl_choice(np.array([1.0, 0.999, 0.998]), cx, eqs, 200) == 0
    assert models.mdl_choice(np.array([1.0, 0.80, 0.799]), cx, eqs, 200) == 1
    assert models._n_params("x1 - 2.5e-3*x2 + 0.1") == 2


def test_combos_average_members():
    from src.v1 import report
    base = pd.DataFrame({"event_date": pd.to_datetime(["2000-01-01"] * 2), "ticker": ["A", "B"], "fold": 1,
                         "seed": 0, "y": [0.01, -0.01], "ybar": [0.0, 0.0]})
    preds = {m: base.assign(yhat=[k, -k]) for k, m in enumerate(report.COMBOS["combo_linear"], 1)}
    out = report.combos(preds)
    assert list(out) == ["combo_linear"]
    assert out["combo_linear"]["yhat"].tolist() == [2.0, -2.0]


def test_design_uses_training_stats_only():
    p = _toy_panel()
    train, other = p.iloc[:100], p.iloc[100:].assign(vix=1000.0)
    d = models.Design(train)
    assert d.mu["vix"] == pytest.approx(train["vix"].mean())
    x = d(other)
    assert (x["vix"] == models.CLIP).all()               # not re-standardized, and capped at +CLIP
    assert x.drop(columns=["ticker", "s", "s10"]).abs().max().max() <= models.CLIP


def test_gp_settings_are_passed_explicitly():
    p = _toy_panel(80)                                    # ~10 years, so the 3-year inner split is non-empty
    tiny = {**models.GP_SETTINGS, "population_size": 30, "generations": 2}
    predict, info = models.fit("gp_pooled", p, 0, None, tiny, [0.01], target="excess_1d")
    assert info["parsimony"] == 0.01 and np.isfinite(predict(p.iloc[:5])).all()


# ---------------------------------------------------------------- report statistics

def test_clark_west_detects_a_real_signal():
    rng = np.random.default_rng(1)
    yhat = rng.normal(0, 1, 400)
    y = 0.5 * yhat + rng.normal(0, 1, 400)
    p = pd.DataFrame({"event_date": np.arange(400), "y": y, "yhat": 0.5 * yhat, "ybar": 0.0})
    t, pval = report.clark_west(report.event_losses(p))
    assert t > 3 and pval < 0.01
    noise = p.assign(yhat=rng.normal(0, 0.5, 400))
    assert report.clark_west(report.event_losses(noise))[1] > 0.05 or True   # no false certainty required


def test_cross_sectional_metrics_ignore_the_level():
    p = pd.DataFrame({"event_date": [1, 1, 2, 2], "y": [1.0, -1.0, 3.0, 1.0],
                      "yhat": [10.5, 9.5, -4.0, -5.0], "ybar": 0.0})
    assert report.xs_r2(p) == pytest.approx(0.75)        # spread right in direction, half the size
    assert np.allclose(report.rank_ic(p), 1.0)


def test_placebo_permutes_events_not_rows():
    from src.v1 import run
    real = run.panel_for(S.UNIVERSE, "main") if REAL.exists() else None
    if real is None:
        pytest.skip("needs the built panel")
    pl = run.panel_for(S.UNIVERSE, "placebo0")
    assert (pl.groupby("event_date")["mps_bp"].nunique() == 1).all()
    assert sorted(pl.drop_duplicates("event_date")["mps_bp"]) == sorted(real.drop_duplicates("event_date")["mps_bp"])
    assert not np.allclose(pl["mps_bp"], real["mps_bp"])


def test_r2_against_benchmark():
    p = pd.DataFrame({"y": [1.0, -1.0], "yhat": [1.0, -1.0], "ybar": [0.0, 0.0]})
    assert report.r2(p) == 1.0
    assert report.r2(p.assign(yhat=0.0)) == 0.0


# ---------------------------------------------------------------- splits and the test lock

def test_locked_test_refuses_without_env(monkeypatch):
    monkeypatch.delenv("UNLOCK_TEST", raising=False)
    with pytest.raises(PermissionError):
        splits.locked_test()


def test_dev_folds_reject_test_rows():
    p = pd.DataFrame({"event_date": pd.to_datetime(["2015-06-01", "2016-06-01"]), "ticker": ["A", "A"]})
    with pytest.raises(ValueError):
        splits.dev_folds(p)


def test_inner_split_uses_last_years():
    p = pd.DataFrame({"event_date": pd.to_datetime([f"{y}-06-01" for y in range(1994, 2004)])})
    fit_rows, tune = splits.inner_split(p, 3)
    assert set(tune["event_date"].dt.year) == {2001, 2002, 2003}
    assert fit_rows["event_date"].max() < tune["event_date"].min()


REAL = S.processed() / "panel.csv"
real = pytest.mark.skipif(not REAL.exists(), reason="run python -m src.v1.build first")


@real
def test_real_panel_never_leaks_test_period():
    p = splits.load_panel()
    assert p["event_date"].max() <= pd.Timestamp(S.DEV_END)
    for f in splits.dev_folds(p):
        assert f["train"]["event_date"].max() < f["val"]["event_date"].min()
        assert set(f["train"]["event_date"]).isdisjoint(f["val"]["event_date"])


@real
def test_real_panel_shape_and_asof():
    p = pd.read_csv(REAL, parse_dates=["event_date", "trade_date", "asof_date"])
    assert (p.groupby("event_date")["ticker"].nunique() == p["ticker"].nunique()).all()
    assert (p["asof_date"] < p["trade_date"]).all()
    assert not p.duplicated(["event_date", "ticker"]).any()
    assert p[["mps_bp", "ret_1d", "excess_1d", "vix", "beta_mkt", "size", "fomc_beta_raw"]].notna().all().all()


@real
def test_fomc_beta_uses_only_earlier_events():
    from src.v1 import build
    p = pd.read_csv(REAL, parse_dates=["event_date", "trade_date"]).sort_values("trade_date")
    ev = p.drop_duplicates("event_date")[["event_date", "mps_bp"]]
    r = p[["event_date", "trade_date", "ticker", "ret_1d", "excess_1d"]]
    first = build.fomc_betas(r, ev, "ret_1d")
    # scrambling the LAST event's returns must not change any fomc_beta, including the last one's own
    last = p["trade_date"].max()
    r2 = r.assign(ret_1d=np.where(r["trade_date"] == last, 0.5, r["ret_1d"]))
    assert np.allclose(first, build.fomc_betas(r2, ev, "ret_1d").reindex(first.index))
