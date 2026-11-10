"""Tests for src/build_returns.py and src/build_events.py.

Synthetic tests monkeypatch the loaders so they run without data files (CI). Real-data tests run
only when data/processed/returns.csv exists.
"""

import numpy as np
import pandas as pd
import pytest

from src import build_events as be
from src import build_returns as br

# ---------------------------------------------------------------- synthetic

N_DAYS = 60


@pytest.fixture()
def fake_prices():
    days = pd.bdate_range("2020-01-06", periods=N_DAYS)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame({
        "SPY": 300 * np.cumprod(1 + rng.normal(0, 0.01, N_DAYS)),
        "XLA": 60 * np.cumprod(1 + rng.normal(0, 0.012, N_DAYS)),
        "XLNEW": 25 * np.cumprod(1 + rng.normal(0, 0.012, N_DAYS)),
    }, index=days)
    prices.iloc[:30, prices.columns.get_loc("XLNEW")] = np.nan   # launches at row 30
    return prices


def _run(monkeypatch, prices, trade_positions):
    days = prices.index
    events = pd.DataFrame({"event_date": days[trade_positions], "trade_date": days[trade_positions]})
    monkeypatch.setattr(br, "load_prices", lambda: prices)
    monkeypatch.setattr(br, "load_events", lambda: events)
    return br.build_returns()


def test_returns_by_hand(monkeypatch, fake_prices):
    out = _run(monkeypatch, fake_prices, [40])
    row = out[out["ticker"] == "XLA"].iloc[0]
    p, s = fake_prices["XLA"], fake_prices["SPY"]
    assert row["ret_1d"] == pytest.approx(p.iloc[40] / p.iloc[39] - 1)
    assert row["ret_5d"] == pytest.approx(p.iloc[44] / p.iloc[39] - 1)
    assert row["spy_1d"] == pytest.approx(s.iloc[40] / s.iloc[39] - 1)
    assert row["excess_1d"] == pytest.approx(row["ret_1d"] - row["spy_1d"])
    assert row["excess_5d"] == pytest.approx(row["ret_5d"] - row["spy_5d"])


def test_spy_is_benchmark_not_a_row(monkeypatch, fake_prices):
    out = _run(monkeypatch, fake_prices, [40])
    assert "SPY" not in set(out["ticker"])


def test_ticker_dropped_before_it_exists(monkeypatch, fake_prices):
    out = _run(monkeypatch, fake_prices, [10, 30, 31, 40])
    new = out[out["ticker"] == "XLNEW"]
    # row 30 needs a day -1 price (row 29), which does not exist yet
    assert set(pd.to_datetime(new["trade_date"])) == set(fake_prices.index[[31, 40]])
    assert len(out[out["ticker"] == "XLA"]) == 4


def test_5d_nan_near_end_of_data(monkeypatch, fake_prices):
    out = _run(monkeypatch, fake_prices, [N_DAYS - 2])
    assert out["ret_1d"].notna().all()
    assert out["ret_5d"].isna().all()


def test_weekend_event_maps_to_next_trading_day(monkeypatch):
    days = pd.bdate_range("2020-03-02", periods=20)
    dgs2 = pd.DataFrame({"DGS2": np.linspace(0.9, 0.3, 20)}, index=days)
    fomc = pd.DataFrame({"date": pd.to_datetime(["2020-03-15", "2020-03-11"]),
                         "type": ["unscheduled", "scheduled"]})
    monkeypatch.setattr(be, "load_dgs2", lambda: dgs2["DGS2"])
    monkeypatch.setattr(be, "load_fomc_dates", lambda: fomc.sort_values("date").reset_index(drop=True))
    ev = be.build_events()
    sun = ev[ev["event_date"] == pd.Timestamp("2020-03-15").date()].iloc[0]
    assert sun["trade_date"] == pd.Timestamp("2020-03-16").date()     # Sunday -> Monday
    i = days.get_loc(pd.Timestamp("2020-03-16"))
    assert sun["surprise_bp"] == pytest.approx(round((dgs2.iloc[i, 0] - dgs2.iloc[i - 1, 0]) * 100, 1))


# ---------------------------------------------------------------- real data

REAL = br.PROCESSED_DIR / "returns.csv"
real = pytest.mark.skipif(not REAL.exists(), reason="run the data pipeline first (data files are not in git)")


@pytest.fixture(scope="module")
def real_returns():
    return pd.read_csv(REAL, parse_dates=["event_date", "trade_date"])


@real
def test_real_excess_identity(real_returns):
    r = real_returns
    assert np.allclose(r["excess_1d"], r["ret_1d"] - r["spy_1d"])
    ok = r["ret_5d"].notna()
    assert np.allclose(r.loc[ok, "excess_5d"], r.loc[ok, "ret_5d"] - r.loc[ok, "spy_5d"])


@real
def test_real_late_etfs_start_when_they_launched(real_returns):
    r = real_returns
    assert r.loc[r["ticker"] == "XLRE", "trade_date"].min() >= pd.Timestamp("2015-10-08")
    assert r.loc[r["ticker"] == "XLC", "trade_date"].min() >= pd.Timestamp("2018-06-19")


@real
def test_real_unique_and_spy_constant_per_event(real_returns):
    r = real_returns
    assert not r.duplicated(["event_date", "ticker"]).any()
    assert (r.groupby("event_date")["spy_1d"].nunique() == 1).all()


@real
def test_real_trade_date_is_a_stock_trading_day(real_returns):
    """trade_date comes from the bond (DGS2) calendar; it must also be a day stocks traded."""
    prices = br.load_prices()
    assert set(real_returns["trade_date"]).issubset(set(prices.index))


@real
def test_real_2008_10_08_coordinated_cut(real_returns):
    """Fed + 5 central banks cut 50bp at 7:00 a.m. ET on Wed 2008-10-08, before the open.

    Day 0 must be that same day, and SPY's day-0 return should be about -1% (S&P 500 closed
    -1.1% that day: 996.23 -> 984.94).
    """
    ev = be.PROCESSED_DIR / "events.csv"
    events = pd.read_csv(ev, parse_dates=["event_date", "trade_date"])
    row = events[events["event_date"] == "2008-10-08"].iloc[0]
    assert row["trade_date"] == pd.Timestamp("2008-10-08")
    assert row["event_type"] == "fomc_unscheduled"
    spy = real_returns.loc[real_returns["event_date"] == "2008-10-08", "spy_1d"].iloc[0]
    assert -0.03 < spy < 0.0
