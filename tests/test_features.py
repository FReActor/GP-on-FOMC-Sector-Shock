"""Tests for src/build_features.py.

Synthetic tests build small price / FRED / event tables in memory, so they run anywhere (CI has no
data files). Real-data tests run only when data/processed/features.csv exists.
"""

import numpy as np
import pandas as pd
import pytest

from src import build_features as bf

# ---------------------------------------------------------------- synthetic fixtures

N_DAYS = 400
EVENT_POSITIONS = [300, 330, 399]       # row positions of day 0 for three events
LATE_START = 250                        # XLLATE has prices only from this row on


@pytest.fixture()
def synth():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2010-01-04", periods=N_DAYS)
    spy_r = rng.normal(0.0004, 0.01, N_DAYS)
    xa_r = 1.3 * spy_r + rng.normal(0, 0.006, N_DAYS)          # true beta about 1.3
    xl_r = 0.7 * spy_r + rng.normal(0, 0.006, N_DAYS)
    prices = pd.DataFrame({
        "SPY": 100 * np.cumprod(1 + spy_r),
        "XLA": 50 * np.cumprod(1 + xa_r),
        "XLLATE": 20 * np.cumprod(1 + xl_r),
    }, index=days)
    prices.iloc[:LATE_START, prices.columns.get_loc("XLLATE")] = np.nan

    fred = pd.DataFrame({
        "DGS2": np.linspace(1.0, 2.0, N_DAYS),
        "DGS10": np.linspace(3.0, 3.5, N_DAYS),
        "VIXCLS": np.linspace(15, 30, N_DAYS),
        "BAA10Y": np.linspace(2.0, 2.5, N_DAYS),
        "DFF": np.linspace(0.1, 0.2, N_DAYS),
    }, index=days)

    trade_dates = days[EVENT_POSITIONS]
    events = pd.DataFrame({"event_date": trade_dates, "trade_date": trade_dates})
    returns = pd.DataFrame([(d, t) for d in trade_dates for t in ["XLA", "XLLATE"]],
                           columns=["event_date", "ticker"])
    return prices, fred, events, returns


def _row(feats, event_date, ticker):
    r = feats[(feats["event_date"] == pd.Timestamp(event_date).date()) & (feats["ticker"] == ticker)]
    assert len(r) == 1
    return r.iloc[0]


# ---------------------------------------------------------------- synthetic tests

def test_one_row_per_returns_row(synth):
    prices, fred, events, returns = synth
    f = bf.build_features(prices, fred, events, returns)
    assert len(f) == len(returns)
    assert not f.duplicated(["event_date", "ticker"]).any()
    assert list(f.columns) == bf.COLUMNS


def test_asof_is_previous_trading_day(synth):
    prices, fred, events, returns = synth
    f = bf.build_features(prices, fred, events, returns)
    for pos in EVENT_POSITIONS:
        row = _row(f, prices.index[pos], "XLA")
        assert row["asof_date"] == prices.index[pos - 1].date()
        assert row["asof_date"] < prices.index[pos].date()


def test_mom_20d_by_hand(synth):
    prices, fred, events, returns = synth
    f = bf.build_features(prices, fred, events, returns)
    pos = EVENT_POSITIONS[0]
    p = prices["XLA"]
    expected = p.iloc[pos - 1] / p.iloc[pos - 21] - 1
    assert _row(f, prices.index[pos], "XLA")["mom_20d"] == pytest.approx(expected, rel=1e-12)
    spy = prices["SPY"]
    assert _row(f, prices.index[pos], "XLA")["mkt_mom_20d"] == pytest.approx(
        spy.iloc[pos - 1] / spy.iloc[pos - 21] - 1, rel=1e-12)


def test_beta_1y_matches_polyfit(synth):
    prices, fred, events, returns = synth
    f = bf.build_features(prices, fred, events, returns)
    pos = EVENT_POSITIONS[1]
    r = prices.pct_change()
    window = r.iloc[pos - bf.BETA_WINDOW: pos]            # 252 returns ending day -1
    expected = np.polyfit(window["SPY"], window["XLA"], 1)[0]
    got = _row(f, prices.index[pos], "XLA")["beta_1y"]
    assert got == pytest.approx(expected, rel=1e-9)
    assert 1.0 < got < 1.6                                 # true beta 1.3


def test_beta_uses_only_overlapping_days_and_needs_min_history(synth):
    prices, fred, events, returns = synth
    f = bf.build_features(prices, fred, events, returns)
    # XLLATE starts at row 250: event at 300 has 49 usable returns (< 126) -> NaN
    assert np.isnan(_row(f, prices.index[300], "XLLATE")["beta_1y"])
    # event at 399: returns from row 251 to 398 = 148 usable days -> defined, equals polyfit on them
    pos = 399
    r = prices.pct_change().iloc[pos - bf.BETA_WINDOW: pos].dropna(subset=["XLLATE"])
    expected = np.polyfit(r["SPY"], r["XLLATE"], 1)[0]
    assert _row(f, prices.index[pos], "XLLATE")["beta_1y"] == pytest.approx(expected, rel=1e-9)


def test_vol_60d_by_hand(synth):
    prices, fred, events, returns = synth
    f = bf.build_features(prices, fred, events, returns)
    pos = EVENT_POSITIONS[0]
    r = prices["XLA"].pct_change().iloc[pos - bf.VOL_WINDOW: pos]
    assert _row(f, prices.index[pos], "XLA")["vol_60d"] == pytest.approx(r.std(ddof=1), rel=1e-9)


def test_macro_values_from_day_minus_1(synth):
    prices, fred, events, returns = synth
    f = bf.build_features(prices, fred, events, returns)
    pos = EVENT_POSITIONS[0]
    row = _row(f, prices.index[pos], "XLA")
    d1 = prices.index[pos - 1]
    assert row["vix"] == pytest.approx(fred.loc[d1, "VIXCLS"])
    assert row["slope_2s10s"] == pytest.approx(fred.loc[d1, "DGS10"] - fred.loc[d1, "DGS2"])
    assert row["credit"] == pytest.approx(fred.loc[d1, "BAA10Y"])
    assert row["ff"] == pytest.approx(fred.loc[d1, "DFF"])
    assert row["macro_asof_date"] == d1.date()


def test_macro_falls_back_to_earlier_value_on_bond_holiday(synth):
    prices, fred, events, returns = synth
    pos = EVENT_POSITIONS[0]
    d1, d2 = prices.index[pos - 1], prices.index[pos - 2]
    fred = fred.copy()
    fred.loc[d1, ["DGS2", "DGS10"]] = np.nan          # bond market closed on day -1
    f = bf.build_features(prices, fred, events, returns)
    row = _row(f, prices.index[pos], "XLA")
    assert row["slope_2s10s"] == pytest.approx(fred.loc[d2, "DGS10"] - fred.loc[d2, "DGS2"])
    assert row["vix"] == pytest.approx(fred.loc[d1, "VIXCLS"])   # VIX still from day -1
    assert row["macro_asof_date"] == d2.date()


def test_stale_macro_is_missing_not_carried_forward(synth):
    prices, fred, events, returns = synth
    pos = EVENT_POSITIONS[0]
    fred = fred.copy()
    fred.iloc[pos - 15: pos, fred.columns.get_loc("VIXCLS")] = np.nan   # 3 weeks without VIX
    f = bf.build_features(prices, fred, events, returns)
    assert np.isnan(_row(f, prices.index[pos], "XLA")["vix"])


def test_no_lookahead_future_data_does_not_change_features(synth):
    """Scramble every price and FRED value from day 0 on: features must not move."""
    prices, fred, events, returns = synth
    base = bf.build_features(prices, fred, events, returns)
    pos = EVENT_POSITIONS[0]
    p2, f2 = prices.copy(), fred.copy()
    p2.iloc[pos:] = p2.iloc[pos:] * 3.7
    f2.iloc[pos:] = -999.0
    moved = bf.build_features(p2, f2, events, returns)
    first = base["event_date"] == prices.index[pos].date()
    pd.testing.assert_frame_equal(base[first].reset_index(drop=True),
                                  moved[first].reset_index(drop=True))


# ---------------------------------------------------------------- real-data tests

REAL = bf.PROCESSED_DIR / "features.csv"
real = pytest.mark.skipif(not REAL.exists(), reason="run the data pipeline first (data files are not in git)")


@pytest.fixture(scope="module")
def real_data():
    feats = pd.read_csv(REAL, parse_dates=["event_date", "asof_date", "macro_asof_date"])
    returns = bf.load_returns()
    events = bf.load_events()
    prices = bf.load_prices()
    return feats, returns, events, prices


@real
def test_real_rows_match_returns(real_data):
    feats, returns, _, _ = real_data
    assert len(feats) == len(returns)
    assert not feats.duplicated(["event_date", "ticker"]).any()
    keys = lambda d: set(zip(d["event_date"], d["ticker"]))
    assert keys(feats) == keys(returns)


@real
def test_real_asof_before_trade_date(real_data):
    feats, _, events, prices = real_data
    m = feats.merge(events[["event_date", "trade_date"]], on="event_date")
    assert (m["asof_date"] < m["trade_date"]).all()
    assert (m["macro_asof_date"] <= m["asof_date"]).all()
    assert ((m["asof_date"] - m["macro_asof_date"]).dt.days <= bf.MAX_STALE_DAYS).all()
    # asof_date is exactly the trading day before trade_date (same day -1 as build_returns)
    days = prices.index
    expected = m["trade_date"].map(lambda t: days[days.searchsorted(t) - 1])
    assert (m["asof_date"] == expected).all()


@real
def test_real_spot_check_three_rows(real_data):
    feats, _, _, prices = real_data
    sample = feats.dropna(subset=["beta_1y"]).sample(3, random_state=42)
    rets = prices.pct_change()
    for _, row in sample.iterrows():
        j = prices.index.get_loc(row["asof_date"])
        p = prices[row["ticker"]]
        assert row["mom_20d"] == pytest.approx(p.iloc[j] / p.iloc[j - 20] - 1, rel=1e-9)
        w = rets.iloc[j - 251: j + 1][[row["ticker"], "SPY"]].dropna()
        assert row["beta_1y"] == pytest.approx(np.polyfit(w["SPY"], w[row["ticker"]], 1)[0], rel=1e-9)


@real
def test_real_values_are_plausible(real_data):
    feats, _, _, _ = real_data
    oct08 = feats[(feats["event_date"] >= "2008-10-01") & (feats["event_date"] <= "2008-10-31")]
    assert oct08["vix"].between(25, 90).all()          # VIX was 30-80 in October 2008
    assert feats["vix"].between(8, 90).all()
    assert feats["slope_2s10s"].between(-1.5, 3.5).all()
    core = feats[feats["ticker"].isin(["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"])]
    assert core["beta_1y"].notna().mean() > 0.98       # only the first 1999 events lack a year of history
    assert core["beta_1y"].dropna().between(0, 3).all()
