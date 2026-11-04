"""Compute each sector ETF's return around each event (the prediction target).

Inputs:
    data/raw/prices.csv         from src/fetch_data.py (adjusted close, one column per ticker)
    data/processed/events.csv   from src/build_events.py

Output: data/processed/returns.csv, one row per (event, sector ETF), with columns
    event_date   announcement date (same as events.csv)
    trade_date   first trading day on/after event_date (same as events.csv)
    ticker       sector ETF (SPY itself is not a row, it is the benchmark)
    ret_1d       simple return, close on day -1 to close on day 0
    ret_5d       simple return, close on day -1 to close on day +4
    spy_1d       same window for SPY
    spy_5d       same window for SPY
    excess_1d    ret_1d - spy_1d
    excess_5d    ret_5d - spy_5d

Day 0 is trade_date, day -1 the trading day before it. This is the same
window as surprise_bp in events.csv (yield change from day -1 to day 0), so
the FOMC statement (released ~2pm on day 0) is inside both windows.
"5d" means trading days 0..4, i.e. 5 daily returns.

Returns are decimals (0.01 = 1%). Excess return is vs. SPY, not beta-adjusted.

Missing data: a (event, ticker) pair is dropped if the ETF has no price on
day -1 or day 0 (e.g. XLC before 2018-06, XLRE before 2015-10). If day +4 is
missing (event near the end of the data) the row is kept with ret_5d = NaN.

Run from the repo root:
    python -m src.build_returns
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

BENCHMARK = "SPY"
HORIZONS = {"1d": 0, "5d": 4}  # name -> last day of the window (day 0 = trade_date)


def load_prices() -> pd.DataFrame:
    """Adjusted close, rows = trading days, columns = tickers."""
    prices = pd.read_csv(RAW_DIR / "prices.csv", index_col=0, parse_dates=True)
    # Drop days where the market was closed for every ticker.
    return prices.sort_index().dropna(how="all")


def load_events() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "events.csv", parse_dates=["event_date", "trade_date"])


def window_return(prices: pd.DataFrame, start: int, end: int) -> pd.Series:
    """Return per ticker from the close at row `start` to the close at row `end`.

    NaN if either end is outside the data or has no price.
    """
    if start < 0 or end >= len(prices):
        return pd.Series(float("nan"), index=prices.columns)
    return prices.iloc[end] / prices.iloc[start] - 1


def build_returns() -> pd.DataFrame:
    prices = load_prices()
    events = load_events()
    days = prices.index
    sectors = [t for t in prices.columns if t != BENCHMARK]

    rows = []
    for _, event in events.iterrows():
        # Day 0 = first price day on/after trade_date (should equal trade_date).
        i = days.searchsorted(event["trade_date"])
        if i == 0 or i >= len(days):
            continue  # no price data before or after this event

        rets = {name: window_return(prices, i - 1, i + last) for name, last in HORIZONS.items()}
        for ticker in sectors:
            if pd.isna(rets["1d"][ticker]):
                continue  # ETF did not exist yet (or missing price)
            row = {
                "event_date": event["event_date"].date(),
                "trade_date": event["trade_date"].date(),
                "ticker": ticker,
            }
            for name in HORIZONS:
                row[f"ret_{name}"] = rets[name][ticker]
            for name in HORIZONS:
                row[f"spy_{name}"] = rets[name][BENCHMARK]
            for name in HORIZONS:
                row[f"excess_{name}"] = rets[name][ticker] - rets[name][BENCHMARK]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    returns = build_returns()
    out = PROCESSED_DIR / "returns.csv"
    returns.to_csv(out, index=False)
    print(f"returns: {returns.shape} -> {out}")
    print(f"events: {returns['event_date'].nunique()}, tickers: {returns['ticker'].nunique()}")
    print((returns.groupby("ticker")["excess_1d"].agg(["count", "mean", "std"]) * [1, 100, 100])
          .round(3).rename(columns={"mean": "mean_%", "std": "std_%"}).to_string())


if __name__ == "__main__":
    main()
