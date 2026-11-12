"""Compute model inputs for each (event, ticker) pair, using only data known before the event.

Inputs:
    data/raw/prices.csv          from src/fetch_data.py (adjusted close, one column per ticker)
    data/raw/fred.csv            from src/fetch_data.py (DGS2, DGS10, VIXCLS, BAA10Y, DFF)
    data/processed/events.csv    from src/build_events.py
    data/processed/returns.csv   from src/build_returns.py (decides which (event, ticker) rows exist)

Output: data/processed/features.csv, one row per (event_date, ticker) in returns.csv, with columns
    event_date       join key (same as events.csv / returns.csv)
    ticker           join key
    asof_date        day -1: the last trading day before trade_date. Every feature uses data up to
                     this date and nothing later.
    macro_asof_date  the FRED date the macro values were read from (<= asof_date; earlier only when
                     the bond market was closed on day -1)
    vix              VIX close (VIXCLS) on day -1, index points
    slope_2s10s      DGS10 - DGS2 on day -1, percentage points (1.0 = 100 bp; positive = upward sloping)
    beta_1y          OLS slope of the ticker's daily returns on SPY's over the 252 trading days
                     ending day -1 (NaN if fewer than 126 usable days)
    mom_20d          ticker's return over the 20 trading days ending day -1: P[-1] / P[-21] - 1
    credit           Baa corporate minus 10y Treasury yield (BAA10Y) on day -1, percentage points
    ff               effective fed funds rate (DFF) on day -1, percent
    vol_60d          standard deviation of the ticker's daily returns over the 60 trading days
                     ending day -1 (daily, not annualized; NaN if fewer than 40 usable days)
    mkt_mom_20d      SPY's return over the same 20 days as mom_20d (identical for all tickers of
                     one event)

The first four (vix, slope_2s10s, beta_1y, mom_20d) are the Stage 1 inputs in config.FEATURES; the
rest are optional extras for a GP v2 / robustness run (docs/STAGE1_PLAN.md, week 8).

Day -1 is found in the *price* calendar, the same way build_returns.py finds the start of its return
window, so features end exactly where the target return begins. FRED series are read "as of" day -1:
the latest value dated on or before it. A value older than MAX_STALE_DAYS calendar days is treated
as missing rather than carried forward.

Returns are decimals (0.01 = 1%).

Run from the repo root:
    python -m src.build_features
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

BENCHMARK = "SPY"
BETA_WINDOW, BETA_MIN = 252, 126   # trading days of daily returns; minimum usable days
VOL_WINDOW, VOL_MIN = 60, 40
MOM_WINDOW = 20                    # P[day -1] / P[day -1 - 20] - 1
MAX_STALE_DAYS = 7                 # calendar days a FRED value may lag day -1

MACRO = {                          # output column -> how it is built from fred.csv
    "vix": lambda f: f["VIXCLS"],
    "slope_2s10s": lambda f: f["DGS10"] - f["DGS2"],
    "credit": lambda f: f["BAA10Y"],
    "ff": lambda f: f["DFF"],
}
COLUMNS = ["event_date", "ticker", "asof_date", "macro_asof_date",
           "vix", "slope_2s10s", "beta_1y", "mom_20d",
           "credit", "ff", "vol_60d", "mkt_mom_20d"]


# ---------- loading (kept separate so tests can pass in small DataFrames) ----------

def load_prices(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Adjusted close, rows = trading days, columns = tickers (same cleaning as build_returns)."""
    prices = pd.read_csv(raw_dir / "prices.csv", index_col=0, parse_dates=True)
    return prices.sort_index().dropna(how="all")


def load_fred(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "fred.csv", index_col=0, parse_dates=True).sort_index()


def load_events(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return pd.read_csv(processed_dir / "events.csv", parse_dates=["event_date", "trade_date"])


def load_returns(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return pd.read_csv(processed_dir / "returns.csv", parse_dates=["event_date", "trade_date"])


# ---------- building blocks ----------

def day_minus_1(trading_days: pd.DatetimeIndex, trade_date: pd.Timestamp) -> int | None:
    """Row position of the last trading day strictly before trade_date (None if there is none)."""
    i = trading_days.searchsorted(trade_date)  # first row on/after trade_date
    return int(i - 1) if i >= 1 else None


def macro_asof(fred: pd.DataFrame, asof: pd.Timestamp,
               macro: dict | None = None) -> tuple[dict, pd.Timestamp | None]:
    """Macro features from the latest FRED row dated <= asof (never a later one).

    Each series is read as of its own latest non-missing value, so a holiday in one series does
    not blank the others. Values older than MAX_STALE_DAYS are returned as NaN.
    `macro` (output name -> builder) defaults to MACRO; Stage 2 passes its own.
    """
    out, used_dates = {}, []
    for name, build in (macro or MACRO).items():
        s = build(fred).dropna()
        s = s[s.index <= asof]
        if s.empty or (asof - s.index[-1]).days > MAX_STALE_DAYS:
            out[name] = np.nan
            continue
        out[name] = float(s.iloc[-1])
        used_dates.append(s.index[-1])
    return out, (min(used_dates) if used_dates else None)


def rolling_beta(rets: pd.DataFrame, benchmark: str = BENCHMARK,
                 window: int = BETA_WINDOW, min_periods: int = BETA_MIN) -> pd.DataFrame:
    """Rolling OLS slope of each column on the benchmark, using only days where both exist.

    Row t uses the `window` returns ending at t (inclusive). Masking the benchmark where the
    ticker is missing keeps numerator and denominator on the same days, so this equals
    np.polyfit(m, x, 1)[0] on the overlapping days.
    """
    m = rets[benchmark]
    betas = {}
    for col in rets.columns:
        if col == benchmark:
            continue
        x = rets[col]
        m_on_x = m.where(x.notna())
        cov = x.rolling(window, min_periods=min_periods).cov(m_on_x)
        var = m_on_x.rolling(window, min_periods=min_periods).var()
        betas[col] = cov / var
    return pd.DataFrame(betas, index=rets.index)


def build_features(prices: pd.DataFrame, fred: pd.DataFrame,
                   events: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """One row per (event_date, ticker) present in `returns`, all values as of day -1."""
    prices = prices.sort_index().dropna(how="all")
    days = prices.index
    rets = prices / prices.shift(1) - 1          # daily simple returns; NaN where a price is missing
    betas = rolling_beta(rets)
    vols = rets.rolling(VOL_WINDOW, min_periods=VOL_MIN).std()
    moms = prices / prices.shift(MOM_WINDOW) - 1

    trade_dates = events.drop_duplicates("event_date").set_index("event_date")["trade_date"]
    rows = []
    for event_date, group in returns.groupby("event_date", sort=True):
        j = day_minus_1(days, pd.Timestamp(trade_dates[event_date]))
        if j is None:
            continue
        asof = days[j]
        macro, macro_date = macro_asof(fred, asof)
        for ticker in group["ticker"]:
            rows.append({
                "event_date": pd.Timestamp(event_date).date(),
                "ticker": ticker,
                "asof_date": asof.date(),
                "macro_asof_date": macro_date.date() if macro_date is not None else None,
                **macro,
                "beta_1y": betas[ticker].iloc[j] if ticker in betas else np.nan,
                "mom_20d": moms[ticker].iloc[j],
                "vol_60d": vols[ticker].iloc[j],
                "mkt_mom_20d": moms[BENCHMARK].iloc[j],
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features = build_features(load_prices(), load_fred(), load_events(), load_returns())
    out = PROCESSED_DIR / "features.csv"
    features.to_csv(out, index=False)
    print(f"features: {features.shape} -> {out}")
    num = features.select_dtypes("number")
    summary = num.describe().T[["count", "mean", "std", "min", "max"]]
    summary["missing"] = len(features) - summary["count"]
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
