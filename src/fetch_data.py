"""Download raw market and macro data into data/raw/.

Run from the repo root:
    python -m src.fetch_data
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

START = "1998-01-01"
END = None  # None = up to today

# SPDR sector ETFs plus the S&P 500 as the market benchmark.
# XLRE (2015) and XLC (2018) start later than the others.
TICKERS = [
    "SPY",
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "XLRE", "XLC",
]

# FRED series: 2y and 10y Treasury yields, VIX, Baa corporate spread, fed funds.
FRED_SERIES = ["DGS2", "DGS10", "VIXCLS", "BAA10Y", "DFF"]


def fetch_prices() -> pd.DataFrame:
    """Daily adjusted close for each ticker, one column per ticker."""
    df = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False, threads=False)
    return df["Close"]


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def fetch_fred() -> pd.DataFrame:
    """Daily FRED series, one column per series (public CSV, no API key)."""
    frames = []
    for series in FRED_SERIES:
        s = pd.read_csv(FRED_CSV.format(series), index_col=0, parse_dates=True, na_values=".")
        s.columns = [series]
        frames.append(s)
    df = pd.concat(frames, axis=1, sort=True)
    return df.loc[START:END]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    prices = fetch_prices()
    prices.to_csv(RAW_DIR / "prices.csv")
    print(f"prices: {prices.shape} -> {RAW_DIR / 'prices.csv'}")

    macro = fetch_fred()
    macro.to_csv(RAW_DIR / "fred.csv")
    print(f"fred: {macro.shape} -> {RAW_DIR / 'fred.csv'}")


if __name__ == "__main__":
    main()
