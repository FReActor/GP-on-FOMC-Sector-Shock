"""Compute each asset's return around each event (the prediction target).

Input:  data/raw/prices.csv, data/processed/events.csv
Output: data/processed/returns.csv with columns
    event_date, ticker, ret_1d, ret_5d, excess_1d, excess_5d

TODO (discuss at meeting):
- window definition (close before the event to 1 and 5 trading days after)
- excess return vs. SPY or vs. a beta-adjusted benchmark
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def build_returns() -> pd.DataFrame:
    raise NotImplementedError


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    returns = build_returns()
    returns.to_csv(PROCESSED_DIR / "returns.csv", index=False)


if __name__ == "__main__":
    main()
