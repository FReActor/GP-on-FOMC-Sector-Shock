"""Compute model inputs for each (event, asset) pair, known before the event.

Input:  data/raw/prices.csv, data/raw/fred.csv, data/processed/events.csv
Output: data/processed/features.csv with columns
    event_date, ticker, surprise_bp, vix, slope_2s10s, beta_1y, mom_20d, ...

TODO (discuss at meeting):
- final feature list
- make sure every feature uses only data available before the event (no look-ahead)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def build_features() -> pd.DataFrame:
    raise NotImplementedError


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features = build_features()
    features.to_csv(PROCESSED_DIR / "features.csv", index=False)


if __name__ == "__main__":
    main()
