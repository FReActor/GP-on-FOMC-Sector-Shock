"""Build the event table: one row per shock (FOMC meeting, crisis date).

Output: data/processed/events.csv with columns
    event_date, event_type, surprise_bp

TODO (discuss at meeting):
- source of FOMC meeting dates (federalreserve.gov calendar, hand-curated list)
- surprise measure (2y yield change on the event day for now)
- which crisis events to include besides FOMC (e.g. 2020-03 COVID)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def build_events() -> pd.DataFrame:
    raise NotImplementedError


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    events = build_events()
    events.to_csv(PROCESSED_DIR / "events.csv", index=False)


if __name__ == "__main__":
    main()
