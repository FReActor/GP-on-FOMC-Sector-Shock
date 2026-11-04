"""Build the event table: one row per FOMC announcement, with its rate surprise.

Inputs:
    data/reference/fomc_dates.csv   FOMC announcement dates 1999-2025 (in git)
    data/raw/fred.csv               from src/fetch_data.py (needs the DGS2 column)

Output: data/processed/events.csv with columns
    event_date   announcement date from fomc_dates.csv
    trade_date   first trading day on/after event_date (differs for weekend moves)
    event_type   fomc_scheduled or fomc_unscheduled
    dgs2_before  2y Treasury yield (%) on the trading day before trade_date
    dgs2_after   2y Treasury yield (%) on trade_date
    surprise_bp  (dgs2_after - dgs2_before) * 100, in basis points

Run from the repo root:
    python -m src.build_events

Surprise measure: the daily change in the 2-year yield. It is a rough proxy
(the whole day's news is in it, not only the Fed), but it is free, goes back
to 1999, and is the measure used in the project plan.

fomc_dates.csv source: federalreserve.gov FOMC calendars. Scheduled meetings
use the last day of the meeting (the day the statement is released).
Unscheduled rows are only the between-meeting rate cuts; conference calls
with no rate action are left out. 2020-03-17/18 was cancelled.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REFERENCE_DIR = ROOT / "data" / "reference"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_fomc_dates() -> pd.DataFrame:
    dates = pd.read_csv(REFERENCE_DIR / "fomc_dates.csv", parse_dates=["date"])
    return dates.sort_values("date").reset_index(drop=True)


def load_dgs2() -> pd.Series:
    """Daily 2y yield, holidays and missing days dropped."""
    fred = pd.read_csv(RAW_DIR / "fred.csv", index_col=0, parse_dates=True)
    return fred["DGS2"].dropna().sort_index()


def build_events() -> pd.DataFrame:
    fomc = load_fomc_dates()
    dgs2 = load_dgs2()
    days = dgs2.index

    rows = []
    for _, meeting in fomc.iterrows():
        # Position of the first trading day on/after the announcement.
        i = days.searchsorted(meeting["date"])
        if i == 0 or i >= len(days):
            continue  # no yield data before or after this date
        before, after = dgs2.iloc[i - 1], dgs2.iloc[i]
        rows.append({
            "event_date": meeting["date"].date(),
            "trade_date": days[i].date(),
            "event_type": "fomc_" + meeting["type"],
            "dgs2_before": before,
            "dgs2_after": after,
            "surprise_bp": round((after - before) * 100, 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    events = build_events()
    out = PROCESSED_DIR / "events.csv"
    events.to_csv(out, index=False)
    print(f"events: {events.shape} -> {out}")
    print(events["surprise_bp"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
