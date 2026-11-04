# Data interface (for teammates)

Run once from the repo root to build everything:

```
python -m src.fetch_data      # -> data/raw/prices.csv, data/raw/fred.csv
python -m src.build_events    # -> data/processed/events.csv
python -m src.build_returns   # -> data/processed/returns.csv
```

## data/processed/events.csv — one row per FOMC announcement (222 rows, 1999-2025)

| column | meaning |
|---|---|
| event_date | announcement date |
| trade_date | first trading day on/after event_date (day 0) |
| event_type | `fomc_scheduled` or `fomc_unscheduled` |
| dgs2_before | 2y Treasury yield (%) on day -1 |
| dgs2_after | 2y Treasury yield (%) on day 0 |
| surprise_bp | (dgs2_after - dgs2_before) * 100, basis points. **The shock / main input.** |

## data/processed/returns.csv — one row per (event, sector ETF), ~2,140 rows

| column | meaning |
|---|---|
| event_date, trade_date | join keys to events.csv |
| ticker | XLB XLE XLF XLI XLK XLP XLU XLV XLY XLRE XLC (no SPY row) |
| ret_1d | ETF return, close day -1 -> close day 0 |
| ret_5d | ETF return, close day -1 -> close day +4 |
| spy_1d, spy_5d | SPY return over the same windows |
| excess_1d, excess_5d | ret - spy. **The prediction target.** |

Returns are decimals (0.01 = 1%). XLRE starts 2015-10, XLC 2018-06; earlier events have no row for them (rows are missing, not NaN).

## How to join

```python
import pandas as pd
ev = pd.read_csv("data/processed/events.csv", parse_dates=["event_date", "trade_date"])
rt = pd.read_csv("data/processed/returns.csv", parse_dates=["event_date", "trade_date"])
df = rt.merge(ev, on=["event_date", "trade_date"])   # one row per event x ticker
```

## Who uses what

- **Linear baseline** (`src/models/`): regress `excess_1d` (and `excess_5d`) on `surprise_bp`, one regression per ticker, then pooled. Train on events before 2016, test on 2016+ (same split the GP will use).
- **build_features** (`src/build_features.py`): add pre-event inputs per (event_date, ticker), e.g. VIX on day -1, 2s10s slope on day -1, 1y beta to SPY, 20d momentum. **Only use data up to day -1** (no look-ahead). Output keyed on event_date + ticker so it joins to returns.csv.
