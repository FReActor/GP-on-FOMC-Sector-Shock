# Data interface (for teammates)

Run once from the repo root to build everything:

```
python -m src.fetch_data      # -> data/raw/prices.csv, data/raw/fred.csv
python -m src.build_events    # -> data/processed/events.csv
python -m src.build_returns   # -> data/processed/returns.csv
python -m src.build_features  # -> data/processed/features.csv
python -m src.data_report     # -> results/data_report.md, results/fig_surprise_hist.png
pytest                        # 24 tests (real-data ones skip if data/ is empty)
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

## data/processed/features.csv — one row per (event, sector ETF), same rows as returns.csv

Everything is measured at **day -1** (the trading day before day 0), so no look-ahead.

| column | meaning |
|---|---|
| event_date, ticker | join keys to returns.csv |
| asof_date | day -1 (the date every feature is measured on) |
| macro_asof_date | FRED date the macro values came from (= asof_date unless the bond market was closed) |
| vix | VIX close, index points |
| slope_2s10s | DGS10 - DGS2, percentage points (1.0 = 100 bp) |
| beta_1y | OLS beta of the ETF's daily returns on SPY's, 252 days ending day -1 (NaN if < 126 days) |
| mom_20d | ETF return over the 20 trading days ending day -1 |
| credit | BAA10Y, Baa corporate minus 10y Treasury, pp (optional) |
| ff | DFF, effective fed funds rate, % (optional) |
| vol_60d | std of daily ETF returns over 60 days ending day -1, not annualized (optional) |
| mkt_mom_20d | SPY's 20-day return, same for every ticker of an event (optional) |

Missing: beta_1y on 34 rows (first 1999 events, first XLRE/XLC events), mom_20d on 1, vol_60d on 11.

## How to join

```python
import pandas as pd
ev = pd.read_csv("data/processed/events.csv", parse_dates=["event_date", "trade_date"])
rt = pd.read_csv("data/processed/returns.csv", parse_dates=["event_date", "trade_date"])
df = rt.merge(ev, on=["event_date", "trade_date"])   # one row per event x ticker
ft = pd.read_csv("data/processed/features.csv", parse_dates=["event_date"])
df = df.merge(ft, on=["event_date", "ticker"], how="left", validate="one_to_one")
```

## Who uses what

- **Linear baseline** (`src/models/`): regress `excess_1d` (and `excess_5d`) on `surprise_bp`, one regression per ticker, then pooled. Train on events before 2016, test on 2016+ (same split the GP will use).
- **build_features** (`src/build_features.py`): add pre-event inputs per (event_date, ticker), e.g. VIX on day -1, 2s10s slope on day -1, 1y beta to SPY, 20d momentum. **Only use data up to day -1** (no look-ahead). Output keyed on event_date + ticker so it joins to returns.csv.
