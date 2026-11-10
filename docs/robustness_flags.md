# Robustness flags for `src/config.py`

`src/config.py` and `src/data.py` live on Person 1's branch, so these changes are proposed here rather
than committed to that file. Paste them in once it is merged; every model then picks them up through
`load_panel()` and needs no other change.

## 1. Add to `src/config.py`

```python
# ---- robustness switches (defaults = main specification) ----
TARGET = "excess_1d"           # "excess_1d" (main) or "excess_5d" (robustness row)
SCHEDULED_ONLY = False         # True drops the 7 between-meeting emergency cuts
DROP_EVENTS = []               # e.g. ["2008-10-08"], see results/data_report.md section 4
```

`HORIZONS` already covers 1d vs 5d for code that loops over both. Use `TARGET` where a script runs
only one horizon, so that the robustness row needs a one-line change.

## 2. Apply them in `load_panel()` in `src/data.py`

After the merge with events, before adding the `split` column:

```python
if config.SCHEDULED_ONLY:
    panel = panel[panel["event_type"] == "fomc_scheduled"]
if config.DROP_EVENTS:
    panel = panel[~panel["event_date"].isin(pd.to_datetime(config.DROP_EVENTS))]
```

## 3. Robustness rows for `summary.csv`

| row | setting | what it tests |
|---|---|---|
| main | defaults | |
| scheduled only | `SCHEDULED_ONLY = True` | are results driven by the 7 crisis cuts? (5 of them fall in the 2001 and 2008 train years) |
| 5-day target | `TARGET = "excess_5d"` | does the effect last or reverse within a week? |
| drop 2008-10-08 | `DROP_EVENTS = ["2008-10-08"]` | the event with a known SPY data issue and among the largest excess returns |
| extra features | `FEATURES += ["credit", "ff", "vol_60d"]` | does GP use more state information? (optional GP v2) |

## 4. Quick check after pasting

```python
from src import config, data
config.SCHEDULED_ONLY = True
p = data.load_panel()
assert (p["event_type"] == "fomc_scheduled").all()
```
