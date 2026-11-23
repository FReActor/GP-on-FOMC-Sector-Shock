# GP for Economic Shock Prediction

SFIC Fall 2026. **Can genetic programming (symbolic regression) find a formula that predicts how each sector reacts to an FOMC rate surprise, better than a linear model?**

## TL;DR

No, not at this data scale. Out-of-sample R² on 2010-2015, pooled over sectors, 1-day horizon:

| model | R² test |
|---|---:|
| zero forecast (predict 0) | +0.080 |
| GP1 (surprise only) | +0.080 |
| linear (OLS) | +0.076 |
| GP2 (surprise + features) | +0.074 |

GP1 collapses to a constant in every seed, so it ties the zero forecast. Per sector, GP2 beats the zero forecast in 4 of 18 (sector, horizon) cells, about what luck gives; the formulas that fit in-sample are the ones that fail out of sample. Positive R² here comes from the benchmark (the per-sector train mean), not from signal. Full numbers: [results/summary.csv](results/summary.csv); write-up: [docs/stage1_writeup.md](docs/stage1_writeup.md).

## Data and method

- **Events:** 222 FOMC announcements, 1999-2025 ([data/reference/fomc_dates.csv](data/reference/fomc_dates.csv)).
- **Shock:** `surprise_bp`, the announcement-day change in the 2-year Treasury yield (FRED DGS2), in bp.
- **Target:** sector ETF return minus SPY, 1-day and 5-day, for the 9 SPDR sector ETFs (XLRE and XLC are too young for this split).
- **Features (GP2):** `vix`, `slope_2s10s`, `beta_1y`, `mom_20d`, all known before the announcement.
- **Split A:** train 1999-2006, validation 2007-2009 (picks GP parsimony), test 2010-2015. 2016+ is held out for Stage 2.
- **Models:** gplearn `SymbolicRegressor` (pop 1000, 40 generations, 10 seeds; settings in [src/config.py](src/config.py)), pooled and per-sector; baselines: zero, train mean, OLS, OLS with surprise interactions.

**Every evolved formula**, with its train and test R², is in [formulas/](formulas/).

## Repo structure

```
data/reference/      FOMC dates (in git)
data/raw, processed/ downloaded and built data (gitignored, rebuilt by the pipeline)
src/                 pipeline: fetch_data, build_events, build_returns, build_features
src/models/          gp.py, linear.py, linear_interact.py
src/compare.py       all models + reference rows -> results/summary.csv
src/plots.py, interpret.py, export_formulas.py   figures, formula analysis, formulas/ export
results/             tables, per-event predictions (preds/), figures/
formulas/            every GP formula as LaTeX (markdown) + CSV
docs/                plans, write-up, data dictionary (data_interface.md)
tests/               pytest, run in CI on every PR
```

## Reproduce

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m src.fetch_data          # Yahoo prices + FRED CSVs, no API key
python -m src.build_events
python -m src.build_returns
python -m src.build_features
python -m src.models.linear
python -m src.models.linear_interact
python -m src.models.gp --version v0          # GP1, ~25 min on 8 cores
python -m src.models.gp --version v1          # GP2, several hours; --scope pooled|tickers to split
python -m src.compare
python -m src.plots
python -m src.export_formulas
pytest
```

Seeds are fixed (`config.SEEDS` = 0-9), so GP runs repeat exactly given the same data. Yahoo may revise prices, so rebuilt data can differ slightly from the frozen results (tag `stage1-freeze`).

## Team

Leads: Razeen Khan, Benny Signer. Members: Charles Cheng, Snata Mohanty, Abhishai Ganta. References and the Stage 2 plan (GECCO 2027): [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md).
