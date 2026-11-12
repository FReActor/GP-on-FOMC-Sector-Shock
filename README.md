# gp-macro-shocks

SFIC Fall 2026: can genetic programming predict how sectors react to unexpected macro shocks (starting with FOMC rate surprises)?

Plans:
- Club presentation (weeks 6–10): [docs/STAGE1_PLAN.md](docs/STAGE1_PLAN.md)
- Full two-stage plan to the GECCO 2027 paper: [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)
- Architecture and research design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Layout

```
data/raw/          downloaded data (not in git)
data/processed/    built tables (not in git)
src/fetch_data.py      download prices (yfinance) and macro series (FRED)
src/build_events.py    one row per shock event
src/build_returns.py   asset returns around each event (target)
src/build_features.py  inputs known before each event
src/data_report.py     data sanity report -> results/
src/config.py          shared paths, split (A/B), tickers, seeds: import from here
src/data.py            load_panel() and the shared metrics (r2_oos, rmse, hit_rate)
src/models/            models (linear baseline, GP, ...)
notebooks/         exploration
results/           reports, tables, figures
tests/             pytest (runs on every PR via .github/workflows/tests.yml)
docs/lit_review/   literature review
```

## Setup

```
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the repo root, in order:

```
python -m src.fetch_data
python -m src.build_events
python -m src.build_returns
python -m src.build_features
python -m src.data_report      # results/data_report.md + histogram
pytest                         # tests; real-data tests skip until the steps above have run
python -m src.models.gp --version v0   # ~25 min on 8 cores; resumes if interrupted
python -m src.models.linear    # linear baseline -> results/linear.csv
python -m src.compare          # all models + zero/train_mean -> results/summary.csv
python -m src.interpret        # which features GP uses, v1 vs v0 -> results/gp_interpretation.md
```

GP v1 can be split across two machines (same `config.GP` settings on both):

```
python -m src.models.gp --version v1 --scope tickers --quick   # ~30 s smoke test, writes to data/processed/gp_quick/
python -m src.models.gp --version v1 --scope pooled            # machine 1 -> results/gp_v1*.csv
python -m src.models.gp --version v1 --scope tickers           # machine 2 -> results/gp_v1_per_ticker*.csv
```

Column definitions: [docs/data_interface.md](docs/data_interface.md). Data checks: [results/data_report.md](results/data_report.md).

### Stage 2 (official Fed surprise, 12 Ken French industries, 1994-2026)

Design and decisions: [docs/STAGE2.md](docs/STAGE2.md). The 2016+ test set is locked; nothing below reads it.

```
python -m src.v1.fetch                              # SF Fed USMPD + Bauer-Swanson, Ken French, FRED -> data/raw/v1/
python -m src.v1.build                              # events, returns, features, panel -> data/processed/v1/, docs/DATA_CARD.md
python -m src.v1.gate                               # Gate 2 sanity check -> results/v1/gate2.md
python -m src.v1.run --models gp_pooled --quick     # ~20 s smoke test
python -m src.v1.run                                # all models on the 4 dev folds -> results/v1/preds/
python -m src.v1.report                             # -> results/v1/dev_report.md, dev_summary.csv
```

## Working together

Commit to a branch and open a pull request into `main`; ask a teammate to review.
