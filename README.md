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
src/config.py          shared paths, split (A/B), tickers, seeds: import from here
src/data.py            load_panel() and the shared metrics (r2_oos, rmse, hit_rate)
src/models/            models (linear baseline, GP, ...)
notebooks/         exploration
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
python -m src.models.gp --version v0   # ~25 min on 8 cores; resumes if interrupted
```

## Working together

Commit to a branch and open a pull request into `main`; ask a teammate to review.
