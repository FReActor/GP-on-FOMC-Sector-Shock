# gp-macro-shocks

SFIC Fall 2026: can genetic programming predict how sectors react to unexpected macro shocks (starting with FOMC rate surprises)?

## Layout

```
data/raw/          downloaded data (not in git)
data/processed/    built tables (not in git)
src/fetch_data.py      download prices (yfinance) and macro series (FRED)
src/build_events.py    one row per shock event
src/build_returns.py   asset returns around each event (target)
src/build_features.py  inputs known before each event
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
```

## Working together

Commit to a branch and open a pull request into `main`; ask a teammate to review.
