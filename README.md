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

### Stage 2 (official Fed surprise, Ken French 49 industries, PySR, 1994-2026)

Design, decisions and dev results: [docs/STAGE2.md](docs/STAGE2.md); full tables:
[results/v1/STAGE2_REPORT.md](results/v1/STAGE2_REPORT.md). The 2016+ test set is locked; nothing
below reads it. PySR's first import downloads Julia (~1 GB, once).

```
python -m src.v1.fetch                                  # SF Fed USMPD + Bauer-Swanson, Ken French 12/49, FRED -> data/raw/v1/
python -m src.v1.build                                  # panels for ff12 and ff49 -> data/processed/v1/<universe>/, docs/DATA_CARD.md
python -m src.v1.gate                                   # Gate 2 sanity check -> results/v1/gate2.md
python -m src.v1.run --models linear_beta,pysr_sens --quick   # ~1 min smoke test
python -m src.v1.run                                    # main spec: ff49, ret_1d and excess_1d, all models (~3 h)
python -m src.v1.run --variant orth                     # robustness: MPS_ORTH shock; also unsched, --universe ff12
python -m src.v1.run --variant placebo0 --seeds 1       # placebo: shock permuted across events (0..19)
python -m src.v1.report                                 # -> results/v1/STAGE2_REPORT.md, fig_*.png
UNLOCK_TEST=1 python -m src.v1.final                    # the one-time test evaluation, after the freeze
```

## Working together

Commit to a branch and open a pull request into `main`; ask a teammate to review.
