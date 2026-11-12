"""Stage 2 (paper) pipeline: official Fed surprises + Ken French industries, walk-forward dev folds,
locked 2016+ test set. Stage 1 code (src/*.py, src/models/) is untouched.

Order (from the repo root):
    python -m src.v1.fetch     # download + manifest   -> data/raw/v1/
    python -m src.v1.build     # events, returns, features, panel -> data/processed/v1/, docs/DATA_CARD.md
    python -m src.v1.gate      # Gate 2 sanity check  -> results/v1/gate2.md
    python -m src.v1.run       # models on dev folds   -> results/v1/
    python -m src.v1.report    # dev results report    -> results/v1/dev_report.md
"""
