"""Shared settings. Import paths, split dates and tickers from here; never hard-code them.

See docs/STAGE1_PLAN.md ("Shared definitions"). Changing the split is a one-line change: SPLIT.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

SPLIT = "A"  # "A" or "B", see docs/STAGE1_PLAN.md (decide at the first meeting)
SPLITS = {
    "A": dict(train_end="2009-12-31", val_start="2007-01-01", test_start="2010-01-01", test_end="2015-12-31"),
    "B": dict(train_end="2015-12-31", val_start="2013-01-01", test_start="2016-01-01", test_end="2099-12-31"),
}
TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"] if SPLIT == "A" else \
          ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
HORIZONS = ["1d", "5d"]
SEEDS = list(range(10))
FEATURES = ["surprise_bp", "vix", "slope_2s10s", "beta_1y", "mom_20d"]

# ---- GP settings: shared so the pooled and per-ticker runs on different machines match ----
# Changing any of these starts a fresh cache automatically (see src/models/gp.py).
GP = dict(
    population_size=1000,
    generations=40,
    tournament_size=20,
    function_set=("add", "sub", "mul", "div", "max", "min", "abs", "neg"),
    const_range=(-1.0, 1.0),
    init_depth=(2, 4),
    max_samples=0.9,
)
GP_PARSIMONY_GRID = [0.0005, 0.001, 0.005, 0.01]  # picked per run on the val years
