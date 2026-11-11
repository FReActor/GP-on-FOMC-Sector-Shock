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
