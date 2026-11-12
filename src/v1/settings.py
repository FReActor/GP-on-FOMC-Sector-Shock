"""Stage 2 settings: paths, sources, sample, splits. Import from here; never hard-code them."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "v1"
PROCESSED = ROOT / "data" / "processed" / "v1"
RESULTS = ROOT / "results" / "v1"

# ---- sources (public, no key) ----
SOURCES = {
    "USMPD.xlsx": "https://www.frbsf.org/wp-content/uploads/USMPD.xlsx",
    "monetary-policy-surprises-data.xlsx":
        "https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx",
    "12_Industry_Portfolios_daily_CSV.zip":
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_daily_CSV.zip",
    "F-F_Research_Data_Factors_daily_CSV.zip":
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
}
FRED_SERIES = ["DGS2", "DGS10", "DGS3MO", "VIXCLS", "BAA10Y", "DFF"]
FRED_START = "1990-01-01"   # >= 1 year of history before the first 1994 event

# ---- sample ----
SAMPLE_START = "1994-02-04"     # first FOMC statement released on the day of the decision
USMPD_SHEET = "Monetary Events"  # statement + press conference window (what a close-to-close return sees)
INDUSTRIES = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
              "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
BENCHMARK = "MKT"               # value-weighted CRSP market = Mkt-RF + RF
MARKET_CLOSE_HOUR = 16          # announcements at/after 4 pm ET react on the next trading day

# ---- splits (docs/ARCHITECTURE.md section 7.1) ----
DEV_END = "2015-12-31"          # everything after this is the LOCKED test set
TEST_START = "2016-01-01"
FOLDS = [  # expanding window: train from SAMPLE_START to train_end, validate on the next 3 years
    dict(fold=1, train_end="2003-12-31", val_start="2004-01-01", val_end="2006-12-31"),
    dict(fold=2, train_end="2006-12-31", val_start="2007-01-01", val_end="2009-12-31"),
    dict(fold=3, train_end="2009-12-31", val_start="2010-01-01", val_end="2012-12-31"),
    dict(fold=4, train_end="2012-12-31", val_start="2013-01-01", val_end="2015-12-31"),
]
INNER_VAL_YEARS = 3             # hyperparameters are tuned on the last 3 years of each fold's train

# ---- main specification ----
SHOCK = "mps_bp"                # main surprise; see docs/DATA_CARD.md for why not mps_orth
TARGET = "excess_1d"
INCLUDE_UNSCHEDULED = False     # main spec drops emergency meetings (robustness row includes them)
WINSOR = (0.005, 0.995)         # target winsorized inside each training fold only
SEEDS = list(range(10))         # paper: 30
