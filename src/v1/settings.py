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
    "49_Industry_Portfolios_daily_CSV.zip":
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip",
    # monthly files: industry book-to-market (annual) and average firm size (monthly)
    "12_Industry_Portfolios_CSV.zip":
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_CSV.zip",
    "49_Industry_Portfolios_CSV.zip":
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_CSV.zip",
}
FRED_SERIES = ["DGS2", "DGS10", "DGS3MO", "VIXCLS", "BAA10Y", "DFF"]
FRED_START = "1990-01-01"   # >= 1 year of history before the first 1994 event

# ---- sample ----
SAMPLE_START = "1994-02-04"     # first FOMC statement released on the day of the decision
USMPD_SHEET = "Monetary Events"  # statement + press conference window (what a close-to-close return sees)
INDUSTRIES = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
              "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]   # the 12-industry universe
UNIVERSES = {"ff12": "12_Industry_Portfolios", "ff49": "49_Industry_Portfolios"}
MIN_INDUSTRY_COVERAGE = 0.99    # ff49: keep industries with returns on >= 99% of dev trading days
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
# Specified before any dev result of this design was seen (docs/STAGE2.md section 4):
#   main universe ff49; two targets, both reported:
#   ret_1d     raw industry return (the Fed's total effect on each industry)
#   excess_1d  industry minus market (the effect beyond the market, i.e. beyond "everything falls")
UNIVERSE = "ff49"
TARGETS = ["ret_1d", "excess_1d"]
TARGET = "excess_1d"            # default for code that takes one target (gate, old calls)
FOMC_BETA_SHRINK = 20           # past-event Fed sensitivity is shrunk by n / (n + 20) toward 0
FOMC_BETA_MIN = 8               # ... and needs at least 8 earlier events
INCLUDE_UNSCHEDULED = False     # main spec drops emergency meetings (robustness row includes them)
WINSOR = (0.005, 0.995)         # target winsorized inside each training fold only
SEEDS = list(range(10))         # paper: 30
PLACEBOS = 20                   # event-level permutations of the shock for the placebo test


def processed(universe: str = UNIVERSE) -> Path:
    return PROCESSED / universe


def results(universe: str = UNIVERSE, target: str = TARGET) -> Path:
    return RESULTS / f"{universe}_{target}"
