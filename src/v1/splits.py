"""Walk-forward dev folds and the LOCKED test set (docs/ARCHITECTURE.md section 7.1).

    from src.v1.splits import load_panel, dev_folds, inner_split, locked_test
    panel = load_panel()                      # dev rows only (event_date <= 2015-12-31)
    for f in dev_folds(panel): train, val = f["train"], f["val"]

Rules enforced here:
- Folds split by event date, so all industries of one event are on the same side.
- load_panel() never returns test-period rows. Only locked_test() does, and only when the
  environment variable UNLOCK_TEST=1 is set; every unlock is appended to results/v1/test_unlock.log
  with the time, the user and the git commit.
- inner_split() carves the last INNER_VAL_YEARS of a training set out for tuning, so a fold's
  validation years are never used to pick hyperparameters.
"""

import getpass
import os
import subprocess
from datetime import datetime, timezone

import pandas as pd

from src.v1 import settings as S


def _read(include_unscheduled: bool) -> pd.DataFrame:
    p = pd.read_csv(S.PROCESSED / "panel.csv", parse_dates=["event_date", "trade_date", "asof_date"])
    if not include_unscheduled:
        p = p[p["event_type"] == "fomc_scheduled"]
    return p.sort_values(["event_date", "ticker"]).reset_index(drop=True)


def load_panel(include_unscheduled: bool = S.INCLUDE_UNSCHEDULED) -> pd.DataFrame:
    """Development rows only. The locked test period is dropped before anything else sees it."""
    p = _read(include_unscheduled)
    return p[p["event_date"] <= S.DEV_END].reset_index(drop=True)


def dev_folds(panel: pd.DataFrame) -> list[dict]:
    if (panel["event_date"] > S.DEV_END).any():
        raise ValueError("dev_folds() got test-period rows; use load_panel()")
    out = []
    for f in S.FOLDS:
        tr = panel[panel["event_date"] <= f["train_end"]]
        va = panel[(panel["event_date"] >= f["val_start"]) & (panel["event_date"] <= f["val_end"])]
        out.append({"fold": f["fold"], "train": tr, "val": va})
    return out


def inner_split(train: pd.DataFrame, years: int = S.INNER_VAL_YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fit, tune): the last `years` calendar years of `train` become the tuning set."""
    last = train["event_date"].dt.year.max()
    tune = train["event_date"].dt.year > last - years
    return train[~tune], train[tune]


def locked_test(include_unscheduled: bool = S.INCLUDE_UNSCHEDULED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train = all dev years, test = 2016+). Refuses unless UNLOCK_TEST=1, and logs the unlock."""
    if os.environ.get("UNLOCK_TEST") != "1":
        raise PermissionError("The 2016+ test set is locked. It is opened once, after the freeze "
                              "(docs/PROJECT_PLAN.md: Dec 29). Set UNLOCK_TEST=1 to open it.")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=S.ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(S.RESULTS / "test_unlock.log", "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{getpass.getuser()}\t{sha}\n")
    p = _read(include_unscheduled)
    return p[p["event_date"] <= S.DEV_END], p[p["event_date"] >= S.TEST_START]
