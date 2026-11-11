"""Linear baseline: OLS of sector excess returns on the Fed surprise (same setup as GP v0).

    excess_h = a + b * surprise_bp + e

Same scopes as src/models/gp.py: one POOLED model over all tickers (single intercept and slope,
like the pooled GP, which also sees no ticker information), plus one model per ticker. Fit on
the full train period (split in {"train", "val"}), metrics on split == "test". r2_oos uses each
row's ticker train-period mean as the benchmark, computed exactly as in gp.score().
OLS is deterministic, so there are no seeds and no parsimony selection.

Outputs:
    results/linear.csv          one row per (ticker, horizon): coef (slope on surprise_bp, in
                                excess-return decimals per bp), tstat (classical OLS), r2_oos,
                                rmse_oos, hit_rate, complexity (= 2 parameters), intercept, n_train, n_test
    results/preds/linear.csv    event_date, ticker, horizon, split, y, yhat for pooled and
                                per-ticker models (column scope = POOLED or the ticker)

Run from the repo root:
    python -m src.models.linear
"""

import numpy as np
import pandas as pd

from src import config
from src.data import hit_rate, load_panel, r2_oos, rmse

INPUTS = ["surprise_bp"]
FULL_TRAIN, TEST = ["train", "val"], ["test"]


def score(preds: pd.DataFrame) -> dict:
    """Same as src.models.gp.score (copied so this file doesn't need gplearn):
    ȳ_train is each row's ticker mean over the full train period."""
    ybar = preds[preds["split"].isin(FULL_TRAIN)].groupby("ticker")["y"].mean()
    te = preds[preds["split"].isin(TEST)]
    return {"n_train": int(preds["split"].isin(FULL_TRAIN).sum()), "n_test": len(te),
            "r2_oos": r2_oos(te["y"], te["yhat"], te["ticker"].map(ybar).to_numpy()),
            "rmse_oos": rmse(te["y"], te["yhat"]), "hit_rate": hit_rate(te["y"], te["yhat"])}


def ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with intercept. Returns (beta = [intercept, slope], classical t-stats)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return beta, beta / se


def fit_predict(panel: pd.DataFrame, horizon: str, ticker: str | None = None) -> tuple[pd.DataFrame, dict]:
    target = f"excess_{horizon}"
    d = panel.dropna(subset=INPUTS + [target]).copy()
    if ticker is not None:
        d = d[d["ticker"] == ticker]
    d["y"] = d[target]
    full = d[d["split"].isin(FULL_TRAIN)]
    beta, t = ols(full["surprise_bp"].to_numpy(float), full["y"].to_numpy(float))
    preds = d[["event_date", "ticker", "split", "y"]].copy()
    preds.insert(2, "horizon", horizon)
    preds["yhat"] = beta[0] + beta[1] * d["surprise_bp"].to_numpy(float)
    info = {"intercept": float(beta[0]), "coef": float(beta[1]), "tstat": float(t[1])}
    return preds, info


def run() -> None:
    panel = load_panel(with_features=False)
    scopes = ["POOLED"] + list(config.TICKERS)
    rows, all_preds = [], []
    for h in config.HORIZONS:
        for s in scopes:
            preds, info = fit_predict(panel, h, None if s == "POOLED" else s)
            rows.append({"ticker": s, "horizon": h, **info, **score(preds), "complexity": 2})
            all_preds.append(preds.assign(scope=s))

    cols = ["ticker", "horizon", "coef", "tstat", "r2_oos", "rmse_oos", "hit_rate", "complexity",
            "intercept", "n_train", "n_test"]
    summary = pd.DataFrame(rows)[cols]
    (config.RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.RESULTS / "linear.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(config.RESULTS / "preds" / "linear.csv", index=False)

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(f"Linear (excess ~ surprise_bp): split {config.SPLIT}")
        print(summary.sort_values(["horizon", "ticker"]).to_string(index=False))


if __name__ == "__main__":
    run()
