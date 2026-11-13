"""Linear baseline with the GP v1 inputs and hand-picked surprise interactions.

    excess_h = a + b0 * s + sum_k c_k * z_k + sum_j d_j * s * z_j + e * s * 1{s > 0}

    s    surprise_bp (raw basis points, as in GP v1)
    z_k  the other v1 inputs in config.FEATURES (vix, slope_2s10s, beta_1y, mom_20d), standardized
         with the full-train mean and std exactly as src/models/gp.design() does
    z_j  the subset in config.INTERACT_WITH (vix, slope_2s10s), from docs/PROJECT_PLAN.md
    s * 1{s > 0}  lets hawkish and dovish surprises have different slopes

This is the "same features" comparison for GP v1 (docs/STAGE1_PLAN.md, Track C): the rows are the
same as GP v1's (rows missing any v1 input are dropped, e.g. early beta_1y), and the go/no-go rule
in src/compare.py asks GP v1 to beat both `linear` and this model.

Same scopes, fit period and metrics as src/models/linear.py: one POOLED model plus one per ticker,
OLS on split in {"train", "val"}, metrics on split == "test", r2_oos against each row's ticker
train-period mean.

Outputs:
    results/linear_interact.csv        one row per (ticker, horizon): coef / tstat (slope on
                                       surprise_bp), r2_oos, rmse_oos, hit_rate, complexity
                                       (= number of parameters), n_train, n_test, plus every
                                       coefficient as b_<term>
    results/preds/linear_interact.csv  event_date, ticker, horizon, split, y, yhat, scope

Run from the repo root:
    python -m src.models.linear_interact
"""

import numpy as np
import pandas as pd

from src import config
from src.data import load_panel
from src.models.linear import FULL_TRAIN, score


def design(panel: pd.DataFrame, horizon: str) -> tuple[pd.DataFrame, list[str]]:
    """Rows with a target and all v1 inputs, plus the regressor columns. Returns (d, terms)."""
    feats = list(config.FEATURES)
    target = f"excess_{horizon}"
    d = panel.dropna(subset=feats + [target]).copy()
    d["y"] = d[target]
    is_train = d["split"].isin(FULL_TRAIN)
    others = [c for c in feats if c != "surprise_bp"]
    for c in others:
        mu, sd = d.loc[is_train, c].mean(), d.loc[is_train, c].std()
        d[c] = (d[c] - mu) / sd
    s = d["surprise_bp"]
    terms = ["surprise_bp"] + others
    for c in config.INTERACT_WITH:
        d[f"s_x_{c}"] = s * d[c]
        terms.append(f"s_x_{c}")
    d["s_x_pos"] = s * (s > 0)
    terms.append("s_x_pos")
    return d, terms


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with intercept prepended. Returns (beta, classical t-stats), beta[0] = intercept."""
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    return beta, beta / se


def fit_predict(panel: pd.DataFrame, horizon: str, ticker: str | None = None) -> tuple[pd.DataFrame, dict]:
    d, terms = design(panel, horizon)
    if ticker is not None:
        d = d[d["ticker"] == ticker]
    full = d[d["split"].isin(FULL_TRAIN)]
    beta, t = ols(full[terms].to_numpy(float), full["y"].to_numpy(float))
    preds = d[["event_date", "ticker", "split", "y"]].copy()
    preds.insert(2, "horizon", horizon)
    preds["yhat"] = beta[0] + d[terms].to_numpy(float) @ beta[1:]
    info = {"coef": float(beta[1]), "tstat": float(t[1]), "complexity": len(beta),
            "b_intercept": float(beta[0]), **{f"b_{k}": float(b) for k, b in zip(terms, beta[1:])},
            **{f"t_{k}": float(v) for k, v in zip(terms, t[1:])}}
    return preds, info


def run() -> None:
    panel = load_panel(with_features=True)
    missing = [c for c in config.FEATURES if c not in panel.columns]
    if missing:
        raise FileNotFoundError(f"needs columns {missing}: run python -m src.build_features first")
    scopes = ["POOLED"] + list(config.TICKERS)
    rows, all_preds = [], []
    for h in config.HORIZONS:
        for s in scopes:
            preds, info = fit_predict(panel, h, None if s == "POOLED" else s)
            rows.append({"ticker": s, "horizon": h, **score(preds), **info})
            all_preds.append(preds.assign(scope=s))

    first = ["ticker", "horizon", "coef", "tstat", "r2_oos", "rmse_oos", "hit_rate", "complexity",
             "n_train", "n_test"]
    summary = pd.DataFrame(rows)
    summary = summary[first + [c for c in summary.columns if c not in first]]
    (config.RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.RESULTS / "linear_interact.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(config.RESULTS / "preds" / "linear_interact.csv", index=False)

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(f"linear_interact: split {config.SPLIT}, inputs {config.FEATURES}, "
              f"interactions s x {config.INTERACT_WITH} and s x 1{{s>0}}")
        print(summary[first].sort_values(["horizon", "ticker"]).to_string(index=False))
        pooled = summary[summary["ticker"] == "POOLED"].set_index("horizon")
        print("\nPooled coefficients (t-stat):")
        terms = [c[2:] for c in summary.columns if c.startswith("t_")]
        for k in terms:
            print(f"  {k:16s} " + "  ".join(f"{h}: {pooled.at[h, 'b_' + k]:+.6f} ({pooled.at[h, 't_' + k]:+.2f})"
                                            for h in pooled.index))


if __name__ == "__main__":
    run()
