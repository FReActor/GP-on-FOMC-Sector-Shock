"""Stage 2 models. Every model has the same shape: fit(train) -> predictor, predictor(rows) -> yhat.

Targets and units: y = excess_1d in PERCENT (x100) inside the models, winsorized on the training
rows; predictions are returned in decimals. The surprise `s` stays in bp; other features are
z-scored with the training rows' mean and sd.

Models (docs/ARCHITECTURE.md section 6, plus the theory-driven pooled model):
    zero              predict 0 (the industry moves with the market)
    mean              each industry's training mean (the r2_oos benchmark, so r2_oos = 0)
    linear            per industry: y = a + b*s                       (Bernanke-Kuttner)
    linear_pooled     one model:   y = a + b*s
    linear_beta       one model:   y = a + b*s + c*s*beta_mkt + d*beta_mkt
                      (theory: excess reaction ~ (beta - 1) x market reaction)
    linear_interact   per industry: s, s*vix, s*term, s*credit, s*hawk, vix, term, credit
    linear_interact_pooled  one model: s and s x {beta_mkt, beta_2y, vix, term, credit, hawk} + mains
    enet              one model: ElasticNet on all features and all s x feature products,
                      alpha / l1_ratio picked on the inner tuning years (8 settings)
    gp_pooled         gplearn, one formula for all industries (inputs below), parsimony picked
                      on the inner tuning years (4 settings), then refit on the whole fold
    gp_industry       gplearn, one formula per industry, same procedure
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from src.v1 import settings as S
from src.v1.splits import inner_split

SCALE = 100.0
STATE = ["vix", "term_10y3m", "credit"]
GP_POOLED_INPUTS = ["s", "beta_mkt", "beta_2y", "vix", "term_10y3m", "credit", "mom_20d", "vol_60d", "mkt_mom_20d"]
GP_INDUSTRY_INPUTS = ["s", "vix", "term_10y3m", "credit", "mom_20d"]
ENET_BASE = ["s", "beta_mkt", "beta_2y", "vix", "slope_2s10s", "term_10y3m", "credit", "ff",
             "mom_20d", "vol_60d", "mkt_mom_20d"]
ENET_GRID = [(a, l) for a in (0.003, 0.01, 0.03, 0.1) for l in (0.2, 0.8)]   # 8 settings
GP_SETTINGS = dict(population_size=1000, generations=50, tournament_size=20,
                   # 'div' left out on purpose: with a continuous surprise, x / s explodes for
                   # surprises near 0 (Stage 1 showed these formulas overfit). See docs/STAGE2.md.
                   function_set=("add", "sub", "mul", "max", "min", "abs", "neg"),
                   const_range=(-1.0, 1.0), init_depth=(2, 4), max_samples=0.9)
GP_PARSIMONY = [0.001, 0.003, 0.01, 0.03]


# ---------------- design ----------------

class Design:
    """Turns panel rows into model inputs, with z-scores fitted on the training rows only."""

    def __init__(self, train: pd.DataFrame, shock: str = S.SHOCK):
        self.shock = shock
        self.cols = [c for c in ENET_BASE if c != "s"]
        self.mu = train[self.cols].mean()
        self.sd = train[self.cols].std().replace(0, 1)

    def __call__(self, rows: pd.DataFrame) -> pd.DataFrame:
        x = (rows[self.cols] - self.mu) / self.sd
        x.insert(0, "s", rows[self.shock].to_numpy(float))
        x["hawk"] = (x["s"] > 0).astype(float)
        x["ticker"] = rows["ticker"].to_numpy()
        return x


def target(rows: pd.DataFrame, lo: float | None = None, hi: float | None = None):
    y = rows[S.TARGET].to_numpy(float) * SCALE
    if lo is None:
        lo, hi = np.quantile(y, S.WINSOR)
    return np.clip(y, lo, hi), (lo, hi)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def _olsp(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b[0] + X @ b[1:]


def _terms(x: pd.DataFrame, spec: list[str]) -> np.ndarray:
    """spec items: a column name, or 'a*b' for a product."""
    cols = []
    for t in spec:
        if "*" in t:
            a, b = t.split("*")
            cols.append(x[a].to_numpy() * x[b].to_numpy())
        else:
            cols.append(x[t].to_numpy())
    return np.column_stack(cols)


SPECS = {
    "linear": ["s"],
    "linear_pooled": ["s"],
    "linear_beta": ["s", "s*beta_mkt", "beta_mkt"],
    "linear_interact": ["s", "s*vix", "s*term_10y3m", "s*credit", "s*hawk", "vix", "term_10y3m", "credit"],
    "linear_interact_pooled": ["s", "s*beta_mkt", "s*beta_2y", "s*vix", "s*term_10y3m", "s*credit", "s*hawk",
                               "beta_mkt", "beta_2y", "vix", "term_10y3m", "credit"],
}
PER_INDUSTRY = {"mean", "linear", "linear_interact", "gp_industry"}


# ---------------- fitting ----------------

def fit_linear(name: str, train: pd.DataFrame):
    d = Design(train)
    x, (y, _) = d(train), target(train)
    spec = SPECS[name]
    if name in PER_INDUSTRY:
        coefs = {t: _ols(_terms(x[x["ticker"] == t], spec), y[(x["ticker"] == t).to_numpy()])
                 for t in x["ticker"].unique()}
        def predict(rows):
            xr = d(rows)
            out = np.zeros(len(rows))
            for t, b in coefs.items():
                m = (xr["ticker"] == t).to_numpy()
                if m.any():
                    out[m] = _olsp(_terms(xr[m], spec), b)
            return out / SCALE
        return predict, {"complexity": len(spec) + 1, "describe": "per-industry OLS"}
    b = _ols(_terms(x, spec), y)
    return (lambda rows: _olsp(_terms(d(rows), spec), b) / SCALE,
            {"complexity": len(spec) + 1, "describe": " + ".join(f"{c:+.4f}*{t}" for c, t in zip(b[1:], spec))})


def fit_simple(name: str, train: pd.DataFrame):
    if name == "zero":
        return (lambda rows: np.zeros(len(rows))), {"complexity": 0, "describe": "0"}
    y, _ = target(train)
    means = pd.Series(y, index=train.index).groupby(train["ticker"]).mean() / SCALE
    return (lambda rows: rows["ticker"].map(means).to_numpy(float)), {"complexity": 1, "describe": "train mean"}


def _enet_X(x: pd.DataFrame) -> np.ndarray:
    base = [c for c in ENET_BASE if c != "s"] + ["hawk"]
    return np.column_stack([x["s"].to_numpy()] + [x[c].to_numpy() for c in base] +
                           [x["s"].to_numpy() * x[c].to_numpy() for c in base])


def fit_enet(train: pd.DataFrame):
    fit_rows, tune_rows = inner_split(train)
    d = Design(fit_rows)
    y_fit, lim = target(fit_rows)
    y_tune = tune_rows[S.TARGET].to_numpy(float) * SCALE
    best, best_mse = None, np.inf
    for a, l in ENET_GRID:
        m = ElasticNet(alpha=a, l1_ratio=l, max_iter=20000).fit(_enet_X(d(fit_rows)), y_fit)
        mse = np.mean((m.predict(_enet_X(d(tune_rows))) - y_tune) ** 2)
        if mse < best_mse:
            best, best_mse = (a, l), mse
    d = Design(train)
    y, _ = target(train)
    m = ElasticNet(alpha=best[0], l1_ratio=best[1], max_iter=20000).fit(_enet_X(d(train)), y)
    return (lambda rows: m.predict(_enet_X(d(rows))) / SCALE,
            {"complexity": int((m.coef_ != 0).sum()) + 1, "describe": f"alpha={best[0]}, l1={best[1]}"})


# ---------------- GP ----------------

def _gp(seed: int, parsimony: float, settings: dict):
    from gplearn.genetic import SymbolicRegressor
    return SymbolicRegressor(**settings, parsimony_coefficient=parsimony, random_state=seed, n_jobs=1)


def fit_gp(train: pd.DataFrame, seed: int, inputs: list[str], settings: dict | None = None,
           grid: list[float] | None = None):
    """settings / grid are passed explicitly (not read from module globals) so that worker
    processes, which re-import this module on Windows, use exactly what the caller asked for."""
    settings, grid = settings or GP_SETTINGS, grid or GP_PARSIMONY
    fit_rows, tune_rows = inner_split(train)
    d = Design(fit_rows)
    y_fit, _ = target(fit_rows)
    y_tune = tune_rows[S.TARGET].to_numpy(float) * SCALE
    best, best_mse = None, np.inf
    for p in grid:
        g = _gp(seed, p, settings).fit(d(fit_rows)[inputs].to_numpy(), y_fit)
        mse = float(np.mean((g.predict(d(tune_rows)[inputs].to_numpy()) - y_tune) ** 2))
        if mse < best_mse:
            best, best_mse = p, mse
    d = Design(train)
    y, _ = target(train)
    g = _gp(seed, best, settings).fit(d(train)[inputs].to_numpy(), y)
    program = str(g._program)
    return (lambda rows: g.predict(d(rows)[inputs].to_numpy()) / SCALE,
            {"complexity": int(g._program.length_), "parsimony": best, "program": program,
             "inputs": ",".join(inputs)})


def fit(name: str, train: pd.DataFrame, seed: int = 0, ticker: str | None = None,
        gp_settings: dict | None = None, gp_grid: list[float] | None = None):
    if ticker is not None:
        train = train[train["ticker"] == ticker]
    if name in ("zero", "mean"):
        return fit_simple(name, train)
    if name in SPECS:
        return fit_linear(name, train)
    if name == "enet":
        return fit_enet(train)
    if name == "gp_pooled":
        return fit_gp(train, seed, GP_POOLED_INPUTS, gp_settings, gp_grid)
    if name == "gp_industry":
        return fit_gp(train, seed, GP_INDUSTRY_INPUTS, gp_settings, gp_grid)
    raise ValueError(name)
