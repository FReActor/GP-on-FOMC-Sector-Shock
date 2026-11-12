"""Stage 2 models. Every model has the same shape: fit(name, train, ...) -> (predictor, info),
predictor(rows) -> yhat in decimals.

Units inside the models: y = target x 100 (percent), winsorized on the training rows. The surprise
`s` is in bp (linear models) or `s10` = s / 10 (symbolic regression, so a coefficient means
"% per 10 bp"). Other features are z-scored with the training rows' mean and sd (missing -> 0,
the training mean) and capped at +-CLIP, so no model can extrapolate far outside the training
range (in 2008 VIX reached 5+ training sd, and unbounded formulas in VIX blew up); zlb and hawk
stay 0/1.

`fomc_beta` is the industry's past-event sensitivity matched to the target: fomc_beta_raw for
ret_1d, fomc_beta_exc for excess_1d (both computed from earlier events only, see build.py).

Models
    zero              predict 0
    mean              each industry's training mean (the r2_oos benchmark, so r2_oos = 0)
    linear            per industry: y = a + b*s                       (Bernanke-Kuttner)
    linear_pooled     one model:   y = a + b*s
    linear_beta       one model:   s, s*beta_mkt, beta_mkt            (CAPM channel)
    linear_fomc       one model:   s, s*fomc_beta, fomc_beta          (industry's own Fed history)
    linear_interact_pooled  one model: s, s x every characteristic and state variable, + mains
    enet              ElasticNet on all features and all s x feature products (8 settings tuned)
    hgb               gradient-boosted trees on s and all features (black-box benchmark, 6 settings)
    gp_pooled         gplearn symbolic regression, one formula for all industries
    pysr_pooled       PySR symbolic regression (constants optimized), one formula for all industries
    pysr_sens         PySR sensitivity model: y = industry mean + s10 * g(x), where g is a free
                      formula of the characteristics, state and hawk flag (not s) found by the
                      search: the industry's response per 10 bp of surprise
    gp_industry       gplearn, one formula per industry (ff12 only)

Tuning: every tuned model picks its setting on the last INNER_VAL_YEARS of the training fold,
then refits on the whole fold. For PySR the "setting" is the formula size: the Pareto front is
fitted on the inner-fit years, the smallest size within one standard error (events) of the best
inner-tune MSE is kept (1-SE rule), and on the refit the largest formula of at most that size is used.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet

from src.v1 import settings as S
from src.v1.splits import inner_split

SCALE = 100.0
CLIP = 3.0
STATE = ["vix", "term_10y3m", "credit"]
CHARS = ["beta_mkt", "beta_2y", "fomc_beta", "bm", "size", "mom_20d", "vol_60d"]
ENET_BASE = ["s", "beta_mkt", "beta_2y", "fomc_beta", "bm", "size", "vix", "slope_2s10s", "term_10y3m",
             "credit", "ff", "mom_20d", "vol_60d", "mkt_mom_20d"]
FLAGS = ["zlb", "hawk"]
ENET_GRID = [(a, l) for a in (0.003, 0.01, 0.03, 0.1) for l in (0.2, 0.8)]          # 8 settings
HGB_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=it, min_samples_leaf=50, l2_regularization=1.0)
            for d in (2, 3) for lr, it in ((0.03, 100), (0.03, 300), (0.1, 100))]   # 6 settings

SR_INPUTS = ["s10", "beta_mkt", "beta_2y", "fomc_beta", "bm", "size", "vix", "term_10y3m", "credit",
             "mom_20d", "vol_60d", "zlb"]
SENS_X = [c for c in SR_INPUTS if c != "s10"] + ["hawk"]   # inputs of the sensitivity g(x)
GP_INDUSTRY_INPUTS = ["s10", "vix", "term_10y3m", "credit", "mom_20d", "zlb"]

GP_SETTINGS = dict(population_size=1000, generations=30, tournament_size=20,
                   # 'div' left out on purpose: gplearn's protected x / y returns 1 for |y| < 0.001,
                   # so x / s jumps for surprises near 0 (Stage 1's formulas were made of these).
                   function_set=("add", "sub", "mul", "max", "min", "abs", "neg"),
                   const_range=(-1.0, 1.0), init_depth=(2, 4), max_samples=0.9)
GP_PARSIMONY = [0.0005, 0.002, 0.008]

PYSR_SETTINGS = dict(niterations=100, populations=20, population_size=40, maxsize=25, maxdepth=8,
                     binary_operators=["+", "-", "*", "max", "min"], unary_operators=[],
                     parsimony=0.0, adaptive_parsimony_scaling=100.0, elementwise_loss="L2DistLoss()",
                     model_selection="accuracy", timeout_in_seconds=1800,
                     batching=True, batch_size=1024)   # candidates scored on random 1,024-row batches


# ---------------- design ----------------

def fomc_col(target: str) -> str:
    return "fomc_beta_raw" if target == "ret_1d" else "fomc_beta_exc"


class Design:
    """Panel rows -> model inputs, with z-scores fitted on the training rows only."""

    def __init__(self, train: pd.DataFrame, target: str = S.TARGET, shock: str = S.SHOCK):
        self.shock, self.fcol = shock, fomc_col(target)
        self.cols = [c for c in ENET_BASE if c != "s"]
        raw = self._raw(train)
        self.mu = raw.mean()
        self.sd = raw.std().replace(0, 1).fillna(1)

    def _raw(self, rows: pd.DataFrame) -> pd.DataFrame:
        return rows.assign(fomc_beta=rows[self.fcol])[self.cols].astype(float)

    def __call__(self, rows: pd.DataFrame) -> pd.DataFrame:
        x = ((self._raw(rows) - self.mu) / self.sd).fillna(0.0).clip(-CLIP, CLIP)
        s = rows[self.shock].to_numpy(float)
        x.insert(0, "s", s)
        x["s10"] = s / 10
        x["hawk"] = (s > 0).astype(float)
        x["zlb"] = rows["zlb"].astype(float).to_numpy()
        x["ticker"] = rows["ticker"].to_numpy()
        return x


def target_values(rows: pd.DataFrame, target: str, lo: float | None = None, hi: float | None = None):
    y = rows[target].to_numpy(float) * SCALE
    if lo is None:
        lo, hi = np.quantile(y, S.WINSOR)
    return np.clip(y, lo, hi), (lo, hi)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


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
    "linear_fomc": ["s", "s*fomc_beta", "fomc_beta"],
    "linear_interact": ["s", "s*vix", "s*term_10y3m", "s*credit", "s*hawk", "vix", "term_10y3m", "credit"],
    "linear_interact_pooled": ["s"] + [f"s*{c}" for c in CHARS + STATE + FLAGS] + CHARS + STATE + ["zlb"],
}
PER_INDUSTRY = {"mean", "linear", "linear_interact", "gp_industry"}


# ---------------- fitting ----------------

def fit_linear(name: str, train: pd.DataFrame, target: str):
    d = Design(train, target)
    x, (y, _) = d(train), target_values(train, target)
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
            {"complexity": len(spec) + 1,
             "describe": f"{b[0]:+.4f} " + " ".join(f"{c:+.4f}*{t}" for c, t in zip(b[1:], spec))})


def fit_simple(name: str, train: pd.DataFrame, target: str):
    if name == "zero":
        return (lambda rows: np.zeros(len(rows))), {"complexity": 0, "describe": "0"}
    y, _ = target_values(train, target)
    means = pd.Series(y, index=train.index).groupby(train["ticker"]).mean() / SCALE
    return (lambda rows: rows["ticker"].map(means).to_numpy(float)), {"complexity": 1, "describe": "train mean"}


def _full_X(x: pd.DataFrame) -> np.ndarray:
    base = [c for c in ENET_BASE if c != "s"] + FLAGS
    return np.column_stack([x["s"].to_numpy()] + [x[c].to_numpy() for c in base] +
                           [x["s"].to_numpy() * x[c].to_numpy() for c in base])


def _tune_then_refit(train, target, make, grid, X_of):
    fit_rows, tune_rows = inner_split(train)
    d = Design(fit_rows, target)
    y_fit, _ = target_values(fit_rows, target)
    y_tune = tune_rows[target].to_numpy(float) * SCALE
    scores = [float(np.mean((make(g).fit(X_of(d(fit_rows)), y_fit).predict(X_of(d(tune_rows))) - y_tune) ** 2))
              for g in grid]
    best = grid[int(np.argmin(scores))]
    d = Design(train, target)
    y, _ = target_values(train, target)
    m = make(best).fit(X_of(d(train)), y)
    return (lambda rows: m.predict(X_of(d(rows))) / SCALE), best, m


def fit_enet(train: pd.DataFrame, target: str):
    predict, best, m = _tune_then_refit(
        train, target, lambda g: ElasticNet(alpha=g[0], l1_ratio=g[1], max_iter=20000), ENET_GRID, _full_X)
    return predict, {"complexity": int((m.coef_ != 0).sum()) + 1, "describe": f"alpha={best[0]}, l1={best[1]}"}


def _hgb_X(x: pd.DataFrame) -> np.ndarray:
    return x[["s"] + [c for c in ENET_BASE if c != "s"] + FLAGS].to_numpy()


def fit_hgb(train: pd.DataFrame, target: str):
    predict, best, m = _tune_then_refit(
        train, target, lambda g: HistGradientBoostingRegressor(**g, random_state=0), HGB_GRID, _hgb_X)
    return predict, {"complexity": int(best["max_iter"]), "describe": str(best)}


# ---------------- symbolic regression ----------------

def _gp(seed: int, parsimony: float, settings: dict):
    from gplearn.genetic import SymbolicRegressor
    return SymbolicRegressor(**settings, parsimony_coefficient=parsimony, random_state=seed, n_jobs=1)


def fit_gp(train: pd.DataFrame, target: str, seed: int, inputs: list[str], settings: dict | None = None,
           grid: list[float] | None = None):
    """settings / grid are passed explicitly (not read from module globals) so worker processes,
    which re-import this module on Windows, use exactly what the caller asked for."""
    settings, grid = settings or GP_SETTINGS, grid or GP_PARSIMONY
    fit_rows, tune_rows = inner_split(train)
    d = Design(fit_rows, target)
    y_fit, _ = target_values(fit_rows, target)
    y_tune = tune_rows[target].to_numpy(float) * SCALE
    best, best_mse = None, np.inf
    for p in grid:
        g = _gp(seed, p, settings).fit(d(fit_rows)[inputs].to_numpy(), y_fit)
        mse = float(np.mean((g.predict(d(tune_rows)[inputs].to_numpy()) - y_tune) ** 2))
        if mse < best_mse:
            best, best_mse = p, mse
    d = Design(train, target)
    y, _ = target_values(train, target)
    g = _gp(seed, best, settings).fit(d(train)[inputs].to_numpy(), y)
    return (lambda rows: g.predict(d(rows)[inputs].to_numpy()) / SCALE,
            {"complexity": int(g._program.length_), "parsimony": best, "program": str(g._program),
             "inputs": ",".join(inputs), "engine": "gplearn"})


def _pysr(seed: int, settings: dict):
    from pysr import PySRRegressor
    return PySRRegressor(**settings, random_state=seed, deterministic=True, parallelism="serial",
                         verbosity=0, progress=False, temp_equation_file=True, delete_tempfiles=True)


def one_se_choice(preds: list[np.ndarray], y: np.ndarray, events) -> int:
    """Index of the smallest candidate (candidates ordered by size) whose tuning MSE is within one
    standard error of the best candidate's (Breiman's 1-SE rule). Losses are averaged within each
    event first, so the standard error counts events, not event x industry rows."""
    ev = pd.Series(np.asarray(events))
    loss = np.array([pd.Series((p - y) ** 2).groupby(ev).mean().to_numpy() for p in preds])
    mean = loss.mean(axis=1)
    best = int(np.argmin(mean))
    se = loss[best].std(ddof=1) / np.sqrt(loss.shape[1])
    return int(np.flatnonzero(mean <= mean[best] + se)[0])


SENS_LOSS = """
function sens_loss(tree, dataset::Dataset{T,L}, options) where {T,L}
    g, ok = eval_tree_array(tree, dataset.X, options)
    !ok && return L(Inf)
    return sum(abs2, g .* dataset.weights .- dataset.y) / dataset.n
end
"""


def fit_pysr(train: pd.DataFrame, target: str, seed: int, inputs: list[str] = SR_INPUTS,
             settings: dict | None = None, sens: bool = False):
    """Free PySR (y = formula(inputs)) or, with sens=True, the sensitivity model
        y = industry training mean + s10 * g(x),   x = SENS_X (characteristics, state, hawk; not s)
    where the search finds g. The surprise enters through PySR's sample weights and SENS_LOSS
    (prediction = g * weight), so g is a free formula and its output is "% per 10 bp"."""
    settings = dict(settings or PYSR_SETTINGS)
    if sens:
        inputs = SENS_X
        settings |= {"loss_function": SENS_LOSS}
        settings.pop("elementwise_loss", None)

    def prepare(rows, d, means):
        x = d(rows)
        base = rows["ticker"].map(means).to_numpy(float) if sens else 0.0
        return x[inputs], base, x["s10"].to_numpy(float)

    def make_means(rows, y):
        return pd.Series(y, index=rows.index).groupby(rows["ticker"]).mean()

    def fit_once(rows):
        d = Design(rows, target)
        y, _ = target_values(rows, target)
        means = make_means(rows, y)
        x, base, s10 = prepare(rows, d, means)
        kw = {"weights": s10} if sens else {}
        m = _pysr(seed, settings).fit(x, y - base, **kw)
        return m, d, means

    def predict_with(m, d, means, rows, i):
        x, base, s10 = prepare(rows, d, means)
        g = np.asarray(m.predict(x, index=i), float)
        return base + s10 * g if sens else g

    fit_rows, tune_rows = inner_split(train)
    m, d, means = fit_once(fit_rows)
    y_tune = tune_rows[target].to_numpy(float) * SCALE
    eq = m.equations_
    size = int(eq["complexity"].iloc[one_se_choice(
        [predict_with(m, d, means, tune_rows, i) for i in range(len(eq))], y_tune, tune_rows["event_date"])])

    m, d, means = fit_once(train)
    eq = m.equations_
    ok = np.flatnonzero(eq["complexity"].to_numpy() <= size)
    idx = int(ok[-1]) if len(ok) else 0
    formula = str(eq["sympy_format"].iloc[idx]) if "sympy_format" in eq else str(eq["equation"].iloc[idx])
    if sens:
        formula = f"mean + s10*({formula})"
    return (lambda rows: predict_with(m, d, means, rows, idx) / SCALE,
            {"complexity": int(eq["complexity"].iloc[idx]), "parsimony": size, "formula": formula,
             "inputs": ",".join(inputs + (["s10"] if sens else [])), "engine": "pysr_sens" if sens else "pysr",
             "front": " || ".join(f"{c}: {e}" for c, e in zip(eq["complexity"], eq["equation"]))})


def fit(name: str, train: pd.DataFrame, seed: int = 0, ticker: str | None = None,
        gp_settings: dict | None = None, gp_grid: list[float] | None = None, target: str = S.TARGET,
        pysr_settings: dict | None = None):
    if ticker is not None:
        train = train[train["ticker"] == ticker]
    if name in ("zero", "mean"):
        return fit_simple(name, train, target)
    if name in SPECS:
        return fit_linear(name, train, target)
    if name == "enet":
        return fit_enet(train, target)
    if name == "hgb":
        return fit_hgb(train, target)
    if name == "gp_pooled":
        return fit_gp(train, target, seed, SR_INPUTS, gp_settings, gp_grid)
    if name == "gp_industry":
        return fit_gp(train, target, seed, GP_INDUSTRY_INPUTS, gp_settings, gp_grid)
    if name == "pysr_pooled":
        return fit_pysr(train, target, seed, SR_INPUTS, pysr_settings)
    if name == "pysr_sens":
        return fit_pysr(train, target, seed, SR_INPUTS, pysr_settings, sens=True)
    raise ValueError(name)
