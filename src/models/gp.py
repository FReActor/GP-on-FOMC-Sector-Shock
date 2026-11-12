"""GP (gplearn symbolic regression) of sector excess returns on the Fed surprise.

Versions:
    v0  inputs: surprise_bp only
    v1  inputs: config.FEATURES (needs data/processed/features.csv); features other than
        surprise_bp are standardized with train-period mean and std

For each scope (one POOLED model over all tickers = main result, plus one model per ticker),
horizon and seed:
    1. fit on split == "train" for each parsimony in PARSIMONY_GRID, score MSE on split == "val"
    2. refit on train + val with the best parsimony
    3. predict every row; metrics are computed on split == "test"
The target is fit in percent (excess * 100); predictions are divided by 100 again.

Outputs (version v0 shown):
    results/gp_v0.csv              one row per (ticker, horizon): median r2_oos across seeds,
                                   r2_oos_min / r2_oos_max, rmse_oos, hit_rate, complexity
    results/gp_v0_seeds.csv        one row per (ticker, horizon, seed), with the formula
    results/gp_formulas.csv        version, ticker, horizon, seed, parsimony, length, formula (sympy)
    results/preds/gp_v0.csv        pooled model predictions: event_date, ticker, horizon, split, seed, y, yhat
    results/preds/gp_v0_ticker.csv same, from the per-ticker models

--scope splits one run across machines (same seeds and settings either way):
    all     POOLED + every ticker (default), file names as above
    pooled  POOLED only        -> results/gp_v1_pooled.csv, gp_v1_pooled_seeds.csv, preds/gp_v1_pooled.csv
    ticker  per-ticker only    -> results/gp_v1_per_ticker.csv, gp_v1_per_ticker_seeds.csv, preds/gp_v1_per_ticker.csv
src/compare.py combines gp_v1_pooled + gp_v1_per_ticker into model gp_v1 when gp_v1.csv is absent.
gp_formulas.csv is shared: a run replaces only the rows of its own version and tickers.

Run from the repo root:
    python -m src.models.gp --version v0
    python -m src.models.gp --version v1 --scope pooled   # Charles
    python -m src.models.gp --version v1 --scope ticker   # Abhi
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import sympy as sp
from gplearn.genetic import SymbolicRegressor

from src import config
from src.data import hit_rate, load_panel, r2_oos, rmse

FUNCS = ("add", "sub", "mul", "div", "max", "min", "abs", "neg")
PARSIMONY_GRID = [0.0005, 0.001, 0.005, 0.01]
SCALE = 100.0  # fit in percent
TRAIN, VAL, TEST = ["train"], ["val"], ["test"]
FULL_TRAIN = TRAIN + VAL


def make_gp(seed, parsimony):
    return SymbolicRegressor(population_size=1000, generations=40, tournament_size=20,
        function_set=FUNCS, parsimony_coefficient=parsimony, const_range=(-1.0, 1.0),
        init_depth=(2, 4), max_samples=0.9, random_state=seed, n_jobs=1)


def inputs_for(version: str) -> list[str]:
    if version == "v0":
        return ["surprise_bp"]
    if version == "v1":
        return list(config.FEATURES)
    raise ValueError(f"unknown version {version!r}")


def design(panel: pd.DataFrame, version: str, horizon: str) -> tuple[pd.DataFrame, list[str]]:
    """Rows with a target and all inputs; non-surprise inputs standardized on full-train rows."""
    cols = inputs_for(version)
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise FileNotFoundError(f"{version} needs columns {missing}: build data/processed/features.csv first")
    target = f"excess_{horizon}"
    d = panel.dropna(subset=cols + [target]).copy()
    d["y"] = d[target]
    is_train = d["split"].isin(FULL_TRAIN)
    for c in cols:
        if c != "surprise_bp":
            mu, sd = d.loc[is_train, c].mean(), d.loc[is_train, c].std()
            d[c] = (d[c] - mu) / sd
    return d, cols


def fit_predict(panel: pd.DataFrame, seed: int, version: str = "v0", horizon: str = "1d",
                ticker: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Select parsimony on val, refit on train + val, predict all rows.

    ticker=None fits the pooled model. Returns (preds, info) where preds has columns
    event_date, ticker, horizon, split, seed, y, yhat (decimals) and info holds the chosen
    parsimony, the formula and its length. Used by src/evaluate/placebo.py as well.
    """
    d, cols = design(panel, version, horizon)
    if ticker is not None:
        d = d[d["ticker"] == ticker]
    X = lambda rows: rows[cols].to_numpy()
    tr, va, full = d[d["split"].isin(TRAIN)], d[d["split"].isin(VAL)], d[d["split"].isin(FULL_TRAIN)]

    val_mse = {}
    for p in PARSIMONY_GRID:
        gp = make_gp(seed, p).fit(X(tr), tr["y"].to_numpy() * SCALE)
        val_mse[p] = float(np.mean((gp.predict(X(va)) / SCALE - va["y"].to_numpy()) ** 2))
    best = min(val_mse, key=val_mse.get)

    gp = make_gp(seed, best).fit(X(full), full["y"].to_numpy() * SCALE)
    preds = d[["event_date", "ticker", "split", "y"]].copy()
    preds.insert(2, "horizon", horizon)
    preds.insert(4, "seed", seed)
    preds["yhat"] = gp.predict(X(d)) / SCALE
    info = {"parsimony": best, "val_mse": val_mse[best], "length": int(gp._program.length_),
            "program": str(gp._program), "inputs": ",".join(f"X{i}={c}" for i, c in enumerate(cols))}
    return preds, info


def score(preds: pd.DataFrame) -> dict:
    """Test-set metrics; ȳ_train is each row's ticker mean over the full train period."""
    ybar = preds[preds["split"].isin(FULL_TRAIN)].groupby("ticker")["y"].mean()
    te = preds[preds["split"].isin(TEST)]
    return {"n_train": int(preds["split"].isin(FULL_TRAIN).sum()), "n_test": len(te),
            "r2_oos": r2_oos(te["y"], te["yhat"], te["ticker"].map(ybar).to_numpy()),
            "rmse_oos": rmse(te["y"], te["yhat"]), "hit_rate": hit_rate(te["y"], te["yhat"])}


CONV = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b, "neg": lambda a: -a, "abs": sp.Abs, "max": sp.Max, "min": sp.Min}
MAX_SIMPLIFY_LENGTH = 60  # sympy.simplify gets very slow on bloated programs


def to_formula(program: str, cols: list[str], length: int | None = None) -> str:
    """gplearn program -> readable sympy formula with feature names instead of X0, X1, ...

    Note: gplearn's div is protected (returns 1 when |denominator| < 0.001); sympy shows plain a/b.
    """
    names = {f"X{i}": sp.Symbol(c) for i, c in enumerate(cols)}
    expr = sp.sympify(program, locals={**CONV, **names})
    if length is None or length <= MAX_SIMPLIFY_LENGTH:
        expr = sp.simplify(expr)
    return str(sp.N(expr, 4))


def save_formulas(seeds: pd.DataFrame, version: str) -> None:
    """Write results/gp_formulas.csv (all versions in one file; this version's rows replaced)."""
    path = config.RESULTS / "gp_formulas.csv"
    new = seeds.assign(version=version)[["version", "ticker", "horizon", "seed", "parsimony", "length", "formula"]]
    if path.exists():
        old = pd.read_csv(path)
        replaced = (old["version"] == version) & old["ticker"].isin(new["ticker"].unique())
        new = pd.concat([old[~replaced], new], ignore_index=True)
    new.to_csv(path, index=False)


def _cache_path(version, scope, horizon, seed):
    return config.PROCESSED / "gp_cache" / f"{version}_split{config.SPLIT}" / f"{scope}_{horizon}_{seed}.pkl"


def _task(panel, version, scope, horizon, seed):
    """One (scope, horizon, seed) run, cached on disk so an interrupted run resumes."""
    path = _cache_path(version, scope, horizon, seed)
    if path.exists():
        return pd.read_pickle(path)
    preds, info = fit_predict(panel, seed, version, horizon, None if scope == "POOLED" else scope)
    res = (scope, preds, {"ticker": scope, "horizon": horizon, "seed": seed, **score(preds), **info})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(res, path)
    return res


SCOPE_SUFFIX = {"all": "", "pooled": "_pooled", "ticker": "_per_ticker"}


def run(version: str, jobs: int, fresh: bool = False, scope: str = "all") -> None:
    panel = load_panel(with_features=(version != "v0"))
    scopes = {"all": ["POOLED"] + list(config.TICKERS), "pooled": ["POOLED"],
              "ticker": list(config.TICKERS)}[scope]
    name = f"gp_{version}{SCOPE_SUFFIX[scope]}"
    tasks = [(s, h, seed) for h in config.HORIZONS for s in scopes for seed in config.SEEDS]
    if fresh:
        for s, h, seed in tasks:
            _cache_path(version, s, h, seed).unlink(missing_ok=True)
    print(f"GP {version} (scope {scope}): split {config.SPLIT}, {len(tasks)} runs x {len(PARSIMONY_GRID) + 1} fits, jobs={jobs}")
    # stdlib process pool (joblib/loky fails when the temp path has non-ASCII characters).
    # If a worker dies (sleep, memory), restart the pool; finished runs come from the cache.
    for attempt in range(3):
        try:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futures = [ex.submit(_task, panel, version, s, h, seed) for s, h, seed in tasks]
                out = []
                for i, f in enumerate(futures, 1):
                    out.append(f.result())
                    if i % 10 == 0 or i == len(futures):
                        print(f"  {i}/{len(futures)} runs done", flush=True)
            break
        except BrokenProcessPool:
            print(f"  worker pool crashed (attempt {attempt + 1}); restarting from cache", flush=True)
    else:
        raise RuntimeError("GP run failed 3 times; rerun the command to resume from the cache")

    seeds = pd.DataFrame([row for _, _, row in out])
    cols_in = inputs_for(version)
    seeds["formula"] = [to_formula(p, cols_in, n) for p, n in zip(seeds["program"], seeds["length"])]
    pooled_preds = [p for s, p, _ in out if s == "POOLED"]
    ticker_preds = [p for s, p, _ in out if s != "POOLED"]

    g = seeds.groupby(["ticker", "horizon"], sort=False)
    summary = g[["r2_oos", "rmse_oos", "hit_rate", "length"]].median().rename(columns={"length": "complexity"})
    summary["r2_oos_min"] = g["r2_oos"].min()
    summary["r2_oos_max"] = g["r2_oos"].max()
    summary["parsimony_mode"] = g["parsimony"].agg(lambda s: s.mode().iloc[0])
    summary.insert(0, "coef", np.nan)
    summary.insert(1, "tstat", np.nan)
    summary = summary.reset_index()

    (config.RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.RESULTS / f"{name}.csv", index=False)
    seeds.to_csv(config.RESULTS / f"{name}_seeds.csv", index=False)
    if pooled_preds:
        pd.concat(pooled_preds, ignore_index=True).to_csv(config.RESULTS / "preds" / f"{name}.csv", index=False)
    if ticker_preds:
        tname = name if scope == "ticker" else f"{name}_ticker"
        pd.concat(ticker_preds, ignore_index=True).to_csv(config.RESULTS / "preds" / f"{tname}.csv", index=False)
    save_formulas(seeds, version)

    cols = ["ticker", "horizon", "r2_oos", "r2_oos_min", "r2_oos_max", "rmse_oos", "hit_rate", "complexity", "parsimony_mode"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summary[cols].sort_values(["horizon", "ticker"]).to_string(index=False))
        pooled = seeds[seeds["ticker"] == "POOLED"][["horizon", "seed", "r2_oos", "parsimony", "length", "formula"]]
        if len(pooled):
            print("\nPooled formulas by seed:")
            print(pooled.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v0", "v1"], default="v0")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                    help="parallel runs (default: up to 8)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached runs in data/processed/gp_cache")
    ap.add_argument("--scope", choices=list(SCOPE_SUFFIX), default="all",
                    help="all = POOLED + every ticker; pooled / ticker split the run across machines")
    args = ap.parse_args()
    run(args.version, args.jobs, args.fresh, args.scope)


if __name__ == "__main__":
    main()
