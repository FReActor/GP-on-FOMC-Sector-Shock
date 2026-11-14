"""GP (gplearn symbolic regression) of sector excess returns on the Fed surprise.

Versions:
    v0  inputs: surprise_bp only
    v1  inputs: config.FEATURES (needs data/processed/features.csv); features other than
        surprise_bp are standardized with train-period mean and std

For each scope (one POOLED model over all tickers = main result, plus one model per ticker),
horizon and seed:
    1. fit on split == "train" for each parsimony in config.GP_PARSIMONY_GRID, score MSE on split == "val"
    2. refit on train + val with the best parsimony
    3. predict every row; metrics are computed on split == "test"
The target is fit in percent (excess * 100); predictions are divided by 100 again.
GP settings (population, generations, functions, ...) live in config.GP so two machines match.

--scope picks which models to run, so the work can be split across machines:
    all      POOLED + every ticker (default)  -> gp_<v>.csv, gp_<v>_seeds.csv,
                                                 preds/gp_<v>.csv (pooled), preds/gp_<v>_ticker.csv
    pooled   POOLED only                      -> gp_<v>.csv, gp_<v>_seeds.csv, preds/gp_<v>.csv
    tickers  one model per ticker only        -> gp_<v>_per_ticker.csv, gp_<v>_per_ticker_seeds.csv,
                                                 preds/gp_<v>_per_ticker.csv
gp_<v>.csv: one row per (ticker, horizon) with median r2_oos across seeds, r2_oos_min / max,
rmse_oos, hit_rate, complexity. *_seeds.csv: one row per (ticker, horizon, seed) with the formula.
results/gp_formulas.csv collects formulas from every run (version, ticker, horizon, seed,
parsimony, length, formula); a run replaces only its own (version, ticker) rows, so the pooled
and per-ticker runs can both write to it.

--quick is a smoke test (small population, few generations, 1 seed, 1d only). It writes to
data/processed/gp_quick/ and never touches results/.

Run from the repo root:
    python -m src.models.gp --version v0
    python -m src.models.gp --version v1 --scope pooled      # one machine
    python -m src.models.gp --version v1 --scope tickers     # the other machine
    python -m src.models.gp --version v1 --scope tickers --quick   # ~1 minute check first
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import sympy as sp
from gplearn.genetic import SymbolicRegressor

from src import config
from src.data import hit_rate, load_panel, r2_oos, rmse

PARSIMONY_GRID = list(config.GP_PARSIMONY_GRID)
SCALE = 100.0  # fit in percent
TRAIN, VAL, TEST = ["train"], ["val"], ["test"]
FULL_TRAIN = TRAIN + VAL
SCOPES = ("all", "pooled", "tickers")
QUICK = dict(population_size=200, generations=5)  # --quick overrides


def make_gp(seed, parsimony, settings: dict | None = None):
    s = settings or config.GP
    return SymbolicRegressor(population_size=s["population_size"], generations=s["generations"],
        tournament_size=s["tournament_size"], function_set=s["function_set"],
        parsimony_coefficient=parsimony, const_range=s["const_range"], init_depth=s["init_depth"],
        max_samples=s["max_samples"], random_state=seed, n_jobs=1)


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
                ticker: str | None = None, settings: dict | None = None,
                grid: list[float] | None = None) -> tuple[pd.DataFrame, dict]:
    """Select parsimony on val, refit on train + val, predict all rows.

    ticker=None fits the pooled model. Returns (preds, info) where preds has columns
    event_date, ticker, horizon, split, seed, y, yhat (decimals) and info holds the chosen
    parsimony, the formula and its length. Used by src/evaluate/placebo.py as well.
    settings / grid default to config.GP / config.GP_PARSIMONY_GRID.
    """
    d, cols = design(panel, version, horizon)
    if ticker is not None:
        d = d[d["ticker"] == ticker]
    X = lambda rows: rows[cols].to_numpy()
    tr, va, full = d[d["split"].isin(TRAIN)], d[d["split"].isin(VAL)], d[d["split"].isin(FULL_TRAIN)]

    val_mse = {}
    for p in (grid or PARSIMONY_GRID):
        gp = make_gp(seed, p, settings).fit(X(tr), tr["y"].to_numpy() * SCALE)
        val_mse[p] = float(np.mean((gp.predict(X(va)) / SCALE - va["y"].to_numpy()) ** 2))
    best = min(val_mse, key=val_mse.get)

    gp = make_gp(seed, best, settings).fit(X(full), full["y"].to_numpy() * SCALE)
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


def save_formulas(seeds: pd.DataFrame, version: str, out_dir=None) -> None:
    """Write gp_formulas.csv (all versions and scopes in one file).

    Only rows with this version AND a ticker in this run are replaced, so a pooled run and a
    per-ticker run (possibly on different machines) don't erase each other's formulas.
    """
    path = (out_dir or config.RESULTS) / "gp_formulas.csv"
    new = seeds.assign(version=version)[["version", "ticker", "horizon", "seed", "parsimony", "length", "formula"]]
    if path.exists():
        old = pd.read_csv(path)
        keep = ~((old["version"] == version) & old["ticker"].isin(new["ticker"].unique()))
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(path, index=False)


def settings_tag(settings: dict, grid: list[float]) -> str:
    """Short hash of the GP settings, so changing config.GP never reuses stale cached runs."""
    blob = json.dumps({"gp": settings, "grid": grid}, sort_keys=True, default=list)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def _cache_path(version, scope, horizon, seed, tag):
    return (config.PROCESSED / "gp_cache" / f"{version}_split{config.SPLIT}_{tag}"
            / f"{scope}_{horizon}_{seed}.pkl")


def _task(panel, version, scope, horizon, seed, settings, grid, tag):
    """One (scope, horizon, seed) run, cached on disk so an interrupted run resumes."""
    path = _cache_path(version, scope, horizon, seed, tag)
    if path.exists():
        return pd.read_pickle(path)
    preds, info = fit_predict(panel, seed, version, horizon, None if scope == "POOLED" else scope,
                              settings, grid)
    res = (scope, preds, {"ticker": scope, "horizon": horizon, "seed": seed, **score(preds), **info})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(res, path)
    return res


def output_names(version: str, scope: str) -> dict[str, str]:
    """File names (relative to the output folder) for one run."""
    stem = f"gp_{version}_per_ticker" if scope == "tickers" else f"gp_{version}"
    names = {"summary": f"{stem}.csv", "seeds": f"{stem}_seeds.csv"}
    if scope in ("all", "pooled"):
        names["pooled_preds"] = f"preds/gp_{version}.csv"
    if scope == "all":
        names["ticker_preds"] = f"preds/gp_{version}_ticker.csv"
    if scope == "tickers":
        names["ticker_preds"] = f"preds/{stem}.csv"
    return names


def run(version: str, jobs: int, fresh: bool = False, scope: str = "all", quick: bool = False) -> None:
    settings, grid = dict(config.GP), list(PARSIMONY_GRID)
    seeds_list, horizons = list(config.SEEDS), list(config.HORIZONS)
    out_dir = config.RESULTS
    if quick:
        settings.update(QUICK)
        grid, seeds_list, horizons = grid[:1], seeds_list[:1], ["1d"]
        out_dir = config.PROCESSED / "gp_quick"
    tag = settings_tag(settings, grid) + ("_quick" if quick else "")

    panel = load_panel(with_features=(version != "v0"))
    scopes = {"all": ["POOLED"] + list(config.TICKERS), "pooled": ["POOLED"],
              "tickers": list(config.TICKERS)}[scope]
    tasks = [(s, h, seed) for h in horizons for s in scopes for seed in seeds_list]
    if fresh:
        for s, h, seed in tasks:
            _cache_path(version, s, h, seed, tag).unlink(missing_ok=True)
    print(f"GP {version} scope={scope}{' QUICK' if quick else ''}: split {config.SPLIT}, "
          f"{len(tasks)} runs x {len(grid) + 1} fits, population {settings['population_size']}, "
          f"generations {settings['generations']}, jobs={jobs}, cache tag {tag}")
    # stdlib process pool (joblib/loky fails when the temp path has non-ASCII characters).
    # If a worker dies (sleep, memory), restart the pool; finished runs come from the cache.
    for attempt in range(3):
        try:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futures = [ex.submit(_task, panel, version, s, h, seed, settings, grid, tag)
                           for s, h, seed in tasks]
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

    g = seeds.groupby(["ticker", "horizon"], sort=False)
    summary = g[["r2_oos", "rmse_oos", "hit_rate", "length"]].median().rename(columns={"length": "complexity"})
    summary["r2_oos_min"] = g["r2_oos"].min()
    summary["r2_oos_max"] = g["r2_oos"].max()
    summary["parsimony_mode"] = g["parsimony"].agg(lambda s: s.mode().iloc[0])
    summary.insert(0, "coef", np.nan)
    summary.insert(1, "tstat", np.nan)
    summary = summary.reset_index()

    names = output_names(version, scope)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / names["summary"], index=False)
    seeds.to_csv(out_dir / names["seeds"], index=False)
    if "pooled_preds" in names:
        pd.concat([p for s, p, _ in out if s == "POOLED"], ignore_index=True).to_csv(
            out_dir / names["pooled_preds"], index=False)
    if "ticker_preds" in names:
        pd.concat([p for s, p, _ in out if s != "POOLED"], ignore_index=True).to_csv(
            out_dir / names["ticker_preds"], index=False)
    save_formulas(seeds, version, out_dir)

    cols = ["ticker", "horizon", "r2_oos", "r2_oos_min", "r2_oos_max", "rmse_oos", "hit_rate", "complexity", "parsimony_mode"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summary[cols].sort_values(["horizon", "ticker"]).to_string(index=False))
        shown = seeds[seeds["ticker"] == "POOLED"] if scope != "tickers" else seeds[seeds["horizon"] == horizons[0]]
        print(f"\n{'Pooled' if scope != 'tickers' else 'Per-ticker'} formulas by seed:")
        print(shown[["ticker", "horizon", "seed", "r2_oos", "parsimony", "length", "formula"]].to_string(index=False))
    print(f"\nwrote {', '.join(names.values())} and gp_formulas.csv to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v0", "v1"], default="v0")
    ap.add_argument("--scope", choices=SCOPES, default="all",
                    help="all = pooled + per-ticker; pooled / tickers split the work across machines")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                    help="parallel runs (default: up to 8)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached runs in data/processed/gp_cache")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: tiny GP, 1 seed, 1d, output to data/processed/gp_quick/")
    args = ap.parse_args()
    run(args.version, args.jobs, args.fresh, args.scope, args.quick)


if __name__ == "__main__":
    main()
