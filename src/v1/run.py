"""Run Stage 2 models on the 4 walk-forward DEV folds (the 2016+ test set stays locked).

    python -m src.v1.run                               # all models, settings.SEEDS
    python -m src.v1.run --models gp_pooled --seeds 30 --jobs 11
    python -m src.v1.run --models gp_pooled --quick    # tiny GP, 1 seed, -> data/processed/v1/quick/

Each (model, fold, seed[, industry]) run is cached under data/processed/v1/cache/<model>/<hash>/,
where <hash> covers every setting that affects the result, so an interrupted run resumes and a
changed setting never reuses stale results.

Outputs (results/v1/):
    preds/<model>.csv   event_date, ticker, fold, seed, y, yhat, ybar   (dev validation rows only;
                        y = excess_1d in decimals, ybar = the industry's training-fold mean)
    runs.csv            one row per (model, fold, seed[, industry]) with complexity, parsimony, formula
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src.v1 import models as M
from src.v1 import settings as S
from src.v1.splits import dev_folds, load_panel

ALL_MODELS = ["zero", "mean", "linear", "linear_pooled", "linear_beta", "linear_interact",
              "linear_interact_pooled", "enet", "gp_pooled", "gp_industry"]
STOCHASTIC = {"gp_pooled", "gp_industry"}


def config_hash(model: str, gp_settings: dict, gp_grid: list[float]) -> str:
    cfg = {"v": 2, "model": model, "shock": S.SHOCK, "target": S.TARGET, "winsor": S.WINSOR,
           "unscheduled": S.INCLUDE_UNSCHEDULED, "folds": S.FOLDS, "inner": S.INNER_VAL_YEARS}
    if model.startswith("gp"):
        cfg |= {"gp": gp_settings, "parsimony": gp_grid,
                "inputs": M.GP_POOLED_INPUTS if model == "gp_pooled" else M.GP_INDUSTRY_INPUTS}
    if model == "enet":
        cfg |= {"grid": M.ENET_GRID, "base": M.ENET_BASE}
    if model in M.SPECS:
        cfg |= {"spec": M.SPECS[model]}
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _task(model, fold, train, val, seed, ticker, cache, gp_settings, gp_grid):
    if cache.exists():
        return pd.read_pickle(cache)
    if ticker is not None:
        val = val[val["ticker"] == ticker]
    predict, info = M.fit(model, train, seed, ticker, gp_settings, gp_grid)
    tr = train if ticker is None else train[train["ticker"] == ticker]
    ybar = tr.groupby("ticker")[S.TARGET].mean()
    preds = pd.DataFrame({"event_date": val["event_date"].to_numpy(), "ticker": val["ticker"].to_numpy(),
                          "fold": fold, "seed": seed, "y": val[S.TARGET].to_numpy(float),
                          "yhat": predict(val), "ybar": val["ticker"].map(ybar).to_numpy(float)})
    run = {"model": model, "fold": fold, "seed": seed, "ticker": ticker or "POOLED", **info}
    res = (preds, run)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(res, cache)
    return res


def run(models: list[str], seeds: list[int], jobs: int, gp_settings: dict | None = None,
        gp_grid: list[float] | None = None, out_dir=None) -> None:
    gp_settings = dict(gp_settings or M.GP_SETTINGS)
    gp_grid = list(gp_grid or M.GP_PARSIMONY)
    out_dir = out_dir or S.RESULTS
    panel = load_panel()
    folds = dev_folds(panel)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.csv"
    old_runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()

    for model in models:
        h = config_hash(model, gp_settings, gp_grid)
        cache_dir = S.PROCESSED / "cache" / model / h
        tickers = sorted(panel["ticker"].unique()) if model == "gp_industry" else [None]
        model_seeds = seeds if model in STOCHASTIC else [0]
        tasks = [(f, seed, t) for f in folds for seed in model_seeds for t in tickers]
        print(f"{model} [{h}]: {len(tasks)} runs", flush=True)
        with ProcessPoolExecutor(max_workers=jobs if model in STOCHASTIC else 1) as ex:
            futs = [ex.submit(_task, model, f["fold"], f["train"], f["val"], seed, t,
                              cache_dir / f"f{f['fold']}_s{seed}_{t or 'POOLED'}.pkl", gp_settings, gp_grid)
                    for f, seed, t in tasks]
            out = []
            for i, fu in enumerate(futs, 1):
                out.append(fu.result())
                if i % 20 == 0 or i == len(futs):
                    print(f"  {i}/{len(futs)}", flush=True)
        preds = pd.concat([p for p, _ in out], ignore_index=True)
        preds.to_csv(out_dir / "preds" / f"{model}.csv", index=False, date_format="%Y-%m-%d")
        runs = pd.DataFrame([r for _, r in out]).assign(config=h)
        if len(old_runs):
            old_runs = old_runs[old_runs["model"] != model]
        old_runs = pd.concat([old_runs, runs], ignore_index=True)
        old_runs.to_csv(runs_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--seeds", type=int, default=len(S.SEEDS), help="number of seeds (0..n-1)")
    ap.add_argument("--jobs", type=int, default=max(1, min(11, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--quick", action="store_true", help="tiny GP + 1 seed (smoke test)")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",")]
    if a.quick:   # separate output folder; settings passed to the workers explicitly
        run(models, [0], a.jobs, {**M.GP_SETTINGS, "population_size": 100, "generations": 3},
            M.GP_PARSIMONY[:1], S.PROCESSED / "quick")
    else:
        run(models, list(range(a.seeds)), a.jobs)


if __name__ == "__main__":
    main()
