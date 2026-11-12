"""Run Stage 2 models on the 4 walk-forward DEV folds (the 2016+ test set stays locked).

    python -m src.v1.run                                        # main: ff49, both targets, all models
    python -m src.v1.run --universe ff12 --targets excess_1d --models gp_industry
    python -m src.v1.run --models pysr_pooled --seeds 10 --jobs 6
    python -m src.v1.run --variant orth                         # robustness (see VARIANTS)
    python -m src.v1.run --variant placebo3 --models linear_fomc,pysr_pooled --seeds 1
    python -m src.v1.run --models gp_pooled,pysr_pooled --quick  # tiny SR, 1 seed -> data/processed/v1/quick/

Variants (everything else identical to the main specification):
    main       shock mps_bp, scheduled meetings
    orth       shock = official Bauer-Swanson MPS_ORTH (macro-news purged; covers all dev years)
    unsched    unscheduled meetings included
    placeboK   the shock is randomly permuted across events (seed K); industries of an event keep
               their shared (wrong) shock. A model that "finds" something here is fitting noise.

Each (model, fold, seed[, industry]) run is cached under
data/processed/v1/cache/<universe>_<target>_<variant>/<model>/<hash>/, where <hash> covers
every setting that affects the result, so an interrupted run resumes and a changed setting
never reuses stale results.

Outputs (results/v1/<universe>_<target>[_<variant>]/):
    preds/<model>.csv   event_date, ticker, fold, seed, y, yhat, ybar   (dev validation rows only;
                        y = target in decimals, ybar = the industry's training-fold mean)
    runs.csv            one row per (model, fold, seed[, industry]) with complexity, formula
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.v1 import models as M
from src.v1 import settings as S
from src.v1.splits import dev_folds, load_panel

MAIN_MODELS = ["zero", "mean", "linear", "linear_pooled", "linear_beta", "linear_fomc",
               "linear_interact_pooled", "enet", "hgb", "gp_pooled", "pysr_pooled", "pysr_sens"]
STOCHASTIC = {"gp_pooled", "gp_industry", "pysr_pooled", "pysr_sens"}
JULIA = {"pysr_pooled", "pysr_sens"}


def out_dir(universe: str, target: str, variant: str):
    return S.results(universe, target) if variant == "main" else S.RESULTS / f"{universe}_{target}_{variant}"


def panel_for(universe: str, variant: str) -> pd.DataFrame:
    p = load_panel(include_unscheduled=(variant == "unsched"), universe=universe)
    if variant == "orth":
        p = p.dropna(subset=["mps_orth_bs_bp"]).assign(mps_bp=lambda d: d["mps_orth_bs_bp"])
    elif variant.startswith("placebo"):
        k = int(variant.removeprefix("placebo"))
        ev = p.drop_duplicates("event_date")[["event_date", "mps_bp"]]
        perm = dict(zip(ev["event_date"], np.random.default_rng(1000 + k).permutation(ev["mps_bp"].to_numpy())))
        p = p.assign(mps_bp=p["event_date"].map(perm))
    elif variant != "main" and variant != "unsched":
        raise ValueError(variant)
    return p


def config_hash(model: str, target: str, gp_settings: dict, gp_grid: list[float], pysr_settings: dict) -> str:
    cfg = {"v": 5, "model": model, "target": target, "shock": S.SHOCK, "winsor": S.WINSOR, "clip": M.CLIP,
           "folds": S.FOLDS, "inner": S.INNER_VAL_YEARS, "base": M.ENET_BASE, "flags": M.FLAGS}
    if model in ("gp_pooled", "gp_industry"):
        cfg |= {"gp": gp_settings, "parsimony": gp_grid,
                "inputs": M.SR_INPUTS if model == "gp_pooled" else M.GP_INDUSTRY_INPUTS}
    if model in ("pysr_pooled", "pysr_sens"):
        cfg |= {"pysr": pysr_settings, "inputs": M.SENS_X if model == "pysr_sens" else M.SR_INPUTS,
                "sens_loss": M.SENS_LOSS if model == "pysr_sens" else None}
    if model == "enet":
        cfg |= {"grid": M.ENET_GRID}
    if model == "hgb":
        cfg |= {"grid": M.HGB_GRID}
    if model in M.SPECS:
        cfg |= {"spec": M.SPECS[model]}
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _task(model, target, fold, train, val, seed, ticker, cache, gp_settings, gp_grid, pysr_settings):
    if cache.exists():
        return pd.read_pickle(cache)
    if ticker is not None:
        val = val[val["ticker"] == ticker]
    predict, info = M.fit(model, train, seed, ticker, gp_settings, gp_grid, target, pysr_settings)
    tr = train if ticker is None else train[train["ticker"] == ticker]
    ybar = tr.groupby("ticker")[target].mean()
    preds = pd.DataFrame({"event_date": val["event_date"].to_numpy(), "ticker": val["ticker"].to_numpy(),
                          "fold": fold, "seed": seed, "y": val[target].to_numpy(float),
                          "yhat": predict(val), "ybar": val["ticker"].map(ybar).to_numpy(float)})
    run = {"model": model, "fold": fold, "seed": seed, "ticker": ticker or "POOLED", **info}
    res = (preds, run)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(res, cache)
    return res


def run(models: list[str], seeds: list[int], jobs: int, universe: str = S.UNIVERSE, target: str = S.TARGET,
        variant: str = "main", gp_settings: dict | None = None, gp_grid: list[float] | None = None,
        pysr_settings: dict | None = None, out=None, julia_jobs: int | None = None) -> None:
    gp_settings = dict(gp_settings or M.GP_SETTINGS)
    gp_grid = list(gp_grid or M.GP_PARSIMONY)
    pysr_settings = dict(pysr_settings or M.PYSR_SETTINGS)
    out = out or out_dir(universe, target, variant)
    panel = panel_for(universe, variant)
    folds = dev_folds(panel)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.csv"
    old_runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    tag = f"{universe}_{target}_{variant}" + ("_quick" if out.name == "quick" or "quick" in str(out) else "")

    for model in models:
        h = config_hash(model, target, gp_settings, gp_grid, pysr_settings)
        cache_dir = S.PROCESSED / "cache" / tag / model / h
        tickers = sorted(panel["ticker"].unique()) if model == "gp_industry" else [None]
        model_seeds = seeds if model in STOCHASTIC else [0]
        tasks = [(f, seed, t) for f in folds for seed in model_seeds for t in tickers]
        workers = 1 if model not in STOCHASTIC else (julia_jobs or min(jobs, 6)) if model in JULIA else jobs
        print(f"{tag} {model} [{h}]: {len(tasks)} runs on {workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_task, model, target, f["fold"], f["train"], f["val"], seed, t,
                              cache_dir / f"f{f['fold']}_s{seed}_{t or 'POOLED'}.pkl",
                              gp_settings, gp_grid, pysr_settings)
                    for f, seed, t in tasks]
            res = []
            for i, fu in enumerate(futs, 1):
                res.append(fu.result())
                if i % 10 == 0 or i == len(futs):
                    print(f"  {model} {i}/{len(futs)}", flush=True)
        preds = pd.concat([p for p, _ in res], ignore_index=True)
        preds.to_csv(out / "preds" / f"{model}.csv", index=False, date_format="%Y-%m-%d")
        runs = pd.DataFrame([r for _, r in res]).assign(config=h)
        if len(old_runs):
            old_runs = old_runs[old_runs["model"] != model]
        old_runs = pd.concat([old_runs, runs], ignore_index=True)
        old_runs.to_csv(runs_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MAIN_MODELS))
    ap.add_argument("--universe", default=S.UNIVERSE, choices=list(S.UNIVERSES))
    ap.add_argument("--targets", default=",".join(S.TARGETS))
    ap.add_argument("--variant", default="main")
    ap.add_argument("--seeds", type=int, default=len(S.SEEDS), help="number of seeds (0..n-1)")
    ap.add_argument("--jobs", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--julia-jobs", type=int, default=None, help="PySR workers (each loads Julia, ~1.5 GB)")
    ap.add_argument("--quick", action="store_true", help="tiny GP/PySR + 1 seed (smoke test)")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",")]
    for target in [t.strip() for t in a.targets.split(",")]:
        if a.quick:   # separate output folder; settings passed to the workers explicitly
            run(models, [0], a.jobs, a.universe, target, a.variant,
                {**M.GP_SETTINGS, "population_size": 100, "generations": 3}, M.GP_PARSIMONY[:1],
                {**M.PYSR_SETTINGS, "niterations": 3, "populations": 4},
                S.PROCESSED / "quick" / f"{a.universe}_{target}", a.julia_jobs)
        else:
            run(models, list(range(a.seeds)), a.jobs, a.universe, target, a.variant, julia_jobs=a.julia_jobs)


if __name__ == "__main__":
    main()
