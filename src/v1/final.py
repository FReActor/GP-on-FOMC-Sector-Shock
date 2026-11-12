"""The one-time evaluation on the LOCKED 2016+ test set. Run this ONCE, after the model list and
every setting are frozen (commit first: the unlock log records the git commit).

    UNLOCK_TEST=1 python -m src.v1.final            # refuses without UNLOCK_TEST=1

What it does, for each target in settings.TARGETS on the main universe:
    1. trains every model in FROZEN on all dev years (1994-2015), with the same inner tuning
       (last 3 dev years) as the dev folds; stochastic models use FINAL_SEEDS seeds
    2. predicts every scheduled FOMC meeting from 2016 on, once
    3. writes results/v1/test/<universe>_<target>/preds/<model>.csv and test_summary.csv and
       results/v1/test/TEST_REPORT.md with the same metrics as the dev report

Nothing here may be re-run with changed settings and reported as the test result.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src.v1 import models as M
from src.v1 import report as R
from src.v1 import settings as S
from src.v1.splits import locked_test

FROZEN = ["zero", "mean", "linear", "linear_pooled", "linear_beta", "linear_fomc",
          "linear_interact_pooled", "enet", "hgb", "gp_pooled", "pysr_pooled", "pysr_sens"]
FINAL_SEEDS = list(range(10))


def _one(model, target, train, test, seed):
    predict, info = M.fit(model, train, seed, target=target)
    ybar = train.groupby("ticker")[target].mean()
    preds = pd.DataFrame({"event_date": test["event_date"].to_numpy(), "ticker": test["ticker"].to_numpy(),
                          "fold": 0, "seed": seed, "y": test[target].to_numpy(float),
                          "yhat": predict(test), "ybar": test["ticker"].map(ybar).to_numpy(float)})
    return preds, {"model": model, "seed": seed, "fold": 0, "ticker": "POOLED", **info}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    train, test = locked_test(universe=S.UNIVERSE)        # raises unless UNLOCK_TEST=1; logs the unlock
    out_md = ["# Stage 2 LOCKED TEST results (2016 onward)", "",
              f"Trained on 1994-2015 ({train['event_date'].nunique()} events), tested once on "
              f"{test['event_date'].nunique()} scheduled meetings from 2016 "
              f"(last {test['event_date'].max().date()}).", ""]
    for target in S.TARGETS:
        d = S.RESULTS / "test" / f"{S.UNIVERSE}_{target}"
        (d / "preds").mkdir(parents=True, exist_ok=True)
        runs = []
        for model in FROZEN:
            seeds = FINAL_SEEDS if model in ("gp_pooled", "pysr_pooled", "pysr_sens") else [0]
            with ProcessPoolExecutor(max_workers=min(a.jobs, len(seeds))) as ex:
                res = list(ex.map(_one, [model] * len(seeds), [target] * len(seeds),
                                  [train] * len(seeds), [test] * len(seeds), seeds))
            pd.concat([p for p, _ in res]).to_csv(d / "preds" / f"{model}.csv", index=False, date_format="%Y-%m-%d")
            runs += [r for _, r in res]
            print(f"{target} {model} done", flush=True)
        pd.DataFrame(runs).to_csv(d / "runs.csv", index=False)
        summ = R.summarize(R.load(d))
        summ.to_csv(d / "test_summary.csv", index=False)
        out_md += [f"## target `{target}`", "", R.main_table(summ), ""]
        f = R.formulas(d)
        if len(f):
            out_md += R.sr_section(f)
    (S.RESULTS / "test" / "TEST_REPORT.md").write_text("\n".join(out_md), encoding="utf-8")
    print("->", S.RESULTS / "test" / "TEST_REPORT.md")


if __name__ == "__main__":
    main()
