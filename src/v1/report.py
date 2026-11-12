"""Stage 2 dev-fold report: who predicts best, is it significant, and what did GP learn?

Reads results/v1/preds/*.csv and results/v1/runs.csv (from python -m src.v1.run) and writes
results/v1/dev_summary.csv and results/v1/dev_report.md. DEV folds only (validation years
2004-2015); the 2016+ test set is never read here.

Metrics:
    r2_oos      1 - SSE(model) / SSE(training-fold industry mean), over all 4 folds' validation rows
                (Campbell-Thompson). GP: one value per seed; we report the median and min/max,
                and the seed-averaged prediction ("ensemble") as a separate row.
    hit_rate    share of rows where sign(yhat) == sign(y)
Significance (event level: the 12 industries of one Fed day share one shock, so losses are
first averaged within each event, giving one number per event):
    cw_t, cw_p  Clark-West test vs the historical mean (nested benchmark), one-sided
    vs_zero_p   one-sided paired t-test that the model's event-level MSE is below the zero forecast's

Run from the repo root:
    python -m src.v1.report
"""

import re

import numpy as np
import pandas as pd
from scipy import stats

from src.v1 import models as M
from src.v1 import settings as S

ORDER = ["zero", "mean", "linear", "linear_pooled", "linear_beta", "linear_interact",
         "linear_interact_pooled", "enet", "gp_pooled", "gp_industry"]
EXPLAIN = {
    "zero": "predict 0 (industry moves with the market)",
    "mean": "industry's training mean (the r2 benchmark)",
    "linear": "per industry: a + b*s",
    "linear_pooled": "one model: a + b*s",
    "linear_beta": "one model: s, s*beta_mkt, beta_mkt (theory)",
    "linear_interact": "per industry: s x {vix, term, credit, hawk}",
    "linear_interact_pooled": "one model: s x {betas, vix, term, credit, hawk}",
    "enet": "ElasticNet, all features and s-products",
    "gp_pooled": "GP, one formula for all industries",
    "gp_industry": "GP, one formula per industry",
}


def r2(g: pd.DataFrame) -> float:
    return float(1 - ((g["y"] - g["yhat"]) ** 2).sum() / ((g["y"] - g["ybar"]) ** 2).sum())


def event_losses(p: pd.DataFrame) -> pd.DataFrame:
    d = p.assign(e_m=(p["y"] - p["yhat"]) ** 2, e_b=(p["y"] - p["ybar"]) ** 2,
                 adj=(p["ybar"] - p["yhat"]) ** 2, e_z=p["y"] ** 2)
    return d.groupby("event_date")[["e_m", "e_b", "adj", "e_z"]].mean()


def clark_west(ev: pd.DataFrame) -> tuple[float, float]:
    f = ev["e_b"] - (ev["e_m"] - ev["adj"])
    t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)))
    return float(t), float(1 - stats.norm.cdf(t))


def vs_zero(ev: pd.DataFrame) -> float:
    d = ev["e_z"] - ev["e_m"]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(1 - stats.t.cdf(t, len(d) - 1))


def load() -> dict[str, pd.DataFrame]:
    out = {}
    for m in ORDER:
        path = S.RESULTS / "preds" / f"{m}.csv"
        if path.exists():
            out[m] = pd.read_csv(path, parse_dates=["event_date"])
    return out


def summarize(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for m, p in preds.items():
        seeds = sorted(p["seed"].unique())
        per_seed = [r2(p[p["seed"] == s]) for s in seeds]
        ens = p.groupby(["event_date", "ticker", "fold"], as_index=False).agg(
            y=("y", "first"), ybar=("ybar", "first"), yhat=("yhat", "mean"))
        ev = event_losses(ens)
        cw_t, cw_p = clark_west(ev)
        by_fold = ens.groupby("fold").apply(r2, include_groups=False)
        rows.append({"model": m, "what": EXPLAIN[m], "n_seeds": len(seeds),
                     "r2_oos": float(np.median(per_seed)), "r2_min": min(per_seed), "r2_max": max(per_seed),
                     "r2_ensemble": r2(ens),
                     **{f"r2_fold{k}": v for k, v in by_fold.items()},
                     "hit_rate": float((np.sign(ens["yhat"]) == np.sign(ens["y"])).mean()) if m != "zero" else np.nan,
                     "cw_t": cw_t, "cw_p": cw_p, "vs_zero_p": vs_zero(ev) if m != "zero" else np.nan})
    return pd.DataFrame(rows)


def gp_formulas() -> pd.DataFrame:
    runs_path = S.RESULTS / "runs.csv"
    if not runs_path.exists():
        return pd.DataFrame()
    runs = pd.read_csv(runs_path)
    if "program" not in runs:
        return pd.DataFrame()
    runs = runs[runs["model"].str.startswith("gp") & runs["program"].notna()].copy()
    if runs.empty:
        return runs
    from src.models.gp import to_formula
    runs["formula"] = [to_formula(p, i.split(","), n) for p, i, n in
                       zip(runs["program"], runs["inputs"], runs["complexity"])]
    names = sorted(set(",".join(runs["inputs"]).split(",")))
    for f in names:
        runs[f"uses_{f}"] = runs["formula"].map(lambda s, f=f: bool(re.search(rf"\b{f}\b", s)))
    runs["constant"] = ~runs[[f"uses_{f}" for f in names]].any(axis=1)
    return runs


def md(df: pd.DataFrame, fmt: str = "{:+.4f}") -> str:
    def c(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else fmt.format(v)
        return str(v)
    return ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" +
            "\n".join("| " + " | ".join(c(v) for v in r) + " |" for r in df.itertuples(index=False)))


def main() -> None:
    preds = load()
    summ = summarize(preds)
    summ.to_csv(S.RESULTS / "dev_summary.csv", index=False)
    zero = summ.set_index("model").loc["zero", "r2_oos"] if "zero" in set(summ["model"]) else np.nan
    best = summ[~summ["model"].isin(["zero", "mean"])].sort_values("r2_ensemble", ascending=False).iloc[0]

    out = ["# Stage 2 dev results (walk-forward, 2004-2015 validation years)", "",
           "Generated by `python -m src.v1.report`. Shock `mps_bp` (official-method Fed surprise), "
           "target = industry minus market return on the announcement day, 12 Ken French industries, "
           "scheduled FOMC meetings. Four expanding-window folds: train from 1994 to 2003 / 2006 / 2009 / "
           "2012, validate on the next 3 years. **The 2016+ test set is locked and was not used.**", "",
           "`r2_oos` > 0 means better than each industry's historical mean. The fair bar is the **zero** "
           f"row ({zero:+.4f}): a model only adds information if it beats predicting 0.", "",
           "## 1. Accuracy", "",
           md(summ[["model", "what", "n_seeds", "r2_oos", "r2_min", "r2_max", "r2_ensemble", "hit_rate",
                    "cw_p", "vs_zero_p"]].rename(columns={"r2_oos": "r2 (median seed)"})), "",
           "`cw_p`: Clark-West p-value vs the historical mean; `vs_zero_p`: p-value that the model's "
           "event-level error is below the zero forecast's. Both one-sided, one loss per event.", "",
           f"Best non-trivial model by r2 (ensemble): **{best['model']}** ({best['r2_ensemble']:+.4f}).", "",
           "### By fold (seed-averaged predictions)", "",
           md(summ[["model"] + [c for c in summ.columns if c.startswith("r2_fold")]]), ""]

    per_ind = []
    for m in ["linear_beta", "gp_pooled", "gp_industry", "linear"]:
        if m in preds:
            p = preds[m].groupby(["event_date", "ticker", "fold"], as_index=False).agg(
                y=("y", "first"), ybar=("ybar", "first"), yhat=("yhat", "mean"))
            per_ind.append(p.groupby("ticker").apply(r2, include_groups=False).rename(m))
    if "zero" in preds:
        per_ind.append(preds["zero"].groupby("ticker").apply(r2, include_groups=False).rename("zero"))
    if per_ind:
        out += ["## 2. By industry (r2_oos, seed-averaged)", "",
                md(pd.concat(per_ind, axis=1).reset_index().rename(columns={"index": "ticker"})), ""]

    f = gp_formulas()
    if len(f):
        out += ["## 3. What GP learned", ""]
        for model, g in f.groupby("model"):
            ucols = [c for c in g.columns if c.startswith("uses_")]
            usage = g[["constant"] + ucols].mean().rename(lambda c: c.replace("uses_", "")).sort_values(ascending=False)
            out += [f"### {model}: share of (fold, seed{', industry' if model == 'gp_industry' else ''}) "
                    f"runs whose formula uses each input ({len(g)} runs)", "",
                    md(usage.reset_index().rename(columns={"index": "input", 0: "share"}), "{:.0%}"), ""]
            last = g[g["fold"] == g["fold"].max()]
            show = last.sort_values("complexity").head(10) if model == "gp_pooled" else \
                last.sort_values(["ticker", "complexity"]).groupby("ticker").head(1)
            out += [f"Fold {int(g['fold'].max())} formulas (trained 1994-2012; output = excess return in %, "
                    "s in bp, other inputs z-scored):", "",
                    md(show[["ticker", "seed", "parsimony", "complexity", "formula"]], "{:g}"), ""]

    out += ["## How to read this", "",
            "- Effects are small by nature: on a Fed day, the surprise explains a few percent of how an "
            "industry moves relative to the market. Positive, significant out-of-sample r2 of 1-3% is a "
            "meaningful result in this literature.",
            "- Stage 1 (2y-yield proxy, SPDR ETFs) found nothing beyond the zero forecast; compare.",
            "- Interpretation needs agreement across seeds and folds, not one formula.", ""]
    (S.RESULTS / "dev_report.md").write_text("\n".join(out), encoding="utf-8")
    with pd.option_context("display.width", 220, "display.float_format", "{:+.4f}".format):
        print(summ[["model", "n_seeds", "r2_oos", "r2_ensemble", "hit_rate", "cw_p", "vs_zero_p"]].to_string(index=False))
    print(f"\n-> {S.RESULTS / 'dev_report.md'}")


if __name__ == "__main__":
    main()
