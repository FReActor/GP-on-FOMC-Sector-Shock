"""Stage 2 dev-fold report: who predicts best, is it real, and what did symbolic regression learn?

Reads results/v1/<universe>_<target>[_<variant>]/ (from python -m src.v1.run) and writes
    results/v1/<universe>_<target>/dev_summary.csv     one row per model
    results/v1/STAGE2_REPORT.md                        everything below, all runs
    results/v1/fig_*.png                               figures for the paper
DEV folds only (validation years 2004-2015); the 2016+ test set is never read here.

Metrics (seed-averaged predictions unless marked; one prediction per event x industry):
    r2_oos      1 - SSE(model) / SSE(training-fold industry mean)            (Campbell-Thompson)
    r2_med      median over seeds of the per-seed r2_oos (stochastic models)
    xs_r2       cross-sectional r2: y and yhat demeaned within each event first, so only the
                ranking/spread across industries counts (what the market-wide level cannot explain)
    rank_ic     mean over events of the Spearman correlation of yhat and y across industries
Significance (event level: the industries of one Fed day share one shock, so losses are averaged
within each event first, giving one number per event; all one-sided):
    cw_p        Clark-West vs the historical mean (nested benchmark)
    zero_p      paired t-test: event-level MSE below the zero forecast's
    vs_lin_p    paired t-test: event-level MSE below linear_fomc's (the strongest linear model)
    ic_p        t-test that mean rank_ic > 0
Placebo: the same models trained on a shock randomly permuted across events (src.v1.run
--variant placeboK). p_placebo = share of placebo runs with r2_oos >= the real r2_oos.

Run from the repo root:
    python -m src.v1.report
"""

import re

import numpy as np
import pandas as pd
from scipy import stats

from src.v1 import settings as S

ORDER = ["zero", "mean", "linear", "linear_pooled", "linear_beta", "linear_fomc", "linear_char", "linear_interact",
         "linear_interact_pooled", "enet", "hgb", "rf", "mlp", "gp_pooled", "gp_industry", "gp_sens",
         "pysr_pooled", "pysr_sens", "pysr_mdl", "pysr_bag", "pysr_anchor", "pysr_mv", "pysr_enet",
         "linear_jk", "pysr_jk"]
# Forecast combinations (Rapach, Strauss & Zhou 2010): equal-weight means of seed-averaged
# forecasts, fixed in advance (not chosen on the results).
COMBOS = {"combo_linear": ["linear_beta", "linear_fomc", "linear_char"],
          "combo_sr": ["pysr_anchor", "pysr_mv", "pysr_bag", "linear_beta", "linear_fomc"],
          "combo_all": ["linear_beta", "linear_fomc", "linear_char", "linear_interact_pooled", "enet", "hgb", "rf",
                        "mlp", "gp_sens", "pysr_sens", "pysr_mdl", "pysr_bag", "pysr_anchor", "pysr_mv",
                        "pysr_enet"]}
EXPLAIN = {
    "zero": "predict 0",
    "mean": "industry's training mean (the r2 benchmark)",
    "linear": "per industry: a + b*s",
    "linear_pooled": "one model: a + b*s",
    "linear_beta": "s, s*beta_mkt, beta_mkt (CAPM channel)",
    "linear_fomc": "s, s*fomc_beta, fomc_beta (own Fed history)",
    "linear_interact": "per industry: s x {vix, term, credit, hawk}",
    "linear_interact_pooled": "s x every characteristic + state",
    "enet": "ElasticNet, all features and s-products",
    "hgb": "gradient-boosted trees (black box)",
    "gp_pooled": "gplearn GP, one formula",
    "gp_industry": "gplearn GP, one formula per industry",
    "pysr_pooled": "PySR symbolic regression, one formula",
    "pysr_sens": "PySR sensitivity: mean + s10*g(x)",
    "linear_char": "s x {beta_mkt, fomc_beta, bm, size} + mains",
    "rf": "random forest (black box)",
    "mlp": "neural net 32-16-8, 5-net average (black box)",
    "gp_sens": "gplearn sensitivity + linear scaling",
    "pysr_mdl": "PySR sensitivity, formula by MDL",
    "pysr_bag": "PySR sensitivity, 8 event-bootstrap bags",
    "pysr_anchor": "linear_beta + PySR residual sensitivity, bagged",
    "pysr_mv": "pysr_anchor + shrunk industry offsets",
    "pysr_enet": "PySR front as features + ElasticNet (FEAT-style)",
    "linear_jk": "linear_beta + s*cbi (uses 30-min stock move)",
    "pysr_jk": "pysr_anchor + cbi input (uses 30-min stock move)",
    "combo_linear": "mean of linear_beta, linear_fomc, linear_char",
    "combo_sr": "mean of pysr_anchor, pysr_mv, pysr_bag, linear_beta, linear_fomc",
    "combo_all": "mean of all 15 non-trivial models",
}
SR = {"gp_pooled", "gp_industry", "gp_sens", "pysr_pooled", "pysr_sens", "pysr_mdl", "pysr_bag", "pysr_anchor",
      "pysr_mv", "pysr_enet", "pysr_jk"}
REF = "linear_fomc"


# ---------------- metrics ----------------

def r2(g: pd.DataFrame) -> float:
    return float(1 - ((g["y"] - g["yhat"]) ** 2).sum() / ((g["y"] - g["ybar"]) ** 2).sum())


def xs_r2(g: pd.DataFrame) -> float:
    y = g["y"] - g.groupby("event_date")["y"].transform("mean")
    yh = g["yhat"] - g.groupby("event_date")["yhat"].transform("mean")
    return float(1 - ((y - yh) ** 2).sum() / (y ** 2).sum())


def rank_ic(g: pd.DataFrame) -> pd.Series:
    def one(e):
        if e["yhat"].nunique() < 2:
            return 0.0
        return stats.spearmanr(e["yhat"], e["y"]).statistic
    return g.groupby("event_date")[["y", "yhat"]].apply(one)


def event_losses(p: pd.DataFrame) -> pd.DataFrame:
    d = p.assign(e_m=(p["y"] - p["yhat"]) ** 2, e_b=(p["y"] - p["ybar"]) ** 2,
                 adj=(p["ybar"] - p["yhat"]) ** 2, e_z=p["y"] ** 2)
    return d.groupby("event_date")[["e_m", "e_b", "adj", "e_z"]].mean()


def clark_west(ev: pd.DataFrame) -> tuple[float, float]:
    f = ev["e_b"] - (ev["e_m"] - ev["adj"])
    t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)))
    return float(t), float(1 - stats.norm.cdf(t))


def paired_p(better: pd.Series, worse: pd.Series) -> float:
    """one-sided p that `better` (event-level MSE) is below `worse`"""
    d = (worse - better).dropna()
    if d.std(ddof=1) == 0:
        return np.nan
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(1 - stats.t.cdf(t, len(d) - 1))


def vs_zero(ev: pd.DataFrame) -> float:
    return paired_p(ev["e_m"], ev["e_z"])


def ensemble(p: pd.DataFrame) -> pd.DataFrame:
    return p.groupby(["event_date", "ticker", "fold"], as_index=False).agg(
        y=("y", "first"), ybar=("ybar", "first"), yhat=("yhat", "mean"))


def load(d) -> dict[str, pd.DataFrame]:
    out = {}
    for m in ORDER:
        path = d / "preds" / f"{m}.csv"
        if path.exists():
            out[m] = pd.read_csv(path, parse_dates=["event_date"])
    return out | combos(out)


def combos(preds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Equal-weight combinations; only built when every member exists (else the combo is skipped)."""
    out = {}
    key = ["event_date", "ticker", "fold"]
    for name, members in COMBOS.items():
        if not all(m in preds for m in members):
            continue
        ens = [ensemble(preds[m]).set_index(key) for m in members]
        p = ens[0][["y", "ybar"]].copy()
        p["yhat"] = pd.concat([e["yhat"] for e in ens], axis=1).mean(axis=1)
        out[name] = p.reset_index().assign(seed=0)
    return out


def summarize(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ref_ev = event_losses(ensemble(preds[REF]))["e_m"] if REF in preds else None
    rows = []
    for m, p in preds.items():
        seeds = sorted(p["seed"].unique())
        per_seed = [r2(p[p["seed"] == s]) for s in seeds]
        ens = ensemble(p)
        ev = event_losses(ens)
        ic = rank_ic(ens)
        cw_t, cw_p = clark_west(ev)
        by_fold = ens.groupby("fold").apply(r2, include_groups=False)
        rows.append({"model": m, "what": EXPLAIN.get(m, m), "n_seeds": len(seeds),
                     "r2_oos": r2(ens), "r2_med": float(np.median(per_seed)),
                     "r2_min": min(per_seed), "r2_max": max(per_seed), "xs_r2": xs_r2(ens),
                     "rank_ic": float(ic.mean()),
                     "ic_p": float(1 - stats.t.cdf(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))), len(ic) - 1))
                     if ic.std(ddof=1) > 0 else np.nan,
                     **{f"r2_fold{k}": v for k, v in by_fold.items()},
                     "cw_p": cw_p, "zero_p": vs_zero(ev) if m != "zero" else np.nan,
                     "vs_lin_p": paired_p(ev["e_m"], ref_ev) if ref_ev is not None and m != REF else np.nan})
    return pd.DataFrame(rows)


# ---------------- formulas ----------------

def formulas(d) -> pd.DataFrame:
    path = d / "runs.csv"
    if not path.exists():
        return pd.DataFrame()
    runs = pd.read_csv(path)
    runs = runs[runs["model"].isin(SR)].copy()
    if runs.empty:
        return runs
    if "formula" not in runs:
        runs["formula"] = np.nan
    if "program" not in runs:
        runs["program"] = np.nan
    gp = runs["model"].str.startswith("gp") & runs["program"].notna()
    if gp.any():
        from src.models.gp import to_formula
        runs.loc[gp, "formula"] = [to_formula(p, i.split(","), n) for p, i, n in
                                   zip(runs.loc[gp, "program"], runs.loc[gp, "inputs"], runs.loc[gp, "complexity"])]
    runs["formula"] = runs["formula"].astype(str)
    # text searched for inputs: every bag's formula for bagged models; only the SR features for
    # pysr_enet (its linear_beta part is fixed, not found by the search)
    text = runs["formula"].copy()
    if "bag_formulas" in runs:
        has = runs["bag_formulas"].notna()
        text[has] = runs.loc[has, "bag_formulas"].astype(str)
    enet = runs["model"] == "pysr_enet"
    text[enet] = text[enet].str.split("]", n=1).str[-1]
    names = sorted(set(",".join(runs["inputs"].dropna()).split(",")))
    for f in names:
        runs[f"uses_{f}"] = text.map(lambda s, f=f: bool(re.search(rf"\b{f}\b", s)))
    runs["constant"] = ~runs[[f"uses_{f}" for f in names]].any(axis=1)
    return runs


# ---------------- formatting ----------------

def md(df: pd.DataFrame, fmt: str = "{:+.4f}") -> str:
    def c(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else fmt.format(v)
        return str(v).replace("|", "\\|")
    return ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" +
            "\n".join("| " + " | ".join(c(v) for v in r) + " |" for r in df.itertuples(index=False)))


def pfmt(v) -> str:
    return "" if pd.isna(v) else ("<0.001" if v < 0.001 else f"{v:.3f}")


def main_table(summ: pd.DataFrame) -> str:
    t = summ[["model", "what", "n_seeds", "r2_oos", "r2_med", "xs_r2", "rank_ic", "cw_p", "zero_p",
              "vs_lin_p", "ic_p"]].copy()
    for c in ["cw_p", "zero_p", "vs_lin_p", "ic_p"]:
        t[c] = t[c].map(pfmt)
    t["r2_med"] = [v if n > 1 else np.nan for v, n in zip(t["r2_med"], t["n_seeds"])]
    return md(t)


def placebo(universe: str, target: str, real: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in sorted(S.RESULTS.glob(f"{universe}_{target}_placebo*")):
        for m, p in load(d).items():
            if m in ("zero", "mean"):
                continue
            ens = ensemble(p)
            rows.append({"model": m, "perm": d.name, "r2_oos": r2(ens), "xs_r2": xs_r2(ens)})
    if not rows:
        return pd.DataFrame()
    pl = pd.DataFrame(rows)
    out = []
    for m, g in pl.groupby("model"):
        if m not in set(real["model"]):
            continue
        rv = real.set_index("model").loc[m]
        out.append({"model": m, "n_placebo": len(g), "real r2": rv["r2_oos"],
                    "placebo r2 median": g["r2_oos"].median(), "placebo r2 95th": g["r2_oos"].quantile(0.95),
                    "p_placebo": float((g["r2_oos"] >= rv["r2_oos"]).mean()),
                    "real xs_r2": rv["xs_r2"], "placebo xs_r2 95th": g["xs_r2"].quantile(0.95),
                    "p_placebo_xs": float((g["xs_r2"] >= rv["xs_r2"]).mean())})
    return pd.DataFrame(out), pl


def figures(summaries: dict, placebos: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []
    keys = [k for k in summaries if k[2] == "main" and k[0] == S.UNIVERSE]
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5), squeeze=False)
        for ax, k in zip(axes[0], keys):
            s = summaries[k].set_index("model").reindex([m for m in ORDER if m in set(summaries[k]["model"])])
            s = s.drop(index=[m for m in ("mean",) if m in s.index])
            y = np.arange(len(s))
            ax.barh(y - 0.2, s["r2_oos"], 0.4, label="r2_oos", color="#3b6ea5")
            ax.barh(y + 0.2, s["xs_r2"], 0.4, label="cross-sectional r2", color="#c9822b")
            ax.set_yticks(y, s.index)
            ax.axvline(0, color="k", lw=0.8)
            if "zero" in s.index:
                ax.axvline(s.loc["zero", "r2_oos"], color="#3b6ea5", ls="--", lw=0.8)
            ax.set_xlim(max(-0.1, min(s["r2_oos"].min(), s["xs_r2"].min()) - 0.01), None)
            ax.set_title(f"{k[0]}, target {k[1]} (dev folds 2004-2015)")
            ax.invert_yaxis()
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(S.RESULTS / "fig_dev_r2.png", dpi=150)
        plt.close(fig)
        made.append("fig_dev_r2.png")
    for (u, t), (tab, pl) in placebos.items():
        if u != S.UNIVERSE or tab.empty:
            continue
        ms = [m for m in ("linear_beta", "linear_fomc", "hgb", "pysr_pooled", "pysr_sens") if m in set(tab["model"])]
        fig, axes = plt.subplots(1, len(ms), figsize=(4 * len(ms), 3.2), squeeze=False)
        for ax, m in zip(axes[0], ms):
            ax.hist(pl[pl["model"] == m]["r2_oos"], bins=12, color="#999")
            ax.axvline(tab.set_index("model").loc[m, "real r2"], color="#c0392b", lw=2, label="real shock")
            ax.set_title(f"{m}\n{t}", fontsize=9)
            ax.legend(fontsize=7)
        fig.tight_layout()
        name = f"fig_placebo_{u}_{t}.png"
        fig.savefig(S.RESULTS / name, dpi=150)
        plt.close(fig)
        made.append(name)
    return made


# ---------------- main ----------------

def sr_section(f: pd.DataFrame) -> list[str]:
    out = []
    for model, g in f.groupby("model"):
        ucols = [c for c in g.columns if c.startswith("uses_")]
        usage = g[["constant"] + ucols].mean().rename(lambda c: c.replace("uses_", "")).sort_values(ascending=False)
        usage = usage[usage > 0]
        out += [f"**{model}**: share of the {len(g)} (fold, seed) formulas that use each input; median size "
                f"{g['complexity'].median():g}", "",
                md(usage.reset_index().rename(columns={"index": "input", 0: "share"}), "{:.0%}"), ""]
        last = g[g["fold"] == g["fold"].max()].sort_values("seed")
        out += [f"Fold {int(g['fold'].max())} formulas (fold 4 = trained on 1994-2012; output in %, s10 = surprise in "
                "10 bp, other inputs z-scored, zlb 0/1):", "",
                md(last[["seed", "complexity", "formula"]], "{:g}"), ""]
    return out


def main() -> None:
    dirs = sorted(d for d in S.RESULTS.iterdir()
                  if d.is_dir() and d.name.split("_")[0] in S.UNIVERSES and (d / "preds").exists())
    summaries, placebos = {}, {}
    for d in dirs:
        parts = d.name.split("_")
        universe, target = parts[0], "_".join(parts[1:3])
        variant = "_".join(parts[3:]) or "main"
        if variant.startswith("placebo"):
            continue
        preds = load(d)
        if not preds:
            continue
        summ = summarize(preds)
        summ.to_csv(d / "dev_summary.csv", index=False)
        summaries[(universe, target, variant)] = summ
        if variant == "main":
            res = placebo(universe, target, summ)
            if isinstance(res, tuple):
                placebos[(universe, target)] = res

    out = ["# Stage 2 dev results", "",
           "Generated by `python -m src.v1.report`. Walk-forward dev folds: train from 1994 through 2003 / "
           "2006 / 2009 / 2012, validate on the next 3 years (2004-2015 in total). Shock `mps_bp` (positive "
           "= hawkish), scheduled FOMC meetings. **The 2016+ test set is locked and was not used.**", "",
           "Columns: `r2_oos` vs each industry's training mean (seed-averaged forecast); `r2_med` median "
           "over seeds; `xs_r2` cross-sectional r2 (within-event spread only); `rank_ic` mean Spearman "
           "across industries per event. p-values are one-sided with one loss per event: `cw_p` "
           f"Clark-West vs the mean, `zero_p` beats predicting 0, `vs_lin_p` beats `{REF}`, `ic_p` rank_ic > 0.", ""]
    for (u, t, v), summ in summaries.items():
        if v != "main":
            continue
        best = summ[~summ["model"].isin(["zero", "mean"])].sort_values("r2_oos", ascending=False).iloc[0]
        out += [f"## {u}, target `{t}`", "", main_table(summ), "",
                f"Best by r2_oos: **{best['model']}** ({best['r2_oos']:+.4f}); zero forecast "
                f"{summ.set_index('model').loc['zero', 'r2_oos']:+.4f}.", "",
                "By fold (r2_oos):", "",
                md(summ[["model"] + [c for c in summ.columns if c.startswith("r2_fold")]]), ""]
        if (u, t) in placebos:
            tab = placebos[(u, t)][0].copy()
            for c in ["p_placebo", "p_placebo_xs"]:
                tab[c] = tab[c].map(lambda v: f"{v:.2f}")
            out += ["Placebo (shock permuted across events):", "", md(tab), ""]
        f = formulas(S.results(u, t))
        if len(f):
            out += ["What symbolic regression learned:", ""] + sr_section(f)

    rob = [(k, s) for k, s in summaries.items() if k[2] != "main"]
    if rob:
        out += ["## Robustness (same models, one change each)", ""]
        for u, t in sorted({(k[0], k[1]) for k, _ in rob}):
            cols = {"main": summaries.get((u, t, "main"))}
            cols |= {k[2]: s for k, s in rob if k[0] == u and k[1] == t}
            tab = pd.concat({v: s.set_index("model")["r2_oos"] for v, s in cols.items() if s is not None}, axis=1)
            tab_xs = pd.concat({f"{v} xs": s.set_index("model")["xs_r2"] for v, s in cols.items() if s is not None},
                               axis=1)
            out += [f"### {u}, `{t}` (r2_oos and xs_r2)", "", md(tab.join(tab_xs).reset_index()), ""]
    figs = figures(summaries, placebos)
    out += ["## Figures", ""] + [f"![{f}]({f})" for f in figs] + [""]
    (S.RESULTS / "STAGE2_REPORT.md").write_text("\n".join(out), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.float_format", "{:+.4f}".format):
        for k, s in summaries.items():
            print("\n", k)
            print(s[["model", "n_seeds", "r2_oos", "xs_r2", "rank_ic", "cw_p", "zero_p", "vs_lin_p"]].to_string(index=False))
    print(f"\n-> {S.RESULTS / 'STAGE2_REPORT.md'}")


if __name__ == "__main__":
    main()
