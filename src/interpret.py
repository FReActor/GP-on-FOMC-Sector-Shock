"""Interpret the GP runs: which features GP uses, example formulas, and whether v1 beats v0.

Inputs (whatever exists in results/):
    gp_v0_seeds.csv                       v0 (surprise only), pooled + per ticker
    gp_v1_seeds.csv, gp_v1_per_ticker_seeds.csv   v1 (surprise + features), pooled / per ticker
    linear.csv                            linear baseline (same test set)
The zero forecast is recomputed here with src.compare.reference_rows().

Outputs:
    results/gp_feature_usage.csv   version, scope, ticker, horizon, n_seeds, constant share,
                                   and for each feature the share of seeds whose formula uses it
    results/gp_interpretation.md   the tables below, in plain words

"Uses a feature" means the feature is still in the formula after sympy simplification, so a
term that cancels out (x - x) does not count. A formula with no feature at all is a constant:
GP decided that none of the inputs help.

Run from the repo root, after the GP runs:
    python -m src.interpret
"""

import re

import numpy as np
import pandas as pd

from src import config
from src.compare import reference_rows

FEATURES = list(config.FEATURES)
SEED_FILES = {"v0": ["gp_v0_seeds.csv"], "v1": ["gp_v1_seeds.csv", "gp_v1_per_ticker_seeds.csv"]}


def load_seeds() -> pd.DataFrame:
    parts = []
    for version, files in SEED_FILES.items():
        for name in files:
            path = config.RESULTS / name
            if path.exists():
                parts.append(pd.read_csv(path).assign(version=version))
    if not parts:
        raise FileNotFoundError("no GP seed files in results/; run src.models.gp first")
    seeds = pd.concat(parts, ignore_index=True)
    seeds = seeds.drop_duplicates(["version", "ticker", "horizon", "seed"], keep="last")
    seeds["scope"] = np.where(seeds["ticker"] == "POOLED", "pooled", "per_ticker")
    seeds["formula"] = seeds["formula"].astype(str)
    for f in FEATURES:
        seeds[f"uses_{f}"] = seeds["formula"].map(lambda s, f=f: bool(re.search(rf"\b{f}\b", s)))
    seeds["constant"] = ~seeds[[f"uses_{f}" for f in FEATURES]].any(axis=1)
    return seeds


def feature_usage(seeds: pd.DataFrame) -> pd.DataFrame:
    """Share of seeds using each feature, per (version, ticker, horizon) and per scope overall."""
    cols = ["constant"] + [f"uses_{f}" for f in FEATURES]
    by_ticker = seeds.groupby(["version", "scope", "ticker", "horizon"])[cols].mean()
    by_ticker["n_seeds"] = seeds.groupby(["version", "scope", "ticker", "horizon"]).size()
    overall = seeds.groupby(["version", "scope", "horizon"])[cols].mean()
    overall["n_seeds"] = seeds.groupby(["version", "scope", "horizon"]).size()
    overall = overall.reset_index().assign(ticker="ALL")
    out = pd.concat([by_ticker.reset_index(), overall], ignore_index=True)
    out = out.rename(columns={c: c.replace("uses_", "") for c in cols})
    return out[["version", "scope", "ticker", "horizon", "n_seeds", "constant"] + FEATURES]


def model_table(seeds: pd.DataFrame) -> pd.DataFrame:
    """Median test r2_oos per (ticker, horizon) for v0, v1, linear and the zero forecast."""
    med = seeds.groupby(["version", "ticker", "horizon"])["r2_oos"].median().unstack("version")
    med.columns = [f"gp_{c}" for c in med.columns]
    ref = reference_rows()
    zero = ref[ref["model"] == "zero"].set_index(["ticker", "horizon"])["r2_oos"].rename("zero")
    table = med.join(zero, how="left")
    lin_path = config.RESULTS / "linear.csv"
    if lin_path.exists():
        lin = pd.read_csv(lin_path).set_index(["ticker", "horizon"])["r2_oos"].rename("linear")
        table = table.join(lin, how="left")
    if "gp_v1" in table:
        v1 = seeds[seeds["version"] == "v1"]
        table["v1_seeds_beat_v0"] = v1.groupby(["ticker", "horizon"]).apply(
            lambda g: int((g["r2_oos"] > table.loc[g.name, "gp_v0"]).sum()) if "gp_v0" in table else np.nan)
        table["v1_seeds_beat_zero"] = v1.groupby(["ticker", "horizon"]).apply(
            lambda g: int((g["r2_oos"] > table.loc[g.name, "zero"]).sum()))
        table["n_v1_seeds"] = v1.groupby(["ticker", "horizon"]).size()
        for c in ("v1_seeds_beat_v0", "v1_seeds_beat_zero", "n_v1_seeds"):
            table[c] = table[c].astype("Int64")
    return table.reset_index()


def md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if v is pd.NA:
            return ""
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else fmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False))


def examples(seeds: pd.DataFrame, version: str, k: int = 3) -> pd.DataFrame:
    """Short non-constant formulas, best test r2 first. Cherry-picked on purpose: illustration only."""
    s = seeds[(seeds["version"] == version) & ~seeds["constant"] & (seeds["length"] <= 15)]
    return s.sort_values("r2_oos", ascending=False).head(k)[
        ["ticker", "horizon", "seed", "length", "r2_oos", "formula"]]


def report(seeds: pd.DataFrame, usage: pd.DataFrame, table: pd.DataFrame) -> str:
    out = ["# What GP learned", "",
           "Generated by `python -m src.interpret`. Test years 2010-2015 (split "
           f"{config.SPLIT}); r2_oos > 0 means better than each sector's 1999-2009 average.", ""]
    versions = sorted(seeds["version"].unique())
    out += [f"Runs found: " + ", ".join(
        f"{v} {sc} ({len(g)} seed-runs)" for (v, sc), g in seeds.groupby(["version", "scope"])), ""]

    out += ["## 1. Which features does GP use?", "",
            "Share of seed-runs whose final formula contains each feature. `constant` = the formula "
            "uses no input at all (GP decided nothing helps).", ""]
    overall = usage[usage["ticker"] == "ALL"].drop(columns="ticker")
    out += [md(overall, "{:.0%}"), ""]
    v1t = usage[(usage["version"] == "v1") & (usage["scope"] == "per_ticker") & (usage["ticker"] != "ALL")]
    if len(v1t):
        out += ["### v1 per ticker", "", md(v1t.drop(columns=["version", "scope"]), "{:.0%}"), ""]
        top = overall[(overall["version"] == "v1")].set_index(["scope", "horizon"])[FEATURES]
        for (scope, h), row in top.iterrows():
            ranked = row.sort_values(ascending=False)
            if ranked.iloc[0] == 0:
                out.append(f"- v1 {scope} {h}: every formula is a constant; GP used no feature.")
                continue
            out.append(f"- v1 {scope} {h}: most used **{ranked.index[0]}** ({ranked.iloc[0]:.0%} of runs), "
                       f"then {ranked.index[1]} ({ranked.iloc[1]:.0%}); surprise_bp in {row['surprise_bp']:.0%}.")
        out.append("")

    out += ["## 2. Does adding features beat surprise-only?", "",
            "Median test r2_oos across seeds. `zero` = always predict 0 (no model); a model only "
            "adds information if it beats `zero`, not just 0.", ""]
    cols = ["ticker", "horizon"] + [c for c in ["gp_v0", "gp_v1", "linear", "zero",
                                               "v1_seeds_beat_v0", "v1_seeds_beat_zero", "n_v1_seeds"]
                                    if c in table]
    out += [md(table[cols], "{:+.3f}"), ""]
    if "gp_v1" in table:
        t = table.dropna(subset=["gp_v1"])
        out += [f"- v1 beats v0 in {(t['gp_v1'] > t['gp_v0']).sum()} of {len(t)} (ticker, horizon) cells, "
                f"and beats the zero forecast in {(t['gp_v1'] > t['zero']).sum()} of {len(t)}.", ""]

    for v in versions:
        ex = examples(seeds, v)
        if len(ex):
            out += [f"## 3{'a' if v == 'v0' else 'b'}. Example {v} formulas", "",
                    "The best-scoring short non-constant formulas: picked by test score, so they "
                    "flatter GP. Use them to show what GP writes, not as evidence. In v1, features "
                    "other than surprise_bp are standardized (0 = train average, 1 = one sd); the "
                    "formula output is the excess return in percent.", "", md(ex, "{:+.3f}"), ""]

    out += ["## Notes for reading formulas", "",
            "- gplearn's division is protected: x/y returns 1 when |y| < 0.001. surprise_bp is a "
            "whole number of basis points and is exactly 0 on 9% of events, so a formula like "
            "`c/surprise_bp` predicts a fixed +1% on those days. That is an artifact, not economics.",
            "- Per-ticker models have 93 training events each; with 5 inputs GP can fit noise easily. "
            "Agreement across seeds (section 1) is a better guide than any single formula.", ""]
    return "\n".join(out)


def main() -> None:
    seeds = load_seeds()
    usage = feature_usage(seeds)
    table = model_table(seeds)
    usage.to_csv(config.RESULTS / "gp_feature_usage.csv", index=False)
    path = config.RESULTS / "gp_interpretation.md"
    path.write_text(report(seeds, usage, table), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(usage[usage["ticker"] == "ALL"].to_string(index=False))
    print(f"\n-> {path}\n-> {config.RESULTS / 'gp_feature_usage.csv'}")


if __name__ == "__main__":
    main()
