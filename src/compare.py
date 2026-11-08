"""Combine every model's results into one table, and write the go/no-go note when ready.

Reads every results/<model>.csv in the shared model format (columns ticker, horizon, r2_oos,
rmse_oos, ...), e.g. linear.csv, linear_interact.csv, gp_v0.csv, gp_v1.csv, and adds two
reference rows computed here on the same test set:
    zero        predict an excess return of 0 for every row
    train_mean  predict each ticker's train-period mean (r2_oos = 0 by definition)
The zero row matters: r2_oos is measured against the train mean, so a model that just
predicts ~0 already scores above 0 when the train means don't carry over to the test years.

Output: results/summary.csv, one row per (model, ticker, horizon), with columns
    model, ticker, horizon, r2_oos, rmse_oos, hit_rate, complexity
and, once gp_v1, linear and linear_interact all exist, results/go_no_go.md.

Run from the repo root:
    python -m src.compare
"""

import numpy as np
import pandas as pd

from src import config
from src.data import load_panel, r2_oos, rmse

REQUIRED = {"ticker", "horizon", "r2_oos", "rmse_oos"}
COLS = ["model", "ticker", "horizon", "r2_oos", "rmse_oos", "hit_rate", "complexity"]


def model_files() -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(config.RESULTS.glob("*.csv")):
        if path.stem in ("summary",):
            continue
        df = pd.read_csv(path)
        if REQUIRED <= set(df.columns) and "seed" not in df.columns:
            out[path.stem] = df
    return out


def reference_rows() -> pd.DataFrame:
    p = load_panel(with_features=False)
    rows = []
    for h in config.HORIZONS:
        y = f"excess_{h}"
        d = p.dropna(subset=[y])
        ybar = d[d["split"].isin(["train", "val"])].groupby("ticker")[y].mean()
        te = d[d["split"] == "test"]
        groups = [("POOLED", te)] + list(te.groupby("ticker"))
        for ticker, g in groups:
            yb = g["ticker"].map(ybar).to_numpy()
            for model, yhat in (("zero", np.zeros(len(g))), ("train_mean", yb)):
                rows.append({"model": model, "ticker": ticker, "horizon": h,
                             "r2_oos": r2_oos(g[y], yhat, yb), "rmse_oos": rmse(g[y], yhat),
                             "hit_rate": np.nan, "complexity": 0})
    return pd.DataFrame(rows)


def go_no_go(summary: pd.DataFrame) -> str | None:
    """Pre-stated rule (docs/STAGE1_PLAN.md, week 7), on pooled excess_1d."""
    seeds_path = config.RESULTS / "gp_v1_seeds.csv"
    have = set(summary["model"])
    missing = [m for m in ("gp_v1", "linear", "linear_interact") if m not in have]
    if missing or not seeds_path.exists():
        print(f"\ngo/no-go not written yet; missing: {missing or ['gp_v1_seeds.csv']}")
        return None

    def pooled(model):
        r = summary[(summary["model"] == model) & (summary["ticker"] == "POOLED") & (summary["horizon"] == "1d")]
        return float(r["r2_oos"].iloc[0])

    gp, lin, lin_i, zero = pooled("gp_v1"), pooled("linear"), pooled("linear_interact"), pooled("zero")
    seeds = pd.read_csv(seeds_path)
    seeds = seeds[(seeds["ticker"] == "POOLED") & (seeds["horizon"] == "1d")]
    n_beat = int((seeds["r2_oos"] > lin).sum())
    go = gp > lin and gp > lin_i and n_beat >= 6
    lines = [
        "# Go / no-go (pooled, excess_1d, test set)",
        "",
        f"Split {config.SPLIT}. Rule stated in docs/STAGE1_PLAN.md before looking: GO if pooled GP v1's median",
        "r2_oos beats both `linear` and `linear_interact`, and at least 6 of 10 seeds beat `linear`.",
        "",
        "| model | r2_oos |",
        "|---|---|",
        f"| gp_v1 (median of {len(seeds)} seeds) | {gp:+.4f} |",
        f"| linear | {lin:+.4f} |",
        f"| linear_interact | {lin_i:+.4f} |",
        f"| zero forecast (reference) | {zero:+.4f} |",
        "",
        f"Seeds beating linear: {n_beat} / {len(seeds)}",
        "",
        f"**Decision: {'GO' if go else 'NO-GO'}**",
    ]
    text = "\n".join(lines) + "\n"
    (config.RESULTS / "go_no_go.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    files = model_files()
    parts = []
    for model, df in files.items():
        df = df.assign(model=model)
        for c in ("hit_rate", "complexity"):
            if c not in df.columns:
                df[c] = np.nan
        parts.append(df[COLS])
    summary = pd.concat(parts + [reference_rows()], ignore_index=True)
    summary = summary.sort_values(["horizon", "ticker", "model"]).reset_index(drop=True)
    summary.to_csv(config.RESULTS / "summary.csv", index=False)

    print(f"models: {', '.join(files)} + zero, train_mean")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summary[summary["ticker"] == "POOLED"].to_string(index=False))
    text = go_no_go(summary)
    if text:
        print("\n" + text)


if __name__ == "__main__":
    main()
