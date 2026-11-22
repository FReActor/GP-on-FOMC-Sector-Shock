"""Slide figures for the Stage 1 presentation (frozen results, split A, test 2010-2015).

Reads results/summary.csv, results/preds/*.csv, results/gp_feature_usage.csv and the processed
panel, and writes PNGs to results/figures/:

    fig1_pooled_r2.png        main result: pooled test r2_oos by model and horizon (slide 8)
    fig2_vs_zero_heatmap.png  per-ticker r2_oos minus the zero forecast, every model (slides 8/11)
    fig3_overfit.png          GP per-ticker seed-runs: train r2 vs test r2_oos (slide 9)
    fig4_xle_formula.png      the XLE 1d formula Min(0.022/s^2, 0.0121/s) on the data (slide 9)
    fig5_zero_artifact.png    train-period vs test-period mean excess return per ticker (slide 10)
    fig6_feature_usage.png    share of GP2 seed-runs using each input (slide 11)

Run from the repo root:
    python -m src.plots
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

from src import config
from src.data import load_panel

OUT = config.RESULTS / "figures"

# Reference categorical palette (light mode), fixed order; zero forecast is a neutral gray.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = ["zero", "linear", "linear_interact", "gp_v0", "gp_v1"]
COLORS = {"zero": "#9a9893", "linear": "#2a78d6", "linear_interact": "#eb6834",
          "gp_v0": "#1baf7a", "gp_v1": "#6250d6"}
LABELS = {"zero": "zero forecast", "linear": "linear", "linear_interact": "linear + features",
          "gp_v0": "GP1 (surprise only)", "gp_v1": "GP2 (+ features)"}
# Slide names: gp_v0 is presented as GP1, gp_v1 as GP2.
SHORT = {"linear": "Linear", "linear_interact": "Linear\n+ features",
         "gp_v0": "GP1\n(surprise only)", "gp_v1": "GP2\n(+ features)"}
MUTED = {"linear": "#a9c6ea", "linear_interact": "#f5bfa6", "gp_v0": "#a6dcc6"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#c8362f", "#f3f2ef", "#2a78d6"])

plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print(f"wrote results/figures/{name}")


def fig1_pooled(summary):
    """Pooled 1-day result: the zero forecast is drawn as the bar to clear, not as another model.
    GP2 (the main model) in full color, the others muted. 5-day is left to the slide text
    (every model is within 0.01 of zero there)."""
    s = summary[(summary["ticker"] == "POOLED") & (summary["horizon"] == "1d")]
    r2 = {m: float(s[s.model == m]["r2_oos"].iloc[0]) for m in MODELS}
    models = MODELS[1:]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(models))
    for xi, m in zip(x, models):
        color = COLORS[m] if m == "gp_v1" else MUTED[m]
        ax.bar(xi, r2[m], 0.6, color=color, edgecolor="white", linewidth=1)
        ax.text(xi, r2[m] - 0.003, f"{r2[m]:+.3f}", ha="center", va="top", fontsize=12,
                color="white" if m == "gp_v1" else INK, fontweight="bold" if m == "gp_v1" else None)
    ax.axhline(r2["zero"], color=INK, linewidth=2, linestyle="--")
    ax.text(len(models) - 0.5, r2["zero"] + 0.002, f"always predict 0 = {r2['zero']:+.3f}",
            ha="right", va="bottom", fontsize=12, color=INK, fontweight="bold")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x, [SHORT[m] for m in models], fontsize=12)
    ax.set_ylabel("R² on test years 2010–2015 (1-day)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 0.1)
    ax.set_xlim(-0.5, len(models) - 0.5)
    save(fig, "fig1_pooled_r2.png")


def fig2_heatmap(summary):
    s = summary[(summary["ticker"] != "POOLED") & summary["model"].isin(MODELS)]
    zero = s[s.model == "zero"].set_index(["ticker", "horizon"])["r2_oos"]
    models = MODELS[1:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    norm = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)
    for ax, h in zip(axes, config.HORIZONS):
        mat = np.array([[float(s[(s.model == m) & (s.ticker == t) & (s.horizon == h)]["r2_oos"].iloc[0])
                         - zero[(t, h)] for t in config.TICKERS] for m in models])
        im = ax.imshow(np.clip(mat, -0.4, 0.4), cmap=DIVERGING, norm=norm, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(mat[i, j]) > 0.25 else INK)
        ax.set_xticks(range(len(config.TICKERS)), config.TICKERS, rotation=45)
        ax.set_yticks(range(len(models)), [LABELS[m] for m in models])
        ax.set_title(f"{h} horizon: beats zero in {int((mat > 0).sum())} of {mat.size} cells",
                     loc="left", fontsize=12, color=INK)
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("R² minus zero forecast (blue = better)")
    fig.suptitle("Per sector: most cells are worse than predicting 0", x=0.06, ha="left",
                 fontsize=14, color=INK)
    save(fig, "fig2_vs_zero_heatmap.png")


def _r2(y, yhat, ybar):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - ybar) ** 2)


def overfit_table() -> pd.DataFrame:
    """In-sample (train+val) r2 vs test r2_oos for every per-ticker GP seed-run."""
    rows = []
    for version, fname in (("gp_v0", "gp_v0_ticker.csv"), ("gp_v1", "gp_v1_per_ticker.csv")):
        p = pd.read_csv(config.RESULTS / "preds" / fname)
        for (t, h, seed), g in p.groupby(["ticker", "horizon", "seed"]):
            tr = g[g["split"].isin(["train", "val"])]
            te = g[g["split"] == "test"]
            ybar = tr["y"].mean()
            if tr["yhat"].std() < 1e-12:  # constant formula: no fit to show
                continue
            rows.append(dict(model=version, ticker=t, horizon=h, seed=seed,
                             r2_train=_r2(tr["y"], tr["yhat"], ybar), r2_test=_r2(te["y"], te["yhat"], ybar)))
    return pd.DataFrame(rows)


def fig3_overfit():
    d = overfit_table()
    d.to_csv(config.RESULTS / "gp_overfit.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for m in ("gp_v0", "gp_v1"):
        g = d[d.model == m]
        ax.scatter(g["r2_train"], g["r2_test"].clip(lower=-1), s=40, color=COLORS[m], alpha=0.75,
                   edgecolor="white", linewidth=1, label=f"{LABELS[m]}, {len(g)} runs")
    lim = max(0.5, float(d["r2_train"].max()) + 0.05)
    ax.plot([0, lim], [0, lim], color=INK2, linestyle="--", linewidth=1)
    ax.text(lim * 0.70, lim * 0.62, "test = train", color=INK2, fontsize=9, rotation=0)
    ax.axhline(0, color=INK2, linewidth=1)
    share = (d["r2_test"] < 0).mean()
    ax.set_xlabel("R² on training years (1999–2009)")
    ax.set_ylabel("R² on test years (2010–2015), clipped at −1")
    ax.set_title(f"GP formulas fit the past, not the future\n{share:.0%} of non-constant runs score below 0 on test",
                 loc="left", fontsize=13, color=INK)
    ax.legend(loc="lower right", fontsize=9.5)
    save(fig, "fig3_overfit.png")


def fig4_xle():
    p = pd.read_csv(config.RESULTS / "preds" / "gp_v0_ticker.csv", parse_dates=["event_date"])
    g = p[(p.ticker == "XLE") & (p.horizon == "1d") & (p.seed == 8)]
    ev = pd.read_csv(config.PROCESSED / "events.csv", parse_dates=["event_date"])[["event_date", "surprise_bp"]]
    g = g.merge(ev, on="event_date", how="left")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for split, color, lab in ((["train", "val"], "#2a78d6", "train events 1999-2009"),
                              (["test"], "#eb6834", "test events 2010-2015")):
        h = g[g["split"].isin(split)]
        ax.scatter(h["surprise_bp"], h["y"] * 100, s=34, color=color, alpha=0.75, edgecolor="white",
                   linewidth=1, label=lab)
    # gplearn's protected division returns 1 for x/0, so the formula is ~0 for s != 0 and
    # jumps to -1% exactly when s == 0 (3 train events, 8 test events).
    curve = g.drop_duplicates("surprise_bp").sort_values("surprise_bp")
    for side in (curve[curve.surprise_bp < 0], curve[curve.surprise_bp > 0]):
        ax.plot(side["surprise_bp"], side["yhat"] * 100, color=INK, linewidth=2)
    at0 = curve[curve.surprise_bp == 0]
    ax.scatter(at0["surprise_bp"], at0["yhat"] * 100, s=120, marker="D", color=INK, zorder=5,
               label="GP formula Min(0.022/s², 0.0121/s)")
    ax.annotate("s = 0: protected division (x/0 = 1)\n→ predicts -1%", xy=(0, -1), xytext=(6, -4.5),
                fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
    n0 = g[g.surprise_bp == 0]["split"].replace("val", "train").value_counts()
    ax.set_xlabel("Fed surprise s (2y yield change, bp)")
    ax.set_ylabel("XLE 1-day excess return (%)")
    ax.set_title(f"What GP 'learned' for energy: ~0 everywhere, -1% when s = 0\n"
                 f"fit on {n0.get('train', 0)} train events with s = 0, scored on {n0.get('test', 0)} test ones; test R² = −0.41",
                 loc="left", fontsize=13, color=INK)
    ax.legend(fontsize=9.5, loc="lower left")
    save(fig, "fig4_xle_formula.png")


def fig5_zero_artifact():
    p = load_panel(with_features=False)
    p["period"] = np.where(p["split"].isin(["train", "val"]), "train", np.where(p["split"] == "test", "test", None))
    m = p.dropna(subset=["period"]).groupby(["ticker", "period"])["excess_1d"].mean().unstack() * 1e4
    # Dumbbell: each sector's 1999-2009 mean (hollow) moves to its 2010-2015 mean (filled).
    # The point of the slide is that almost every arrow heads toward 0.
    names = {"XLB": "Materials", "XLE": "Energy", "XLF": "Financials", "XLI": "Industrials",
             "XLK": "Tech", "XLP": "Staples", "XLU": "Utilities", "XLV": "Health care",
             "XLY": "Cons. discretionary"}
    m = m.loc[config.TICKERS].sort_values("train")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(m))
    ax.axvline(0, color=INK, linewidth=1.5)
    for yi, (tr, te) in zip(y, m[["train", "test"]].to_numpy()):
        ax.annotate("", xy=(te, yi), xytext=(tr, yi),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=1.4, shrinkA=6, shrinkB=6))
    ax.scatter(m["train"], y, s=90, facecolor="white", edgecolor="#2a78d6", linewidth=2,
               zorder=3, label="1999–2009 average (what R² compares against)")
    ax.scatter(m["test"], y, s=90, color="#eb6834", zorder=3, label="2010–2015 average (what happened)")
    ax.set_yticks(y, [f"{names[t]} ({t})" for t in m.index])
    ax.set_xlabel("average 1-day excess return on Fed days (bp)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_xlim(-55, 40)
    save(fig, "fig5_zero_artifact.png")


def fig6_features():
    u = pd.read_csv(config.RESULTS / "gp_feature_usage.csv")
    u = u[(u.version == "v1") & (u.scope == "per_ticker")]
    cols = ["surprise_bp", "vix", "slope_2s10s", "beta_1y", "mom_20d", "constant"]
    agg = {h: [np.average(u[u.horizon == h][c], weights=u[u.horizon == h]["n_seeds"]) for c in cols]
           for h in config.HORIZONS}
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(cols))
    for i, (h, color) in enumerate(zip(config.HORIZONS, ("#2a78d6", "#eb6834"))):
        ax.barh(y + (i - 0.5) * 0.38, np.array(agg[h]) * 100, 0.36, color=color, label=f"{h} horizon",
                edgecolor="white", linewidth=1)
    ax.set_yticks(y, [c if c != "constant" else "no input (constant)" for c in cols])
    ax.invert_yaxis()
    ax.set_xlabel("share of GP2 per-sector seed-runs using the input (%)")
    ax.set_title("GP2 uses the features, mostly at 5 days, but they don't help on test",
                 loc="left", fontsize=12.5, color=INK)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=9.5, loc="center right")
    save(fig, "fig6_feature_usage.png")


def main():
    summary = pd.read_csv(config.RESULTS / "summary.csv")
    fig1_pooled(summary)
    fig2_heatmap(summary)
    fig3_overfit()
    fig4_xle()
    fig5_zero_artifact()
    fig6_features()


if __name__ == "__main__":
    main()
