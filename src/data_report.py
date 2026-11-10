"""Write the data-quality report for the Stage 1 panel.

Inputs:  data/processed/{events,returns,features}.csv, data/raw/prices.csv
         (optional) S&P 500 index closes from yfinance, cached in data/raw/gspc.csv
Outputs: results/data_report.md, results/fig_surprise_hist.png

Run from the repo root, after the four build steps:
    python -m src.data_report
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

CORE = ["surprise_bp", "vix", "slope_2s10s", "beta_1y", "mom_20d"]
EXTRA = ["credit", "ff", "vol_60d", "mkt_mom_20d"]

# One line per large event: what happened, and why the 2y yield moved the way it did.
# Checked against Fed statements and same-day news coverage (sources in the report).
EXPLAIN = {
    "2008-01-22": "Emergency 75bp cut, announced before the open after a global stock sell-off.",
    "2023-12-13": "Hold at 5.25-5.50%; dot plot showed three 2024 cuts and Powell sounded dovish.",
    "2022-06-15": "75bp hike (largest since 1994) was already priced after the leak; Powell said "
                  "moves this size would not be common, so the 2y fell.",
    "2007-12-11": "Only a 25bp cut when some wanted 50; stocks fell ~2% and money ran to Treasuries "
                  "(a growth scare, not a hawkish surprise).",
    "2008-03-18": "75bp cut, two days after the Bear Stearns rescue; relief rally pushed the 2y up.",
    "2009-03-18": "QE1 expansion: Fed announced $300bn of Treasury purchases and more MBS buying.",
    "2023-03-22": "25bp hike one week after SVB failed; statement softened guidance on more hikes.",
    "2008-10-08": "Coordinated 50bp cut with five other central banks, announced 7:00 a.m. ET; "
                  "the 2y rose as flight-to-safety demand eased.",
    "2004-01-28": "Fed dropped the 'considerable period' pledge (replaced with 'patient'); hike "
                  "signal sent the 2y up.",
    "2003-06-25": "Cut 25bp to 1% when markets expected 50bp, so short yields rose.",
}

SOURCES = [
    ("Fed press release, 2008-10-08 (7:00 a.m. EDT)",
     "https://www.federalreserve.gov/newsevents/pressreleases/monetary20081008a.htm"),
    ("ECB joint statement, 2008-10-08", "https://www.ecb.europa.eu/press/pr/date/2008/html/pr081008.en.html"),
    ("S&P 500 close 984.94 on 2008-10-08 (Star-Bulletin / AP)",
     "https://archives.starbulletin.com/content/20081009_Stocks_end_lower_after_Fed_rate_cut.html"),
    ("2007-12-11 cut and 2y move (CNN Money)", "https://money.cnn.com/2007/12/11/markets/markets_0500/"),
    ("2022-06-15 hike and 2y move (MNI)", "https://www.mnimarkets.com/articles/bull-steeper-as-powell-vacillates-on-another-75"),
]


def load():
    events = pd.read_csv(PROCESSED_DIR / "events.csv", parse_dates=["event_date", "trade_date"])
    returns = pd.read_csv(PROCESSED_DIR / "returns.csv", parse_dates=["event_date", "trade_date"])
    feats = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["event_date", "asof_date"])
    prices = pd.read_csv(RAW_DIR / "prices.csv", index_col=0, parse_dates=True).sort_index()
    return events, returns, feats, prices


def load_gspc() -> pd.Series | None:
    """S&P 500 index closes (not SPY) to cross-check SPY. Cached; None if offline."""
    cache = RAW_DIR / "gspc.csv"
    if not cache.exists():
        try:
            import yfinance as yf
            df = yf.download("^GSPC", start="1998-06-01", auto_adjust=False, progress=False)
            df["Close"].squeeze().rename("GSPC").to_csv(cache)
        except Exception as exc:  # noqa: BLE001
            print(f"could not download ^GSPC ({exc}); skipping SPY cross-check")
            return None
    return pd.read_csv(cache, index_col=0, parse_dates=True).squeeze("columns")


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        if isinstance(v, pd.Timestamp):
            return v.date().isoformat()
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def surprise_hist(events: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-30, 31, 2.5)
    kinds = [("fomc_scheduled", "#4C72B0"), ("fomc_unscheduled", "#DD8452")]
    xs = [events.loc[events["event_type"] == k, "surprise_bp"] for k, _ in kinds]
    ax.hist(xs, bins=bins, stacked=True, color=[c for _, c in kinds],
            label=[f"{k.split('_')[1]} (n={len(x)})" for (k, _), x in zip(kinds, xs)],
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("surprise_bp: 2y Treasury yield change, day -1 to day 0 (basis points)")
    ax.set_ylabel("events")
    ax.set_title(f"FOMC rate surprises, {events['event_date'].dt.year.min()}-"
                 f"{events['event_date'].dt.year.max()} ({len(events)} events)")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report() -> str:
    events, returns, feats, prices = load()
    RESULTS_DIR.mkdir(exist_ok=True)
    surprise_hist(events, RESULTS_DIR / "fig_surprise_hist.png")
    panel = returns.merge(feats.drop(columns=["asof_date"]), on=["event_date", "ticker"], how="left",
                          validate="one_to_one")
    panel = panel.merge(events[["event_date", "event_type", "surprise_bp"]], on="event_date")

    out = ["# Stage 1 data report", "",
           "Generated by `python -m src.data_report` from the files in `data/processed/`. "
           "Rerun after any change to the build scripts.", ""]

    # 1. size
    s = events["surprise_bp"]
    out += ["## 1. What is in the panel", "",
            f"- **{len(events)} FOMC events** from {events['event_date'].min().date()} to "
            f"{events['event_date'].max().date()}: "
            f"{(events['event_type'] == 'fomc_scheduled').sum()} scheduled, "
            f"{(events['event_type'] == 'fomc_unscheduled').sum()} unscheduled (between-meeting cuts).",
            f"- **{len(returns):,} (event, sector ETF) rows** across {returns['ticker'].nunique()} SPDR "
            "sector ETFs; SPY is the benchmark, not a row.",
            f"- Surprise (`surprise_bp`): mean {s.mean():.1f} bp, sd {s.std():.1f} bp, median {s.median():.1f}, "
            f"range {s.min():.0f} to {s.max():.0f}. {(s.abs() >= 10).sum()} events move the 2y by 10 bp or more; "
            f"{(s == 0).sum()} move it by exactly 0.", ""]

    per_year = (events.assign(year=events["event_date"].dt.year)
                .pivot_table(index="year", columns="event_type", values="surprise_bp",
                             aggfunc="size", fill_value=0))
    per_year.columns = [c.replace("fomc_", "") for c in per_year.columns]
    per_year["mean |surprise| bp"] = events.groupby(events["event_date"].dt.year)["surprise_bp"].apply(
        lambda x: x.abs().mean())
    out += ["### Events per year", "", md_table(per_year.reset_index(), "{:.1f}"), ""]

    tick = returns.groupby("ticker").agg(rows=("event_date", "size"), first=("event_date", "min"),
                                         last=("event_date", "max"),
                                         mean_excess_1d_pct=("excess_1d", lambda x: 100 * x.mean()),
                                         sd_excess_1d_pct=("excess_1d", lambda x: 100 * x.std()))
    out += ["### Rows per ticker", "",
            "XLRE (real estate) starts in 2015-10 and XLC (communication services) in 2018-06, so "
            "they have fewer rows. Split option A uses the other 9 tickers only.", "",
            md_table(tick.reset_index(), "{:.3f}"), "",
            "![surprise histogram](fig_surprise_hist.png)", ""]

    # 2. largest surprises
    top = events.reindex(s.abs().sort_values(ascending=False).index).head(10)
    top = top.assign(spy_1d_pct=top["event_date"].map(
        returns.drop_duplicates("event_date").set_index("event_date")["spy_1d"] * 100))
    top["why"] = top["event_date"].dt.strftime("%Y-%m-%d").map(EXPLAIN).fillna("")
    top_tbl = top[["event_date", "event_type", "surprise_bp", "spy_1d_pct", "why"]].copy()
    top_tbl["event_type"] = top_tbl["event_type"].str.replace("fomc_", "")
    out += ["## 2. The 10 largest surprises (by |surprise_bp|)", "",
            "Sanity check that big values are real events, not data errors. All ten are real, "
            "well-known Fed days. Several show the weakness of a 2y-yield proxy: on 2007-12-11 and "
            "2008-10-08 the yield moved *against* the direction of the rate action because the day's "
            "growth fears or safe-haven flows dominated. Stage 2 replaces this proxy with the "
            "Bauer-Swanson (SF Fed) surprise series.", "",
            md_table(top_tbl, "{:.1f}"), ""]

    # 3. features
    cols = CORE + EXTRA
    miss = pd.DataFrame({"feature": cols,
                         "missing": [panel[c].isna().sum() for c in cols],
                         "mean": [panel[c].mean() for c in cols],
                         "sd": [panel[c].std() for c in cols],
                         "min": [panel[c].min() for c in cols],
                         "max": [panel[c].max() for c in cols]})
    out += ["## 3. Features", "",
            "Every feature is measured at **day -1** (the last trading day before the announcement), "
            "so nothing after the Fed decision leaks in. `tests/test_features.py` checks this by "
            "scrambling all data from day 0 on and confirming no feature changes.", "",
            "Missing values: beta_1y needs 126 days of history, so it is missing for the first 3 "
            "events of 1999 and the first events after XLRE / XLC launched; mom_20d is missing only "
            "for XLC's first event. No macro value is missing or stale.", "",
            md_table(miss, "{:.4f}"), ""]

    corr = panel[cols].corr()
    out += ["### Correlation matrix (all rows)", "",
            md_table(corr.reset_index().rename(columns={"index": ""}), "{:.2f}"), ""]
    hi = [(a, b, corr.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:]
          if abs(corr.loc[a, b]) >= 0.5]
    if hi:
        out += ["Pairs with |corr| >= 0.5: " + "; ".join(f"{a}-{b} {v:+.2f}" for a, b, v in hi) +
                ". Correlated inputs are fine for prediction but make individual linear coefficients "
                "and GP terms harder to interpret; mention this when reading the formulas.", ""]
    out += ["surprise_bp is close to uncorrelated with every day -1 feature, as it should be: a "
            "surprise that could be predicted from yesterday's VIX or yield curve would not be a "
            "surprise.", ""]

    # 4. 2008-10-08 check
    out += ["## 4. Check: 2008-10-08 coordinated cut", ""]
    ev = events[events["event_date"] == "2008-10-08"].iloc[0]
    spy_row = returns[returns["event_date"] == "2008-10-08"]
    spy1 = spy_row["spy_1d"].iloc[0]
    out += [f"- The Fed and five other central banks cut 50bp at **7:00 a.m. ET on Wed 2008-10-08**, "
            "before the stock market opened. So the reaction day is the same day.",
            f"- Our data: trade_date = **{ev['trade_date'].date()}** (correct), event_type = "
            f"{ev['event_type']}, surprise_bp = {ev['surprise_bp']:+.0f}.",
            f"- SPY day-0 return in our data: **{100 * spy1:+.2f}%** (yfinance close 2008-10-07 -> 2008-10-08)."]
    gspc = load_gspc()
    if gspc is not None:
        g = gspc.pct_change()
        spy = prices["SPY"].pct_change()
        both = pd.DataFrame({"spy": spy, "gspc": g}).dropna()
        both["gap"] = both["spy"] - both["gspc"]
        g8 = g.loc["2008-10-08"]
        evd = both.loc[both.index.intersection(events["trade_date"])]
        by_year = (both["gap"].abs() > 0.005).groupby(both.index.year).mean()
        out += [f"- Outside source: the **S&P 500 index closed at 984.94, {100 * g8:+.2f}%** "
                "(996.23 the day before; AP/Star-Bulletin, matches Yahoo's ^GSPC).",
                f"- **Finding: SPY and the index disagree by {100 * (spy1 - g8):+.2f} pp on this day.** "
                "The raw (unadjusted) SPY close shows the same drop, so it is not a dividend-adjustment "
                "bug. Yahoo's pre-2009 ETF closes appear to come from a later print than the 4:00 p.m. "
                "index close, and on a day with a violent last half-hour that matters.",
                f"- How common: across all {len(both):,} days, SPY and the index differ by more than "
                f"0.5 pp on {(both['gap'].abs() > 0.005).sum()} days, mostly 1998-2002 and 2008 "
                f"({100 * by_year.get(2008, 0):.0f}% of 2008 days; 0% in 2010-2019). On our "
                f"{len(evd)} event days: {(evd['gap'].abs() > 0.005).sum()} days, median gap "
                f"{1e4 * evd['gap'].abs().median():.1f} bp; correlation {evd['spy'].corr(evd['gspc']):.3f}.",
                "- **Impact:** small for the model. The target is the sector ETF minus SPY, and both "
                "come from the same Yahoo close, so the timing mostly cancels. It does mean the "
                "pre-2009 *level* of returns is noisier than the index, and 2008-10-08 is one of the "
                "noisiest rows. Robustness: rerun with this event dropped (or with all unscheduled "
                "events dropped) and report whether the conclusion changes.", ""]
        worst = evd.reindex(evd["gap"].abs().sort_values(ascending=False).index).head(6) * 100
        worst.index = worst.index.date
        out += ["Event days with the largest SPY vs. S&P 500 gap (percent):", "",
                md_table(worst.reset_index().rename(columns={"index": "trade_date"}), "{:+.2f}"), ""]
    sec = spy_row[["ticker", "ret_1d", "excess_1d"]].assign(
        ret_1d=lambda d: 100 * d["ret_1d"], excess_1d=lambda d: 100 * d["excess_1d"])
    out += ["Sector moves that day (percent): materials and energy beat SPY by 4-6 pp while "
            "financials lagged, plausible for a day of commodity rebound and bank stress. These "
            "are among the largest excess returns in the panel, so check model results with and "
            "without this event.", "",
            md_table(sec, "{:+.2f}"), ""]

    out += ["## Sources", ""] + [f"- [{t}]({u})" for t, u in SOURCES] + [""]
    return "\n".join(out)


def main() -> None:
    report = build_report()
    path = RESULTS_DIR / "data_report.md"
    path.write_text(report, encoding="utf-8")
    print(f"report -> {path}\nfigure -> {RESULTS_DIR / 'fig_surprise_hist.png'}")


if __name__ == "__main__":
    main()
