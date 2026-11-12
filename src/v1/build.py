"""Build the Stage 2 panel: events (official surprises), Ken French industry returns, features.

Outputs (data/processed/v1/<universe>/ for ff12 and ff49, not in git) keep Stage 1's names:
    events.csv    one row per FOMC monetary event 1994-02-04 onward
    returns.csv   one row per (event, industry): ret_1d, ret_5d, spy_1d, spy_5d, excess_1d, excess_5d
                  ("spy_*" is the benchmark column name from Stage 1; here it is the CRSP market)
    features.csv  one row per (event, industry), everything known before day 0:
                  macro state at day -1; beta_mkt, beta_2y, mom_20d, vol_60d at day -1;
                  bm (log book-to-market, Fama-French timing), size (log average firm size, last
                  month end before day 0); fomc_beta_raw / fomc_beta_exc (the industry's slope on
                  mps_bp over EARLIER events only, % per 10 bp, shrunk toward 0 by n / (n + 20))
    panel.csv     events x returns x features
and docs/DATA_CARD.md.

The surprise (events.csv):
    mps_bp       main shock. First principal component of the changes in ED1-ED4 (Eurodollar,
                 SOFR from 2022) over the USMPD "Monetary Events" window (statement + press
                 conference), scaled so a 1 bp surprise moves ED4 by 1 bp. This is the Bauer &
                 Swanson (2023) construction, extended to 2026 with the SF Fed's USMPD. The
                 loadings and scale are estimated on the dev years only (<= 2015), so nothing
                 from the locked test period shapes the shock definition. Positive = hawkish.
    mps_stmt_bp  the same, on the 30-minute statement window (closest to Bauer-Swanson's data)
    mps_bs_bp, mps_orth_bs_bp   the official Bauer-Swanson series (ends 2023-12), robustness
    target_bp    MP1: surprise in the current-meeting rate (fed funds futures)
    path_bp      ED4 change not explained by MP1 (forward guidance), dev-period regression
    surprise_bp  Stage 1's proxy (2y Treasury yield change, day -1 to day 0), robustness
    zlb          fed funds <= 0.25% on day -1

Day 0 = first trading day on/after the announcement, or the next trading day if the release
was at/after 4 pm ET or on a weekend (e.g. 2020-03-15, Sunday 5 pm -> Monday 2020-03-16).

Run from the repo root (after python -m src.v1.fetch):
    python -m src.v1.build
"""

import numpy as np
import pandas as pd

from src import build_features as bf
from src.v1 import fetch
from src.v1 import settings as S

ED = ["ED1", "ED2", "ED3", "ED4"]
MACRO_V1 = {
    "vix": lambda f: f["VIXCLS"],
    "slope_2s10s": lambda f: f["DGS10"] - f["DGS2"],
    "term_10y3m": lambda f: f["DGS10"] - f["DGS3MO"],
    "credit": lambda f: f["BAA10Y"],
    "ff": lambda f: f["DFF"],
}
FEATURES = ["vix", "slope_2s10s", "term_10y3m", "credit", "ff",
            "beta_mkt", "beta_2y", "mom_20d", "vol_60d", "mkt_mom_20d",
            "bm", "size", "fomc_beta_raw", "fomc_beta_exc"]


# ---------------- surprises ----------------

def fit_mps(ed: pd.DataFrame, dev: pd.Series) -> tuple[np.ndarray, float]:
    """PC1 loadings of the ED changes (dev rows only) and the scale that gives ED4 a unit slope."""
    x = ed[dev].to_numpy(float)
    xc = x - x.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(xc, rowvar=False))
    w = vecs[:, -1]
    if w.sum() < 0:
        w = -w                                   # positive = rates up = hawkish
    pc = xc @ w
    slope = np.polyfit(pc, x[:, -1], 1)[0]       # ED4 on pc
    return w, slope                              # mps = slope * pc  ->  ED4 on mps has slope 1


def apply_mps(ed: pd.DataFrame, w: np.ndarray, scale: float, center: np.ndarray) -> np.ndarray:
    return scale * ((ed.to_numpy(float) - center) @ w)


def day0(date_time: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    d = date_time.normalize()
    i = trading_days.searchsorted(d)
    if i < len(trading_days) and trading_days[i] == d and date_time.hour >= S.MARKET_CLOSE_HOUR:
        i += 1
    return trading_days[i] if i < len(trading_days) else None


def build_events(usmpd: pd.DataFrame, stmt: pd.DataFrame, bs: pd.DataFrame, fred: pd.DataFrame,
                 trading_days: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    u = usmpd[usmpd["Date"] >= S.SAMPLE_START].reset_index(drop=True)
    st = stmt.set_index("Date").reindex(u["Date"])
    dev = (u["Date"] <= S.DEV_END).to_numpy()
    info = {}

    w, k = fit_mps(u[ED], pd.Series(dev))
    center = u.loc[dev, ED].to_numpy(float).mean(axis=0)
    w_s, k_s = fit_mps(st[ED].reset_index(drop=True), pd.Series(dev))
    center_s = st[ED].to_numpy(float)[dev].mean(axis=0)
    info["loadings_monetary_events"] = dict(zip(ED, np.round(w, 3)))
    info["loadings_statement"] = dict(zip(ED, np.round(w_s, 3)))

    ev = pd.DataFrame({
        "event_date": u["Date"].dt.normalize(),
        "event_time": u["date_time"].dt.strftime("%H:%M"),
        "event_type": np.where(u["Unscheduled"] == 1, "fomc_unscheduled", "fomc_scheduled"),
        "press_conf": u["PC"].astype(int),
        "mps_bp": 100 * apply_mps(u[ED], w, k, center),
        "mps_stmt_bp": 100 * apply_mps(st[ED], w_s, k_s, center_s),
        "target_bp": 100 * u["MP1"].to_numpy(float),
        "hf_sp500_pct": u["SP500"].to_numpy(float),       # high-frequency S&P 500 return in the window (%)
    })
    # path = ED4 not explained by MP1 (dev coefficients)
    b = np.polyfit(u.loc[dev, "MP1"], u.loc[dev, "ED4"], 1)
    ev["path_bp"] = 100 * (u["ED4"] - np.polyval(b, u["MP1"])).to_numpy(float)

    bsi = bs.set_index("Date")
    ev["mps_bs_bp"] = 100 * ev["event_date"].map(bsi["MPS"])
    ev["mps_orth_bs_bp"] = 100 * ev["event_date"].map(bsi["MPS_ORTH"])

    ev["trade_date"] = [day0(t, trading_days) for t in u["date_time"]]
    ev = ev.dropna(subset=["trade_date"])

    # Stage 1 proxy and ZLB flag, from FRED
    dgs2 = fred["DGS2"].dropna()
    dff = fred["DFF"].dropna()
    before, after, zlb = [], [], []
    for t in ev["trade_date"]:
        j = bf.day_minus_1(trading_days, t)
        d_1 = trading_days[j]
        b_ = dgs2[dgs2.index <= d_1]
        a_ = dgs2[dgs2.index <= t]
        before.append(b_.iloc[-1] if len(b_) else np.nan)
        after.append(a_.iloc[-1] if len(a_) and a_.index[-1] > d_1 else np.nan)
        z = dff[dff.index <= d_1]
        zlb.append(bool(len(z) and z.iloc[-1] <= 0.25))
    ev["surprise_bp"] = (np.array(after) - np.array(before)) * 100
    ev["zlb"] = zlb

    ok = ev["mps_bs_bp"].notna() & (ev["event_date"] <= S.DEV_END)
    info["corr_mps_vs_official"] = round(float(ev.loc[ok, "mps_bp"].corr(ev.loc[ok, "mps_bs_bp"])), 3)
    info["corr_mps_stmt_vs_official"] = round(float(ev.loc[ok, "mps_stmt_bp"].corr(ev.loc[ok, "mps_bs_bp"])), 3)
    info["corr_mps_vs_2y_proxy"] = round(float(ev.loc[ok, "mps_bp"].corr(ev.loc[ok, "surprise_bp"])), 3)
    cols = ["event_date", "trade_date", "event_time", "event_type", "press_conf", "mps_bp", "mps_stmt_bp",
            "mps_bs_bp", "mps_orth_bs_bp", "target_bp", "path_bp", "surprise_bp", "hf_sp500_pct", "zlb"]
    return ev[cols].reset_index(drop=True), info


# ---------------- returns ----------------

def build_returns(rets: pd.DataFrame, events: pd.DataFrame, industries: list[str],
                  source: str) -> pd.DataFrame:
    days = rets.index
    growth = (1 + rets)
    rows = []
    for _, e in events.iterrows():
        i = days.get_loc(e["trade_date"])
        r1 = rets.iloc[i]
        r5 = growth.iloc[i:i + 5].prod() - 1 if i + 5 <= len(days) else pd.Series(np.nan, index=rets.columns)
        for ind in industries:
            rows.append({"event_date": e["event_date"], "trade_date": e["trade_date"], "ticker": ind,
                         "ret_1d": r1[ind], "ret_5d": r5[ind],
                         "spy_1d": r1[S.BENCHMARK], "spy_5d": r5[S.BENCHMARK],
                         "excess_1d": r1[ind] - r1[S.BENCHMARK], "excess_5d": r5[ind] - r5[S.BENCHMARK],
                         "source": source})
    return pd.DataFrame(rows)


# ---------------- features ----------------

def fomc_betas(returns: pd.DataFrame, events: pd.DataFrame, col: str) -> pd.Series:
    """Each industry's slope of `col` (in %) on mps_bp over events strictly before this one (by
    trade date), per 10 bp, shrunk toward 0 by n / (n + FOMC_BETA_SHRINK); 0 with fewer than
    FOMC_BETA_MIN earlier events. Indexed by (event_date, ticker)."""
    r = returns.merge(events[["event_date", "mps_bp"]], on="event_date").sort_values("trade_date")
    out = pd.Series(0.0, index=r.index)
    for _, g in r.groupby("ticker"):
        x, y, t = g["mps_bp"].to_numpy(), g[col].to_numpy() * 100, g["trade_date"].to_numpy()
        for k, i in enumerate(g.index):
            prior = t < t[k]
            n = int(prior.sum())
            if n >= S.FOMC_BETA_MIN:
                xp, yp = x[prior], y[prior]
                b = np.cov(xp, yp, ddof=1)[0, 1] / np.var(xp, ddof=1)
                out[i] = 10 * b * n / (n + S.FOMC_BETA_SHRINK)
    return pd.Series(out.to_numpy(), index=pd.MultiIndex.from_frame(r[["event_date", "ticker"]]))


def characteristics_asof(bm: pd.DataFrame, size: pd.DataFrame, trade_date: pd.Timestamp,
                         ind: str) -> tuple[float, float]:
    """log BE/ME for the Fama-French year in force (row t is used from July of year t) and log
    average firm size at the last month end strictly before the trade date's month."""
    yr = trade_date.year if trade_date.month >= 7 else trade_date.year - 1
    b = bm[ind][bm.index.year <= yr].dropna()
    z = size[ind][size.index < trade_date.normalize().replace(day=1)].dropna()
    return (float(np.log(b.iloc[-1])) if len(b) and b.iloc[-1] > 0 else np.nan,
            float(np.log(z.iloc[-1])) if len(z) else np.nan)


def build_features(rets: pd.DataFrame, fred: pd.DataFrame, events: pd.DataFrame, industries: list[str],
                   bm: pd.DataFrame, size: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    days = rets.index
    prices = (1 + rets.fillna(0)).cumprod()
    betas = bf.rolling_beta(rets, benchmark=S.BENCHMARK)
    d2 = fred["DGS2"].reindex(days)
    d2chg = d2 - d2.shift(1)                     # NaN when either day is missing (bond holidays)
    beta2y = bf.rolling_beta(rets[industries].assign(DGS2CHG=d2chg), benchmark="DGS2CHG")
    vols = rets.rolling(bf.VOL_WINDOW, min_periods=bf.VOL_MIN).std()
    moms = prices / prices.shift(bf.MOM_WINDOW) - 1

    rows = []
    for _, e in events.iterrows():
        j = bf.day_minus_1(days, e["trade_date"])
        asof = days[j]
        macro, mdate = bf.macro_asof(fred, asof, MACRO_V1)
        for ind in industries:
            b_, z_ = characteristics_asof(bm, size, e["trade_date"], ind)
            rows.append({"event_date": e["event_date"], "ticker": ind, "asof_date": asof,
                         "macro_asof_date": mdate, **macro,
                         "beta_mkt": betas[ind].iloc[j], "beta_2y": beta2y[ind].iloc[j],
                         "mom_20d": moms[ind].iloc[j], "vol_60d": vols[ind].iloc[j],
                         "mkt_mom_20d": moms[S.BENCHMARK].iloc[j], "bm": b_, "size": z_})
    f = pd.DataFrame(rows)
    key = pd.MultiIndex.from_frame(f[["event_date", "ticker"]])
    for col, name in [("ret_1d", "fomc_beta_raw"), ("excess_1d", "fomc_beta_exc")]:
        f[name] = fomc_betas(returns, events, col).reindex(key).to_numpy()
    return f


# ---------------- data card ----------------

def data_card(panels: dict, events: pd.DataFrame, info: dict, manifest: dict) -> str:
    panel = panels[S.UNIVERSE]
    ev = events.assign(year=events["event_date"].dt.year,
                       period=np.where(events["event_date"] <= S.DEV_END, "dev", "LOCKED test"))
    cov = manifest.get("_coverage", {})
    out = ["# DATA_CARD: Stage 2 (v1) panel", "",
           "Generated by `python -m src.v1.build`. Do not edit by hand.", "",
           "## Sources", "",
           "| file | url | bytes | sha256 (first 16) | downloaded (UTC) |", "|---|---|---|---|---|"]
    for name, m in manifest.items():
        if name.startswith("_"):
            continue
        out.append(f"| {name} | {m['url']} | {m['bytes']:,} | {m['sha256'][:16]} | {m['downloaded_utc']} |")
    out += ["", "Coverage (first, last, rows): " + "; ".join(f"{k} {v[0]} to {v[1]} ({v[2]})" for k, v in cov.items()), "",
            "## Surprise construction", "",
            "Main shock `mps_bp`: first principal component of ED1-ED4 changes in the USMPD Monetary "
            "Events window, scaled to a unit effect on ED4 (Bauer & Swanson 2023 construction), "
            "estimated on dev years only. Positive = hawkish.", "",
            f"- loadings (Monetary Events window): {info['loadings_monetary_events']}",
            f"- loadings (statement window): {info['loadings_statement']}",
            f"- correlation with the official Bauer-Swanson MPS, dev years: **{info['corr_mps_vs_official']}** "
            f"(statement window: {info['corr_mps_stmt_vs_official']})",
            f"- correlation with Stage 1's 2y-yield proxy, dev years: {info['corr_mps_vs_2y_proxy']}", "",
            "Why `mps_bp` and not `mps_orth` as the main shock (deviation from ARCHITECTURE.md, decided "
            "before any test-period result was seen): the official MPS_ORTH ends in December 2023 and "
            "cannot be extended without the authors' macro predictors, which would cut the locked test "
            "period short and drop the 2024-2026 cutting cycle. `mps_orth_bs_bp` is kept as a robustness "
            "row on the years it covers.", "",
            "## Events", "",
            md(ev.pivot_table(index="period", columns="event_type", values="mps_bp", aggfunc="size",
                              fill_value=0).reset_index()), "",
            md(ev.groupby("period")["mps_bp"].describe()[["count", "mean", "std", "min", "max"]].round(2).reset_index()), "",
            "Events per year (scheduled / unscheduled):", "",
            md(ev.pivot_table(index="year", columns="event_type", values="mps_bp", aggfunc="size",
                              fill_value=0).reset_index()), "",
            "## Panels", "",
            *[f"- `{u}`: {len(p):,} rows = {p['event_date'].nunique()} events x {p['ticker'].nunique()} industries"
              for u, p in panels.items()],
            "- targets: `ret_1d` = industry return, close day -1 to close day 0; `excess_1d` = the same "
            "minus the CRSP value-weighted market",
            f"- tables below: main universe `{S.UNIVERSE}`",
            "", "Missing values per column:", "",
            md(panel.isna().sum().rename("missing").reset_index().rename(columns={"index": "column"})
               .query("missing > 0") if panel.isna().sum().any()
               else pd.DataFrame({"column": ["(none)"], "missing": [0]})), "",
            "Feature summary (dev years):", "",
            md(panel[panel["event_date"] <= S.DEV_END][[S.SHOCK, "ret_1d", "excess_1d"] + FEATURES].describe().T[
                ["count", "mean", "std", "min", "max"]].round(4).reset_index().rename(columns={"index": "column"})),
            ""]
    return "\n".join(out)


def md(df: pd.DataFrame) -> str:
    def c(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:g}"
        return str(v)
    return ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" +
            "\n".join("| " + " | ".join(c(v) for v in r) + " |" for r in df.itertuples(index=False)))


def main() -> None:
    import json
    fred = fetch.load_fred()
    panels = {}
    for uni in S.UNIVERSES:
        out = S.processed(uni)
        out.mkdir(parents=True, exist_ok=True)
        rets = fetch.load_industry_returns(universe=uni)
        rets = rets[rets.index >= "1990-01-01"]
        dev = rets.loc[S.SAMPLE_START:S.DEV_END]
        industries = [c for c in rets.columns if c != S.BENCHMARK
                      and dev[c].notna().mean() >= S.MIN_INDUSTRY_COVERAGE]
        bm, size = fetch.load_characteristics(universe=uni)
        events, info = build_events(fetch.load_usmpd(), fetch.load_usmpd(sheet="Statements"),
                                    fetch.load_bauer_swanson(), fred, rets.index)
        events = events[events["trade_date"] <= rets.index.max()]
        returns = build_returns(rets, events, industries, f"french_{uni}")
        features = build_features(rets, fred, events, industries, bm, size, returns)
        panel = (returns.merge(events, on=["event_date", "trade_date"], validate="many_to_one")
                 .merge(features, on=["event_date", "ticker"], validate="one_to_one"))
        for name, df in [("events", events), ("returns", returns), ("features", features), ("panel", panel)]:
            df.to_csv(out / f"{name}.csv", index=False, date_format="%Y-%m-%d")
            print(f"{uni}/{name}: {df.shape}")
        panels[uni] = panel
    manifest = json.loads((S.RAW / "manifest.json").read_text(encoding="utf-8"))
    (S.ROOT / "docs" / "DATA_CARD.md").write_text(data_card(panels, events, info, manifest), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
