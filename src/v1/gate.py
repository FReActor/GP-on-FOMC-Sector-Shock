"""Gate 2: before any model, check the Stage 2 data shows the known Bernanke-Kuttner pattern.

On the DEV years only (1994-2015, scheduled meetings), regress returns on the surprise:
  (a) high-frequency S&P 500 return in the USMPD window on mps   -> expect clearly negative
  (b) daily market return (close day -1 to close day 0) on mps    -> expect negative
  (c) each industry's raw daily return on mps                     -> negative for most; cyclical
      and long-duration industries (BusEq, Durbl, Manuf) more than defensive ones (Utils, NoDur)
  (d) the same with Stage 1's 2y-yield proxy, on the same events  -> how much cleaner is the
      official surprise?
  (e) each industry's EXCESS return on mps: the actual prediction target. If even the in-sample
      t-stats are small here, out-of-sample prediction will be hard for any model.

PASS if (a) and (b) are negative with t < -2.

Run from the repo root:
    python -m src.v1.gate
"""

import numpy as np
import pandas as pd

from src.v1 import settings as S
from src.v1.splits import load_panel


def ols(x, y) -> tuple[float, float, float, int]:
    """slope (per bp), t-stat (HC1 robust), R^2, n."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    cov = xtx_inv @ (X.T * e**2) @ X @ xtx_inv * n / (n - k)
    r2 = 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))
    return float(b[1]), float(b[1] / np.sqrt(cov[1, 1])), float(r2), int(n)


def md(df, fmt="{:+.3f}"):
    def c(v):
        if isinstance(v, (float, np.floating)):
            return fmt.format(v)
        return str(v)
    return ("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" +
            "\n".join("| " + " | ".join(c(v) for v in r) + " |" for r in df.itertuples(index=False)))


def main() -> None:
    p = load_panel()
    ev = p.drop_duplicates("event_date")
    rows = []
    a = ols(ev[S.SHOCK], ev["hf_sp500_pct"])
    rows.append(("(a) S&P 500, high-frequency window (%)", a))
    b = ols(ev[S.SHOCK], ev["spy_1d"] * 100)
    rows.append(("(b) market, daily close-to-close (%)", b))
    b2 = ols(ev["surprise_bp"], ev["spy_1d"] * 100)
    rows.append(("(d) market daily (%), Stage 1 2y proxy", b2))
    head = pd.DataFrame([{"regression": n, "slope per 10bp": 10 * r[0], "t": r[1], "R2": r[2], "n": str(r[3])}
                         for n, r in rows])

    ind = []
    for t, g in p.groupby("ticker"):
        raw = ols(g[S.SHOCK], g["ret_1d"] * 100)
        exc = ols(g[S.SHOCK], g["excess_1d"] * 100)
        prox = ols(g["surprise_bp"], g["ret_1d"] * 100)
        ind.append({"industry": t, "raw slope/10bp": 10 * raw[0], "raw t": raw[1],
                    "excess slope/10bp": 10 * exc[0], "excess t": exc[1], "excess R2": exc[2],
                    "proxy raw t": prox[1]})
    ind = pd.DataFrame(ind).sort_values("raw slope/10bp")
    passed = a[0] < 0 and a[1] < -2 and b[0] < 0 and b[1] < -2
    n_neg = int((ind["raw slope/10bp"] < 0).sum())
    n_sig_exc = int((ind["excess t"].abs() > 2).sum())

    out = ["# Gate 2: does the Stage 2 data show the known pattern?", "",
           f"Dev years only (1994-2015), scheduled meetings, {len(ev)} events. Shock `{S.SHOCK}` "
           "(positive = hawkish). Returns in percent; slopes per +10 bp surprise; HC1 robust t-stats.", "",
           md(head), "",
           f"**Gate 2: {'PASS' if passed else 'FAIL'}** (rule: (a) and (b) negative with t < -2).", "",
           "## By industry (daily, close day -1 to close day 0)", "",
           f"- raw return slope negative for {n_neg} of {len(ind)} industries",
           f"- excess return has |t| > 2 for {n_sig_exc} of {len(ind)} industries", "",
           md(ind), "",
           "## How to read this", "",
           "- (a) vs (b): the market reacts strongly inside the 30-100 minute window; over the full day "
           "other news is added, so the daily slope is noisier. The models must work with daily data.",
           "- `proxy raw t` uses Stage 1's 2y-yield surprise on the same events, for comparison.",
           "- The excess-return columns are the honest preview: they show how much of each industry's "
           "reaction is *different from the market's*, which is what the models try to predict.", ""]
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    (S.RESULTS / "gate2.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
