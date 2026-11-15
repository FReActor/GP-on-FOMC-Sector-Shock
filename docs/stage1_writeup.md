# Stage 1 write-up: can GP find how sectors react to Fed surprises?

Each section is about half a page. The owners fill in their own (see docs/STAGE1_PLAN.md).

## 1. Question and motivation
*(leads)*

## 2. Data & features

**Events.** We use all 222 FOMC announcements from February 1999 to December 2025: 215 scheduled
meetings and 7 between-meeting emergency cuts (2001 ×3, 2008 ×2, 2020 ×2). Dates come from the
Federal Reserve's meeting calendars. "Day 0" is the first trading day on or after the announcement;
for the two Sunday announcements (2020-03-15, for example) that is the following Monday.

**Shock.** The surprise is the change in the 2-year Treasury yield from the close of day −1 to the
close of day 0, in basis points (FRED series DGS2). The 2y yield moves when markets change their
expectation of where the Fed is heading, so a large move means the Fed did something the market had
not priced in. Surprises are mostly small: standard deviation 7.4 bp, median 0, and only 39 events move
the 2y by 10 bp or more (`results/fig_surprise_hist.png`). The measure is free and goes back to
1999, but it is a proxy: it contains the whole day's news, not only the Fed's. Two of the ten largest
values move *opposite* to the policy action (2007-12-11 and 2008-10-08), because growth fears or
safe-haven flows dominated that day. Stage 2 replaces it with the Bauer-Swanson surprise series
from the San Francisco Fed, which uses a 30-minute window around the announcement.

**Target.** For each of the 11 SPDR sector ETFs we compute the return from the close of day −1 to the
close of day 0 (`1d`), and to day +4 (`5d`), minus SPY's return over the same window. This
"excess return" asks whether a sector beats or lags the market, so it strips out the market-wide
move. The panel has 2,142 (event, sector) rows. XLRE and XLC start in 2015 and 2018, so the main
split uses the other 9 sectors.

**Features.** Each row gets four state variables measured at **day −1**, so no information from the
announcement day leaks in:

| feature | what it is | why it might matter |
|---|---|---|
| `vix` | VIX close | in fearful markets a surprise may hit harder |
| `slope_2s10s` | 10y minus 2y Treasury yield (pp) | where we are in the rate cycle |
| `beta_1y` | sector's 1-year beta to SPY | how cyclical the sector currently is |
| `mom_20d` | sector's return over the past 20 trading days | whether the sector is already trending |

Optional extras for a GP v2: `credit` (Baa minus 10y spread), `ff` (fed funds rate), `vol_60d`
(60-day daily volatility) and `mkt_mom_20d` (SPY's 20-day return). Macro values are read "as of" day −1:
if the bond market was closed that day, the most recent earlier value is used, and none is more than
a week old. Beta needs 126 days of history, so it is missing only for the first three 1999 events and
the first events after XLRE and XLC launched (34 of 2,142 rows).

**Checks.** `tests/test_features.py` and `tests/test_returns.py` (24 tests, run on every pull request)
recompute features by hand, check the excess-return identity and the ETF start dates, and include a
look-ahead test: scrambling every price and yield from day 0 onward must leave the features unchanged.
The surprise is close to uncorrelated with all day −1 features (|r| ≤ 0.11), as a true surprise
should be. Some features overlap (VIX with credit spread, r = 0.68; slope with fed funds, r = −0.71),
which matters when interpreting coefficients but not for prediction.

**Known data issue.** On the 2008-10-08 coordinated cut (announced 7 a.m., before the open) day 0 is
correctly 2008-10-08. But SPY's return in our Yahoo data is −2.5% while the S&P 500 index fell
only −1.1%. Before 2009, Yahoo's ETF closes do not always match the 4 p.m. index close; SPY and the
index differ by more than 0.5 pp on 6 of our 222 event days, all but one before 2009. Because the
target is sector minus SPY on the same data source, most of this cancels, but we report results
with and without that event. Full details: `results/data_report.md`.

## 3. Baseline
*(Person 2)*

## 4. GP model

We use gplearn's `SymbolicRegressor` (`src/models/gp.py`). Each run starts from 1,000 random
formulas (depth 2-4) built from `add sub mul div max min abs neg` and constants in [-1, 1], and
evolves them for 40 generations with tournament selection (size 20) on 90% subsamples. The target is
the excess return in percent. Two versions: **v0** uses `surprise_bp` only; **v1** adds the four day −1
features, standardized with train-period mean and sd. Two scopes: one **pooled** model over all nine
sectors (the main result) and one model **per sector** (~93 training events each).

Overfitting control: the parsimony coefficient (length penalty) is chosen from
{0.0005, 0.001, 0.005, 0.01} by fitting on 1999-2006 and scoring MSE on the 2007-2009 validation
years; the model is then refit on all of 1999-2009 with that value. Each configuration is run with
10 seeds and we report the median test score, so no result depends on a lucky seed. Final formulas
are simplified with sympy (`results/gp_formulas.csv`). All settings are in `config.GP` so the pooled
(Charles) and per-sector (Abhi) runs are identical.

## 5. Results and robustness

Full table: `results/summary.csv`; figures: `results/figures/` (`python -m src.plots`); slide text:
`docs/slides_results.md`.

**Pooled (main result).** Test r2_oos at 1 day: zero forecast +0.080, GP v0 +0.080, linear +0.076,
GP v1 +0.074, linear with features +0.056. At 5 days every model is within 0.01 of zero. The
go/no-go rule fixed in advance (GP v1 must beat both linear models, with ≥ 6 of 10 seeds above
linear) gives **NO-GO**: only 1 of 10 seeds beats linear (`results/go_no_go.md`). All 20 pooled
GP v0 seed-runs return a constant formula.

**Per sector.** GP v1 beats the zero forecast in 4 of 18 (sector, horizon) cells and its median gap
to zero is negative at both horizons (−0.04 at 1d, −0.07 at 5d); it fails badly on XLB. Linear beats
zero in 5 of 18, linear with features in 3 of 18.

**Overfitting.** Of the 167 per-sector GP seed-runs whose formula uses any input, 76% have negative
test r2_oos, including runs with in-sample r2 of 0.3-0.55 (`results/gp_overfit.csv`). The XLE 1d
formula `Min(0.022/s², 0.0121/s)` is ~0 for every s ≠ 0 and −1% at s = 0 through gplearn's
protected division; it fits 3 training days and scores −0.41 on the 8 test days with s = 0.

**The zero-forecast artifact.** r2_oos is measured against each sector's 1999-2009 mean. Those means
were large (XLU −49 bp, XLP −43 bp, XLF +32 bp) and shrank in 2010-2015, so predicting 0 already
gives +0.08 pooled and up to +0.25 for one sector. We therefore judge every model against the zero
forecast, not against 0.

**Features (Track C).** GP v1 does use the features (at 5 days, `mom_20d` and `beta_1y` appear in
over half of the formulas, `results/gp_interpretation.md`), but test performance does not improve,
and linear with features is worse than plain linear. We keep Track C as a negative result.

**Robustness not run in Stage 1.** The planned flags (`TARGET`, `SCHEDULED_ONLY`, `DROP_EVENTS`,
`docs/robustness_flags.md`) were not merged before the freeze. Stage 2 (branch `stage2-v1`) runs
its own placebo and unscheduled-event checks on the new data.

## 6. Limitations and Stage 2
*(leads)*
