# Stage 2: official Fed surprise, 12 industries, 1994-2026

Stage 1 used a 2-year-yield proxy for the Fed surprise and SPDR sector ETFs from 1999. It found
nothing beyond the zero forecast. Stage 2 rebuilds the data on the sources the literature uses,
keeps the 2016+ period locked, and runs every model through the same walk-forward folds. Code is in
`src/v1/`; the run order is in the README.

## 1. Data

| piece | source | notes |
|---|---|---|
| Fed surprise | SF Fed US Monetary Policy Event-Study Database (USMPD, Acosta et al.), sheet "Monetary Events" | ED1-ED4 futures changes in a window around the statement and press conference, 1994-02 to 2026-10 |
| Official Bauer-Swanson MPS | SF Fed `monetary-policy-surprises-data.xlsx` | used only to validate our construction (ends 2023-12) |
| Industry returns | Ken French 12 industries, daily, value-weighted | 1926 to 2026-09; replaces SPDR ETFs, so the sample starts in 1994 |
| Market | Fama-French Mkt-RF + RF | the benchmark for excess returns |
| Macro state | FRED: DGS2, DGS10, DGS3MO, VIXCLS, BAA10Y, DFF | read as of the day before each event |

Every download is recorded in `data/raw/v1/manifest.json` (URL, bytes, sha256, time) and summarized
in [DATA_CARD.md](DATA_CARD.md).

## 2. Construction

- **Shock `mps_bp`**: first principal component of ED1-ED4 changes, sign so positive = hawkish,
  scaled so a 1 bp shock moves ED4 by 1 bp (Bauer & Swanson 2023). The PCA weights are fitted on
  dev years only. Correlation with the official MPS over the dev years: 0.975.
- **Event day**: a release at or after 16:00 ET counts on the next trading day; weekend releases
  roll forward. Pre-open releases (e.g. 2008-10-08) count on the same day.
- **Target `excess_1d`**: industry return minus market return, close of day -1 to close of day 0.
- **Features** (all from data before day 0): VIX, 2s10s and 10y-3m slopes, BAA-10y credit spread,
  fed funds, 60-day beta to the market and to 2-year yield changes, 20-day momentum, 60-day
  volatility, market 20-day momentum.
- Panel: 279 events x 12 industries.

## 3. Evaluation design

- **Dev**: 1994-2015. Four expanding folds: train from 1994 through 2003 / 2006 / 2009 / 2012,
  validate on the next three years. Hyperparameters (ElasticNet penalty, GP parsimony) are picked on
  the last three training years of each fold, then the model is refit on the full fold.
- **Locked test**: 2016 onward. `splits.locked_test()` raises unless `UNLOCK_TEST=1` and logs every
  unlock (time, user, git SHA) to `results/v1/test_unlock.log`. It is opened once, after the model
  list and settings are frozen.
- **Metrics**: out-of-sample r2 against each industry's training mean (Campbell-Thompson); the
  **zero forecast** row is the fair bar. Significance uses one loss per event (the 12 industries of
  one Fed day share a shock): Clark-West vs the mean, and a paired test vs zero.
- GP runs 10 seeds (pooled) / 5 seeds (per industry); we report the median seed, the range, and the
  seed-averaged forecast.

## 4. Decisions and deviations (made before seeing any test-period result)

1. **`mps_bp`, not the orthogonalized MPS, as the main shock.** The official MPS_ORTH stops in
   2023-12 and needs the authors' macro predictors to extend, which would cut the test period and
   drop the 2024-2026 cuts. `mps_orth_bs_bp` is kept for a robustness row.
2. **No protected division in GP.** gplearn returns 1 when the denominator is near 0, so `x / s`
   jumps for small surprises. Stage 1's formulas were dominated by these artifacts.
3. **Scheduled meetings only** in the main sample (unscheduled ones are few and often
   emergencies). Switch with `INCLUDE_UNSCHEDULED` in `src/v1/settings.py`.
4. **Target winsorized** at 0.5% / 99.5% on training rows only; validation rows are not touched.

## 5. Results so far (dev folds only)

See `results/v1/gate2.md` and `results/v1/dev_report.md` for the full tables.

**Gate 2 passes.** Over the dev years, a +10 bp hawkish surprise moves the S&P 500 by -0.48% inside
the event window (t = -6.0) and the market by -0.58% over the day (t = -2.7). Stage 1's proxy gave
t = -0.65 on the same events. All 12 industries fall on hawkish surprises; cyclicals fall most.

**Dev-fold accuracy** (validation years 2004-2015; r2 vs each industry's training mean):

| model | r2_oos | Clark-West p (vs mean) | p (beats zero) |
|---|---|---|---|
| zero (fair bar) | +0.0106 | 0.002 | |
| linear (per industry, a + b*s) | +0.0065 | 0.09 | 0.60 |
| linear_pooled | +0.0102 | 0.002 | 0.63 |
| **linear_beta** (s, s*beta, beta) | **+0.0173** | <0.001 | 0.12 |
| linear_interact (per industry) | -0.2428 | 0.92 | 0.99 |
| enet (all features x s) | -0.0400 | 0.10 | 0.88 |
| gp_pooled (10 seeds, median) | +0.0108 | 0.002 | 0.20 |

What this says:

- The theory model (an industry's reaction scales with its market beta) is the best and the only
  one clearly above zero in point estimate, but not yet significantly (p = 0.12).
- Flexible models (interactions, ElasticNet) overfit and lose to zero.
- **Pooled GP returns a constant in 39 of 40 runs.** A scratch diagnostic on fold 4 showed why:
  in standardized units the surprise terms move the in-sample fit by about 0.0002 of variance, which
  is smaller than any useful complexity penalty. With the penalty near zero, GP builds large
  formulas that lose to zero on the tuning years. Rescaling the inputs did not change this. GP is
  behaving correctly: on daily data, how an industry moves *relative to the market* on a Fed day
  carries very little predictable signal beyond beta.
- Per-industry GP (5 seeds x 12 industries x 4 folds) is in `results/v1/dev_report.md` once run.

Implication for the paper: the excess-return target is the honest "beyond beta" test, and it is
close to a null. Options are a target with a stronger signal (raw industry returns, where the
market reaction is t = -2.7 daily and -6.0 intraday), a narrower window, or framing the result as
"interpretable search recovers the beta channel and nothing else".

## 6. Still to do before the paper

- 30 seeds for GP, and PySR as a second symbolic-regression engine.
- Placebo tests (shuffled surprises, non-FOMC days) and the `mps_orth` robustness row.
- Freeze the model list, then open the locked test set once.
