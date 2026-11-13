# Stage 2: official Fed surprise, Ken French industries, symbolic regression

Stage 1 used a 2-year-yield proxy for the Fed surprise and 11 SPDR sector ETFs from 1999, and
found nothing beyond the zero forecast. Stage 2 rebuilds the study on the data the literature
uses, widens the cross-section to 49 industries, adds PySR, gradient-boosted trees and a placebo
test, and keeps the 2016+ period locked until the end. Code: `src/v1/`. Run order: README.

## 1. Question

On an FOMC announcement day, how does each industry's stock return respond to the monetary
policy surprise, and can symbolic regression find an interpretable formula for that response
that predicts out of sample better than standard models?

Two targets, both reported (fixed before any result of this design was seen):

- `ret_1d`: the industry's return, close of day -1 to close of day 0. The Fed's total effect.
- `excess_1d`: the same minus the CRSP value-weighted market. The effect beyond the market,
  the hardest test: it asks which industries react more or less than the market.

## 2. Data

| piece | source | notes |
|---|---|---|
| Fed surprise | SF Fed US Monetary Policy Event-Study Database (USMPD, Acosta et al.), sheet "Monetary Events" | ED1-ED4 futures changes around the statement and press conference, 1994-02 to 2026-10 |
| Official Bauer-Swanson MPS | SF Fed `monetary-policy-surprises-data.xlsx` | validates our construction; MPS_ORTH used as a robustness shock (ends 2023-12) |
| Industry returns | Ken French 49 industries (main) and 12 industries, daily, value-weighted | 49: all industries have returns on 100% of dev trading days |
| Industry characteristics | Ken French industry files: BE/ME (annual), average firm size (monthly) | Fama-French timing: BE/ME row t is used from July of year t |
| Market | Fama-French Mkt-RF + RF | benchmark for `excess_1d` |
| Macro state | FRED: DGS2, DGS10, DGS3MO, VIXCLS, BAA10Y, DFF | as of day -1 |

Every download is logged in `data/raw/v1/manifest.json` (URL, bytes, sha256, time) and in
[DATA_CARD.md](DATA_CARD.md).

## 3. Construction

- **Shock `mps_bp`**: first principal component of ED1-ED4 changes, positive = hawkish, scaled
  so a 1 bp shock moves ED4 by 1 bp (Bauer & Swanson 2023). Weights fitted on dev years only.
  Correlation with the official MPS over the dev years: 0.975 (Stage 1's proxy: 0.54).
- **Event day**: a release at or after 16:00 ET counts on the next trading day; weekend releases
  roll forward; pre-open releases (2008-10-08) count the same day.
- **Features**, all known before day 0:
  - industry: 60-day beta to the market (`beta_mkt`) and to 2-year yield changes (`beta_2y`),
    20-day momentum, 60-day volatility, log book-to-market (`bm`), log average firm size (`size`),
    and `fomc_beta`: the industry's own slope on the surprise over all **earlier** FOMC events
    (% per 10 bp), shrunk toward 0 by n / (n + 20), matched to the target (raw or excess).
  - state: VIX, 2s10s and 10y-3m slopes, BAA-10y credit spread, fed funds, market 20-day
    momentum, zero-lower-bound flag.
- Panel: 279 events x 49 industries (13,671 rows); 176 scheduled dev events.

## 4. Models

All models see the same rows and inputs. Features are z-scored with the training rows' statistics
and capped at +-3 sd (in 2008 VIX reached 5+ training sd, and unbounded formulas blew up).

| model | what |
|---|---|
| zero, mean | predict 0; each industry's training mean (the r2 benchmark) |
| linear, linear_pooled | a + b*s per industry / pooled (Bernanke-Kuttner) |
| linear_beta | s, s*beta_mkt, beta_mkt (CAPM channel) |
| linear_fomc | s, s*fomc_beta, fomc_beta (the industry's own Fed history) |
| linear_interact_pooled | s times every characteristic and state variable |
| enet | ElasticNet on all features and all s-products |
| hgb | gradient-boosted trees (black-box benchmark) |
| gp_pooled | gplearn genetic programming (Stage 1's engine), one formula |
| pysr_pooled | PySR, one free formula of s and all inputs |
| pysr_sens | PySR sensitivity model: y = industry mean + s10 * g(x), where g is a free formula of the characteristics, state and hawk flag found by the search; g is the industry's response per 10 bp |
| linear_char | s times {beta_mkt, fomc_beta, B/M, size}, plus main effects (the richest hand-built linear model) |
| rf, mlp | random forest; feed-forward net 32-16-8 averaged over 5 initialisations (the Gu-Kelly-Xiu 2020 black boxes) |
| gp_sens | gplearn in the sensitivity form with linear scaling (Keijzer 2003): each program g is scored after the best a, b in y = s10*(a + b*g) |
| pysr_mdl | sensitivity model, one search on the whole fold, formula picked by minimum description length (Bartlett et al. 2024, simplified, N = events) |
| pysr_bag | sensitivity model, 8 bootstrap samples of events, each bag's formula picked on its out-of-bag events, forecasts averaged |
| pysr_anchor | linear_beta plus a bagged PySR search for the *residual* sensitivity; "add nothing" is always a candidate |
| pysr_mv | pysr_anchor plus industry-specific sensitivity offsets, fitted on out-of-bag residuals and shrunk like fomc_beta (multi-view SR as random effects) |
| pysr_enet | every formula on the PySR Pareto front becomes a feature s10*g_k(x); an ElasticNet over those and the linear_beta terms weights them (FEAT-style, La Cava et al. 2019) |
| linear_jk, pysr_jk | add the Jarocinski-Karadi (2020) information-shock flag: rates and stocks moved the same way in the 30-minute window. Uses same-day market information, so reported separately |
| combo_linear, combo_sr, combo_all | equal-weight forecast combinations fixed in advance (Rapach, Strauss & Zhou 2010) |

Why PySR: gplearn draws constants at random and never refines them, so small coefficients (a
0.1% effect per 10 bp) are nearly unreachable, and its protected division creates spikes. PySR
optimizes every candidate's constants (BFGS), keeps a Pareto front of accuracy vs size, and
accepts a custom loss. Why the sensitivity model: an event study's object is the *response to the
surprise*. Free symbolic regression on raw returns spent its formulas on the market's mood on Fed
days (VIX, zero lower bound) with no surprise term at all; that is not a Fed effect. `pysr_sens`
says only that the surprise acts through a sensitivity g(x) and leaves g free: it does not hand the
search beta or any other answer, and g can be any formula of 11 inputs. Implementation: the
surprise is passed as PySR's sample weights and a custom Julia loss scores g * weight (a
two-formula template was 12 times slower for the same structure).

Tuning: every tuned model picks its setting on the last 3 training years of each fold, then refits
on the whole fold. For PySR the setting is the formula size, chosen by the 1-SE rule: the
smallest formula whose inner-tune error is within one standard error (counted in events) of the
best. PySR and gplearn run 10 / 5 seeds; we report the seed-averaged forecast and the
median seed.

Why the hybrid SR models (added after the first full dev run; see deviation 7). The 1-SE rule made
`pysr_sens` choose a constant sensitivity in every fold and seed, so it only re-estimated the
average Fed effect. Two things can stop a search from keeping real structure: the selection rule is
too strict for a 3-year tuning window (about 24 events), or the structure is too noisy to survive a
single search. The hybrids test each explanation separately. `pysr_mdl` replaces the inner split
with a description-length rule on all training years. `pysr_bag` averages many searches, so a
richer formula is kept only if it helps on held-out events. `pysr_anchor` starts from the CAPM
channel, so the search only has to find what beta misses, which is the paper's actual question.
`pysr_mv` adds shrunk per-industry offsets on top. `pysr_enet` uses the search to generate
candidate nonlinear features and lets a penalised regression decide how much each one helps. A
PySR template with 49 per-industry parameters (true multi-view SR) did not finish 40 iterations
in 30 minutes, so `pysr_mv` uses shrunk offsets instead.

## 5. Evaluation

- **Dev**: 1994-2015, four expanding folds (train through 2003 / 2006 / 2009 / 2012, validate on
  the next three years).
- **Locked test**: 2016 onward. `splits.locked_test()` refuses unless `UNLOCK_TEST=1` and logs every
  unlock (time, user, git SHA). `python -m src.v1.final` is the one-time evaluation, run after the
  freeze.
- **Metrics**: out-of-sample r2 vs each industry's training mean; cross-sectional r2 (within-event
  spread only); rank IC (Spearman across industries per event). Significance with one loss per
  event: Clark-West vs the mean, paired tests vs zero and vs `linear_fomc`.
- **Placebo**: the shock is permuted across events (20 permutations for the fast models, 10 for
  PySR). A real finding must beat the placebo distribution.
- **Robustness**: official MPS_ORTH as the shock; unscheduled meetings included; 12 industries.

## 6. Decisions and deviations (made before any test-period result was seen)

1. `mps_bp` rather than the orthogonalized MPS as the main shock: the official MPS_ORTH stops in
   2023-12, which would cut the test period. MPS_ORTH covers every dev year, so it is a full
   robustness row.
2. No protected division in GP (Stage 1's formulas were made of x / s spikes).
3. Scheduled meetings only in the main sample.
4. Target winsorized at 0.5% / 99.5% on training rows only.
5. Two changes made during dev, both driven by dev-fold failures and applied before the
   final run: features capped at +-3 sd for every model (fold 2, 2007-2009, broke every
   nonlinear model through VIX extrapolation), and the `pysr_sens` model added because free PySR
   modelled market mood instead of the surprise. The free `pysr_pooled` is still reported. PySR
   scores candidates on random 1,024-row batches (its recommended setting for data this size).
6. PySR formula size moved from "lowest inner-tune error" to the 1-SE rule after the first full
   run picked 11-17-node formulas on a 3-year tuning window and overfit (fold 2). Same rule for
   both PySR models and their placebo runs.
7. After the first full dev run (with `pysr_sens` constant everywhere) we added the hybrid SR models,
   `gp_sens`, the `rf`, `mlp` and `linear_char` baselines, the `cbi` models and the three
   combinations. All were specified before they were run and are reported in full, including the
   ones that lose. They run on the same dev folds and get the same placebo test (10
   permutations). This adds models, so the paper reports how many were tried and treats the
   locked test (run once, on the frozen list) as the evidence. Compute: one full PySR search takes
   about 7 minutes on one core, so the dev run uses 2 seeds for the bagged models (each bag runs 40
   iterations, not 100), 3 for `pysr_enet`, 5 for `pysr_mdl` and `gp_pooled`, and 5 placebo
   permutations for `pysr_anchor`, `pysr_mv` and `pysr_mdl`. The final run uses more.

## 7. Dev results (validation years 2004-2015; full tables in `results/v1/STAGE2_REPORT.md`)

**Gate 2 passes.** A +10 bp hawkish surprise moves the S&P 500 by -0.48% inside the event window
(t = -6.0) and the market by -0.58% over the day (t = -2.7); Stage 1's proxy gave t = -0.65.
All 49 industries fall on hawkish surprises; 10 of 49 move significantly differently from the
market (Mines, Construction, Gold, Steel most).

**Raw returns (`ret_1d`, 49 industries)**

| model | r2_oos | beats zero (p) | placebo p | cross-sectional r2 | rank IC (p) |
|---|---|---|---|---|---|
| zero | -0.038 | | | | |
| linear_pooled (a + b*s) | +0.066 | 0.013 | <0.05 | 0 | |
| **linear_beta** (s, s*beta, beta) | **+0.079** | 0.018 | <0.05 | **+0.015** | **0.12 (0.001)** |
| linear_fomc | +0.064 | 0.016 | <0.05 | -0.004 | 0.05 (0.03) |
| ElasticNet / trees / interactions | -0.12 to -0.23 | no | | | |
| gplearn GP | -0.049 | no | | | |
| PySR free formula | -0.027 | no | | | |
| **PySR sensitivity** | **+0.061** | 0.016 | <0.10 (0 of 10) | -0.006 | |

**Excess returns (`excess_1d`, 49 industries)**

| model | r2_oos | beats zero (p) | placebo p | cross-sectional r2 | rank IC (p) |
|---|---|---|---|---|---|
| zero | +0.008 | | | | |
| linear_beta | +0.018 | 0.029 | 0.10 | +0.009 (placebo p 0.50) | 0.11 (0.005) |
| **linear_fomc** | +0.013 | 0.064 | **<0.05** | +0.005 (placebo p <0.05) | 0.06 (0.010) |
| ElasticNet | +0.011 | 0.34 | 0.05 | | |
| trees, interactions, GP, PySR | -0.04 to +0.007 | no | | | |

What this says:

1. **The Fed surprise predicts industry returns on announcement days, out of sample.** Every
   model built around "return = sensitivity x surprise" earns r2 of 6-9% (12 industries: up to
   9.4%), significant against the zero forecast, and far above every placebo run.
2. **Symbolic regression recovers that law on its own, and finds it is simple.** With the
   sensitivity structure, PySR picks the same formula in every seed and fold: a constant
   sensitivity of -0.28% (trained to 2003), -0.38%, -0.58%, -0.65% (trained to 2012) per 10 bp.
   It is stable across seeds to the third decimal, and it grows as later meetings enter the
   training data: the estimated sensitivity more than doubles once 2007-2012 is included.
3. **Free-form search does not find it.** Unconstrained PySR and gplearn spend their formulas
   on the market's mood on Fed days (VIX, zero lower bound) and lose to zero; with the 1-SE
   rule they return constants. The structure of the search, not the engine, decides whether the
   Fed effect is found.
4. **Heterogeneity across industries is real but small.** Only market beta reliably ranks
   industries (rank IC 0.12, cross-sectional r2 +1.5%, beats every placebo). Beyond the market
   (`excess_1d`), the industry's own past Fed sensitivity is the one signal that survives the
   placebo test; book-to-market, size, momentum and macro-state interactions add nothing out of
   sample, and flexible models (ElasticNet, trees) overfit.
5. **Where it is weaker.** The last fold (2013-2015, near-zero rates, tiny surprises) is close to
   zero for every model. With MPS_ORTH (the surprise purged of the part predictable from prior
   macro news, Bauer & Swanson 2023) the raw-return r2 falls to 2-3%: part of the predictable
   reaction travels with that component. Including unscheduled
   meetings changes little.

Honest limits: 176 dev events is a small sample for discovering nonlinear heterogeneity; the
symbolic search, held to the 1-SE rule on a 3-year tuning window, never found a characteristic
whose effect was robust enough to keep. The result is a clean, interpretable confirmation of the
beta channel and of a time-varying aggregate sensitivity, not a new nonlinear law.

## 8. Run it

```
python -m src.v1.fetch && python -m src.v1.build && python -m src.v1.gate
python -m src.v1.run                     # main: ff49, both targets (~3 h with PySR on 6 workers)
python -m src.v1.run --variant orth      # robustness; also: unsched, --universe ff12
python -m src.v1.run --variant placebo3 --models linear_beta,pysr_sens --seeds 1
python -m src.v1.run --models linear_jk,pysr_jk          # information-shock models (same-day stock move)
python -m src.v1.report                  # -> results/v1/STAGE2_REPORT.md, fig_*.png
UNLOCK_TEST=1 python -m src.v1.final     # ONCE, after the freeze
```

## 9. Still to do before the paper

- Freeze the model list, then run `UNLOCK_TEST=1 python -m src.v1.final` once.
- 30 seeds for the final PySR runs.
- Finish the free-PySR placebos (2 of 10 ran; that model never uses the surprise, so no conclusion depends on them):
  `for k in 2..9: python -m src.v1.run --variant placebo$k --models pysr_pooled --seeds 1`
