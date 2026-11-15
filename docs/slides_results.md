# Stage 1 slides: Charles's part (content + speaker notes)

Frozen results (Sun 11-15, tag `stage1-freeze`). Split A: train 1999-2009, test 2010-2015 (48 FOMC
events × 9 sectors = 432 test rows). Metric `r2_oos` = 1 − SSE / SSE of each sector's 1999-2009 mean.
Figures: `results/figures/` (regenerate with `python -m src.plots`). Numbers: `results/summary.csv`.

Slide numbers follow the outline in `docs/STAGE1_PLAN.md`. Slides 2-5, 7 and 11 are Snata's and
Abhi's; this file only covers slides 1, 6, 8, 9, 10 and 12.

---

## Slide 1: Title

**Can Genetic Programming Predict How Sectors React to Fed Surprises?**
SFIC Fall 2026 · Stage 1
Leads: Razeen Khan, Benny Signer · Members: Charles Cheng, Snata Mohanty, Abhishai Ganta

---

## Slide 6: Models

Three ways to predict a sector's excess return after an FOMC announcement:

| model | what it does | inputs |
|---|---|---|
| zero forecast | always predict 0 (no reaction beyond the market) | none |
| linear | OLS: excess return = a + b · surprise | surprise_bp |
| linear + features | OLS with VIX, 2s10s slope, beta, momentum, and surprise × VIX / slope / sign | 5 inputs + interactions |
| GP v0 | genetic programming searches over formulas | surprise_bp |
| GP v1 | same, with the 4 pre-event features | 5 inputs |

**GP in one line:** start with 1,000 random formulas built from `+ − × ÷ min max abs neg`, keep the
ones that fit the training years best, recombine and mutate them for 40 generations. A parsimony
penalty (picked on 2007-2009 validation years) punishes long formulas. 10 random seeds per model.

*Visual:* one small tree diagram, e.g. `mul(X3, 0.162)` = `0.162 × beta_1y` (a real pooled v1 formula).

*Speaker notes:* GP's selling point is that it can find a nonlinear formula a human wouldn't write
and still print it. The question for us is whether that flexibility finds real structure or just
fits noise in ~93 events per sector.

---

## Slide 8: Main result: nothing beats "predict 0"

*Figure:* `results/figures/fig1_pooled_r2.png` (pooled), optionally `fig2_vs_zero_heatmap.png` as backup.

**Pooled (all 9 sectors in one model), test 2010-2015:**

| model | 1-day | 5-day |
|---|---|---|
| zero forecast | **+0.080** | **+0.010** |
| linear | +0.076 | +0.010 |
| linear + features | +0.056 | +0.000 |
| GP v0 (surprise only) | +0.080 | +0.004 |
| GP v1 (+ features) | +0.074 | +0.008 |

- Pre-registered rule (written before we looked): GO if GP v1 beats both linear models and ≥ 6 of
  10 seeds beat linear. Result: **1 of 10 seeds. NO-GO.**
- Per sector: GP v1 beats the zero forecast in 4 of 18 (sector, horizon) cells, about what luck gives.
  It blows up on materials (XLB: −0.43 at 1d, −0.36 at 5d).
- Adding features made linear *worse* (+0.056 vs +0.076), and GP v1 is not reliably better than v0.

*Speaker notes:* The bars that look best are the gray ones: a model that says "nothing happens"
is as good as anything we built. The GP pooled v0 formulas are all constants (see slide 9), so
GP itself concluded the surprise carries no usable signal when sectors are pooled.

---

## Slide 9: What GP learned, and why it is overfitting

*Figures:* `fig4_xle_formula.png` (main), `fig3_overfit.png` (second panel or backup).

**Example: energy (XLE), 1-day, seed 8**
`Min(0.022 / s², 0.0121 / s)` where s = surprise in bp.
- For any s ≠ 0 this is almost exactly 0.
- gplearn's "protected division" returns 1 when dividing by 0, so at **s = 0 the formula predicts
  −1%**. GP found that on the 3 training days with s = 0, energy happened to lag the market.
- In 2010-2015 there are 8 events with s = 0 (rates at the zero lower bound), energy didn't lag,
  and test r2_oos is **−0.41**.

**Across all GP formulas:**
- Pooled GP v0: 20 of 20 seed-runs returned a constant (no input used).
- Per-sector: of the 167 seed-runs that used any input, **76% score below 0 on test**, including runs
  with in-sample r2 of 0.3-0.55 (`fig3_overfit.png`, data in `results/gp_overfit.csv`).

*Speaker notes:* The formula is printable, which is GP's advantage, but printable doesn't mean
meaningful: this one encodes three days of noise through a software quirk. With ~93 training events
per sector and a search over millions of formulas, GP will always find something that fits.

---

## Slide 10: A trap: why "+0.08" is not skill

*Figure:* `results/figures/fig5_zero_artifact.png`

- r2_oos compares a model with each sector's **1999-2009 average** reaction on FOMC days.
- Those averages were large (utilities −49 bp, staples −43 bp, financials +32 bp) and **did not
  carry over** to 2010-2015 (−7, −13, +13 bp).
- So simply predicting 0 already scores +0.08 pooled, and up to +0.25 for single sectors (XLY 1d).
- Any model that shrinks toward 0 inherits that score. That's why every result is compared with
  the **zero forecast**, not with 0.

*Speaker notes:* If we had only reported "GP r2_oos = +0.08" this would look like a success. It's the
most important methodological point of the project, and it carries into Stage 2: always report the
zero forecast next to r2_oos.

---

## Slide 12: Conclusion, limitations, future work

**Conclusion**
1. Using the FOMC-day 2-year yield change as the surprise, sector excess returns after FOMC
   announcements are **not predictable out of sample** (2010-2015) by linear models or GP.
2. GP **overfits** small event samples: its non-constant formulas look good in-sample and lose on
   test; the pooled GP decides the best formula is a constant.
3. Adding pre-event state (VIX, curve slope, beta, momentum) **does not rescue it**: GP uses these
   features, but they don't improve test performance.
4. r2_oos against a train mean can look positive for a no-skill model: compare with the zero forecast.

**Limitations**
- Surprise proxy: the full-day 2y yield change also contains non-Fed news; two of the ten largest
  "surprises" move opposite to the policy action.
- Small samples: ~93 training and 48 test events per sector; 2010-2015 is mostly zero lower bound
  (8 test events with no 2y move at all).
- Yahoo ETF closes before 2009 don't always match the 4 p.m. index close (6 of 222 event days).
- One GP configuration (gplearn, 1,000 × 40); no tuning beyond parsimony.

**Future work (Stage 2, already started on branch `stage2-v1`)**
- Official 30-minute-window Fed surprise (Bauer-Swanson, SF Fed) instead of the daily 2y change.
- 49 Ken French industries instead of 9 sector ETFs: more cross-sectional data.
- PySR and hybrid symbolic regression; forecast combinations; placebo tests.
- Locked test set 2016+ (untouched in Stage 1).
- Application idea: sector rotation around FOMC days, only if a model beats the zero forecast.

*Speaker notes:* A negative result here is useful: it tells Stage 2 that a better surprise measure and
more data matter more than a more flexible model.
