# Stage 1 Plan: Club Presentation (Weeks 6–10)

**Goal:** by the week 10 SFIC presentation, answer one question with honest out-of-sample evidence:

> Does genetic programming find readable, nonlinear rules for how sector ETFs react to Fed surprises that a linear model misses, or does it just overfit?

Either answer is a good presentation. This plan covers **Stage 1 only**. The paper (Stage 2) is in [PROJECT_PLAN.md](PROJECT_PLAN.md) and starts after the presentation. Everything built here is reused there, so keep the file names and columns below.

| | |
|---|---|
| **Data** | v0, already on `main`: 222 FOMC events (1999–2025) × sector ETFs, `surprise_bp` = 2y yield change, target `excess_1d` / `excess_5d` vs SPY |
| **Models** | linear (per ticker + pooled), linear + interactions, GP (gplearn) on the surprise alone and with features |
| **Checks** | sanity check vs known results (Gate 0), 10 GP seeds, shuffled-surprise placebo |
| **Key dates** | Wed **Nov 11**: features + linear done · Sun **Nov 15**: go/no-go · Sun **Nov 22**: final results frozen · Sun **Nov 29**: slides rehearsed · week 10: present |

---

## Roles

| | Owns | Main outputs |
|---|---|---|
| **Person 1, GP & evaluation** | the GP model, the combined comparison, the main results figures | `src/config.py`, `src/data.py`, `src/models/gp.py`, `src/compare.py`, `results/gp.csv`, `results/summary.csv`, figures |
| **Person 2, Linear baseline & checks** | the baseline models, the sanity gate, the placebo and significance checks | `src/models/linear.py`, `results/linear.csv`, `results/gate0.md`, `src/evaluate/placebo.py`, `src/evaluate/stats.py`, figures |
| **Person 3, Features & data quality** | pre-event features, data tests, data sanity report, CI | `src/build_features.py`, `features.csv`, `tests/`, `results/data_report.md`, CI workflow |

**Rules for everyone**
- Work on your own branch (`p1-gp`, `p2-linear`, `p3-features`) and open a PR into `main`. One teammate reviews it before merge.
- Never hard-code split dates or paths; import them from `src/config.py`.
- Data files stay out of git. Small result CSVs and figures under `results/` **are** committed.
- If a piece is late, the others keep going (from the current plan): GP falls back to `surprise_bp` alone, or to a quick baseline of its own.

---

## Shared definitions (Person 1 creates these by Thu Nov 5)

### Split: decide at the first meeting
The paper later needs **2016+ untouched**. For Stage 1:

| | Train | Validation (inside train, for picking GP settings) | Test |
|---|---|---|---|
| **Option A (recommended)** | 1999–2009 | 2007–2009 | **2010–2015** |
| Option B (original plan) | 1999–2015 | 2013–2015 | 2016–2025 |

Option A keeps 2016+ clean for the paper. Option B gives a bigger test set now, but the paper would have to disclose that its test years were already looked at. **Changing the split must be a one-line change in `config.py`.**

With option A, use the **9 original SPDR sectors** (XLB XLE XLF XLI XLK XLP XLU XLV XLY). XLRE and XLC have almost no events before 2016.

### `src/config.py`
```python
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, RESULTS = ROOT/"data"/"raw", ROOT/"data"/"processed", ROOT/"results"

SPLIT = "A"                      # "A" or "B", see docs/STAGE1_PLAN.md
SPLITS = {
    "A": dict(train_end="2009-12-31", val_start="2007-01-01", test_start="2010-01-01", test_end="2015-12-31"),
    "B": dict(train_end="2015-12-31", val_start="2013-01-01", test_start="2016-01-01", test_end="2099-12-31"),
}
TICKERS = ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"] if SPLIT == "A" else \
          ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY","XLRE","XLC"]
HORIZONS = ["1d", "5d"]
SEEDS = list(range(10))
FEATURES = ["surprise_bp", "vix", "slope_2s10s", "beta_1y", "mom_20d"]
```

### `src/data.py`
```python
def load_panel(with_features: bool = True) -> pd.DataFrame:
    """returns ⋈ events (⋈ features if the file exists). Filter to TICKERS. Adds column `split`
    in {"train","val","test","unused"} from config.SPLITS[SPLIT] (val is a subset of train years)."""
```

### Result file formats (all models)
- `results/preds/<model>.csv` with columns `event_date, ticker, horizon, split, seed, y, yhat` (seed = 0 for deterministic models)
- `results/<model>.csv` with columns `ticker, horizon, coef, tstat, r2_oos, rmse_oos, hit_rate, complexity` (`ticker = "POOLED"` for pooled rows; coef and tstat blank for GP)

### Metric definitions (identical for every model)
- `r2_oos = 1 − Σ(y − ŷ)² / Σ(y − ȳ_train)²` over **test** rows, where `ȳ_train` is that ticker's train-period mean. Negative means worse than predicting the average.
- `rmse_oos`, and `hit_rate` = share of test rows where `sign(ŷ) == sign(y)`.

---

## Week 6 (Tue Nov 3 – Sun Nov 8): setup and first code

### All three (by Thu Nov 5)
1. `git pull` on `main`, then create a fresh virtual environment.
2. `pip install -r requirements.txt statsmodels matplotlib sympy pytest`.
3. `python -m src.fetch_data`, then `python -m src.build_events`, then `python -m src.build_returns`.
4. Confirm `events.csv` is 222 × 6 and `returns.csv` is 2,142 × 9. Post the numbers in the group chat.
5. **gplearn check** (Person 1 runs this first and tells the others): `python -c "from gplearn.genetic import SymbolicRegressor as S; import numpy as n; S(generations=2).fit(n.random.rand(50,2), n.random.rand(50))"`. If it errors with `_validate_data` or something similar, the installed scikit-learn is too new for gplearn 0.4.2. Use Python 3.11 or 3.12 with `pip install "scikit-learn==1.5.2"`.
6. First meeting: decide **split A or B**, the presentation date, and who reviews whose PRs.

### Person 1: GP & evaluation
1. **Thu Nov 5:** create `src/config.py` and `src/data.py` (above) and merge them quickly, since the others import them.
2. Create `src/models/gp.py` with:
   ```python
   from gplearn.genetic import SymbolicRegressor
   FUNCS = ("add", "sub", "mul", "div", "max", "min", "abs", "neg")
   def make_gp(seed, parsimony):
       return SymbolicRegressor(population_size=1000, generations=40, tournament_size=20,
           function_set=FUNCS, parsimony_coefficient=parsimony, const_range=(-1.0, 1.0),
           init_depth=(2, 4), max_samples=0.9, random_state=seed, n_jobs=1)
   ```
3. **Scale the target:** fit on `excess_1d * 100` (percent), then divide predictions by 100. gplearn's constants live in (−1, 1), and raw returns around 0.005 make it struggle.
4. **GP v0 (surprise only):**
   - inputs: `surprise_bp`
   - one **pooled** model over all tickers (main), plus per-ticker models (secondary; only about 90 training events each, so expect overfitting)
   - parsimony grid `[0.0005, 0.001, 0.005, 0.01]`: fit on train minus val years, pick the value with the lowest val MSE, refit on the full train, predict test
   - repeat for `SEEDS` (10 seeds)
5. Write `results/preds/gp_v0.csv` and `results/gp_v0.csv` (median r2_oos across seeds per ticker; also store the min and max).
- **Done when:** `python -m src.models.gp --version v0` runs end to end and prints a per-ticker table.

### Person 2: Linear baseline
1. Create `src/models/linear.py`:
   - for each ticker in `TICKERS` and each horizon, `statsmodels.OLS(excess_h ~ const + surprise_bp)` on **train**
   - store `coef`, `tstat` (HC1 robust SE), and the test-set `r2_oos`, `rmse_oos`, `hit_rate`
   - a **pooled** row: one regression on all tickers stacked
2. Write `results/linear.csv` (format above) and `results/preds/linear.csv`.
3. Start the **Gate 0** script `src/evaluate/gate0.py`:
   - regress **SPY's** `spy_1d` on `surprise_bp` over all years (one row per event)
   - regress each ticker's **raw** `ret_1d` on `surprise_bp`
   - expected (Bernanke-Kuttner 2005): the SPY slope is **negative** (a hawkish surprise lowers stocks); the largest |slopes| are in cyclical / long-duration sectors (XLK, XLY, XLF, XLI) and the smallest in XLU, XLP, XLV
- **Done when:** `results/linear.csv` exists with real numbers, even if not final.

### Person 3: Features
1. Create `src/build_features.py`:
   - load `prices.csv`, `fred.csv`, `events.csv`
   - for each event, **day −1** = the last trading day *before* `trade_date` in the **price** calendar
   - `vix`, `slope_2s10s = DGS10 − DGS2`: the FRED value on day −1; if missing (bond holiday), use the most recent earlier value, **never a later one**
   - `mom_20d = P[day −1] / P[day −21] − 1` (21 trading rows back)
   - `beta_1y`: OLS slope of the ticker's daily returns on SPY's over the 252 trading days ending day −1; NaN if fewer than 126 days
   - `asof_date = day −1`
   - output `data/processed/features.csv` with columns `event_date, ticker, vix, slope_2s10s, beta_1y, mom_20d, asof_date`, one row per (event_date, ticker) that exists in `returns.csv`
2. Print summary stats per column; check that VIX looks right (for example, about 25–80 in Oct 2008).
- **Done when:** `features.csv` row count equals `returns.csv`'s (2,142), with NaNs only where expected.

---

## Week 7 (Mon Nov 9 – Sun Nov 15): first results and go/no-go

### Person 3: features PR + tests + CI (PR by Wed Nov 11)
1. Open the PR for `build_features.py`. Add the build step to the README's run list.
2. Create `tests/test_features.py`:
   - `asof_date < trade_date` for every row
   - pick 3 random rows and recompute `mom_20d` and `beta_1y` by hand in the test; they match to 1e-9
   - for every event, FRED's day −1 value comes from a date ≤ the price calendar's day −1
   - no duplicate (event_date, ticker)
3. Create `tests/test_returns.py`: `excess_1d == ret_1d − spy_1d`, and no row for XLRE before 2015-10 or XLC before 2018-06.
4. Add `.github/workflows/tests.yml` running `pytest` on every PR. Since data isn't in git, have the tests build from a small fixture: commit `tests/fixtures/prices_small.csv` and `fred_small.csv` (about 300 days around 2008) and point the builders at them through a `data_dir` argument.
- **Done when:** the PR is merged and CI is green.

### Person 2: linear PR, Gate 0, interactions (PR by Wed Nov 11)
1. Open the PR for `linear.py` + `results/linear.csv`.
2. Finish Gate 0 and write `results/gate0.md`: a table of slopes and t-stats for SPY and each ticker, plus a **PASS/FAIL** line against the expected signs. Post it in the chat. **If it fails, Person 3 and Person 2 debug the data before the go/no-go.**
3. Once features are merged, add `linear_interact` to `linear.py`:
   `excess_1d ~ surprise_bp + surprise_bp:vix + surprise_bp:slope_2s10s + surprise_bp:hawk + vix + slope_2s10s`, with `hawk = 1{surprise_bp > 0}`. Fit per ticker and pooled, and write `results/linear_interact.csv`.
- **Done when:** `gate0.md` is posted and `linear_interact.csv` exists.

### Person 1: GP v1 with features + comparison (go/no-go Sun Nov 15)
1. **GP v1:** inputs `surprise_bp, vix, slope_2s10s, beta_1y, mom_20d`. Standardize the features using **train-period** mean and std only (not `surprise_bp`, which stays in bp for readability). Same parsimony and seed procedure as v0.
2. Formula extraction in `src/models/gp.py`:
   ```python
   import sympy as sp
   CONV = {"add": lambda a,b: a+b, "sub": lambda a,b: a-b, "mul": lambda a,b: a*b,
           "div": lambda a,b: a/b, "neg": lambda a: -a, "abs": sp.Abs,
           "max": sp.Max, "min": sp.Min}
   formula = sp.simplify(sp.sympify(str(est._program), locals=CONV))
   ```
   Rename `X0, X1, …` to the feature names. Save the chosen formula for each seed to `results/gp_formulas.csv` with columns `version, ticker, seed, parsimony, length, formula`.
3. Create `src/compare.py`: read every `results/*.csv` model file and write `results/summary.csv`, one row per (model, ticker, horizon), with r2_oos (median for GP), rmse_oos, hit_rate and complexity. Print the pooled rows.
4. **Sun Nov 15 go/no-go meeting:** show `summary.csv`. The decision rule, stated before you look:
   - **GO** ("GP finds something") if pooled GP v1's median r2_oos beats both `linear` and `linear_interact`, **and** at least 6 of 10 seeds beat `linear`
   - otherwise **NO-GO**: the story is "GP overfits shocks". Keep going with the same plan; only the slide narrative changes
- **Done when:** the decision and the numbers are written in `results/go_no_go.md`.

---

## Week 8 (Mon Nov 16 – Sun Nov 22): checks, figures, final numbers

### Person 2: placebo + significance + baseline figure
1. Create `src/evaluate/placebo.py`:
   - `shuffle_surprise(panel, seed)`: permute `surprise_bp` across **events** (all tickers of one event get the same shuffled value), train rows only
   - run linear and pooled GP v1 on 10 shuffles. Person 1 provides a `fit_predict(panel, seed)` function in `gp.py` so Person 2 can call it.
   - write `results/placebo.csv` (model, shuffle_seed, r2_oos). The expected r2_oos is ≤ 0. If the real GP isn't clearly better than its placebo version, that's the "overfitting" evidence.
2. Create `src/evaluate/stats.py`, a **Clark-West test** of pooled GP v1 vs linear:
   - first average squared errors across tickers **within each event**, giving one number per event (the 9 tickers on one Fed day aren't independent)
   - `f_t = e_lin² − (e_gp² − (ŷ_lin − ŷ_gp)²)`; regress `f_t` on a constant with Newey-West SE (`statsmodels`, `cov_type="HAC", maxlags=4`); one-sided p-value
   - write the result to `results/stats.md`
3. Figure `results/fig_baseline_scatter.png`: a 3×3 grid, one panel per ticker, `surprise_bp` (x) vs `excess_1d` (y) with the train-fit line; title each panel with slope and t-stat.
4. Write a ½-page "Baseline" section in `docs/stage1_writeup.md`.

### Person 1: final GP results + figures
1. Rerun GP v0 and v1 with the final settings. **Freeze the numbers Sun Nov 22**: no new model changes after that.
2. Figure `results/fig_r2_by_ticker.png`: grouped bars of r2_oos by ticker for linear, linear_interact, GP v0, GP v1 (with min–max whiskers across seeds), and a zero line.
3. Figure `results/fig_gp_surface.png`: for the most common pooled GP v1 formula, a heatmap of predicted `excess_1d` with `surprise_bp` (−25…+25) on x, `vix` (10…60) on y, other features at their train median. One panel each for XLF, XLK, XLU.
4. Formula table: the 3 most frequent formula *shapes* across seeds, with their length, plus "how many of the 10 seeds include `surprise_bp × vix` (or × slope)". That shows whether the "rule" is stable or a fluke.
5. Write a ½-page "GP model" section in `docs/stage1_writeup.md`: settings, seeds, what the formulas say in plain English.

### Person 3: data report + extra features + 2008-10-08 check
1. **Fix the 2008-10-08 SPY check** (open item from the current plan). That day was the coordinated 50 bp cut announced **before** the market opened. Check that `trade_date` = 2008-10-08, and that SPY's day-0 return matches an outside source (about −1%). Document what was wrong and how it was fixed in `results/data_report.md`.
2. `results/data_report.md`, a data sanity report:
   - events per year; rows per ticker
   - a `surprise_bp` histogram (`results/fig_surprise_hist.png`)
   - the **10 largest |surprise_bp| events**, each with a one-line real-world explanation (for example 2001-01-03 emergency cut, 2008 crisis cuts, 2022 hikes), plus a note on any that look like non-Fed news days (this measure's known weakness)
   - correlation matrix of the features
3. Optional extra features for a GP v2 (only if done by Wed Nov 18): `credit` (BAA10Y), `ff` (DFF), `vol_60d`. Add them as columns; Person 1 decides whether to run them.
4. Robustness rows (Person 1 runs them, Person 3 writes the configs as flags in `config.py`): **scheduled meetings only** (drop the 7 emergency cuts), and **`excess_5d`** instead of `excess_1d`.
5. Write a ½-page "Data & features" section in `docs/stage1_writeup.md`.

### Leads: review (Sat–Sun Nov 21–22)
Review `summary.csv`, the figures, `placebo.csv` and `stats.md`. Ask: do the claims match the numbers?

---

## Week 9 (Mon Nov 23 – Sun Nov 29): slides and rehearsal

### Slide plan (about 12 slides, 12–15 minutes)
| # | Slide | Owner |
|---|---|---|
| 1 | Title + question | Leads |
| 2 | Why it matters: Fed surprises move sectors differently (one real example day) | Leads |
| 3 | What GP is, in one picture (formula tree, evolve → select → mutate) | Person 1 |
| 4 | Data: 222 Fed days × 9 sectors; what a "surprise" is; timeline of day −1 → announcement → day 0 | Person 3 |
| 5 | Data sanity: surprise histogram + biggest surprise days | Person 3 |
| 6 | Baseline: the linear relationship (scatter grid) + Gate 0 passed | Person 2 |
| 7 | How we test fairly: train / validate / test years, no future data, 10 seeds | Person 2 |
| 8 | Results: OOS R² by sector, all models (bar chart) | Person 1 |
| 9 | What GP found: formula + response surface | Person 1 |
| 10 | Is it real? Placebo + Clark-West + seed stability | Person 2 |
| 11 | Answer: GO or NO-GO story, plus limitations (small sample, proxy surprise) | Leads |
| 12 | Next: the paper plan (official Fed surprise data, more models, GECCO 2027) | Leads |

### Tasks by person
- **Person 1:** slides 3, 8, 9. Export figures at 200 dpi, and add one-sentence captions written so a non-quant can follow them.
- **Person 2:** slides 6, 7, 10. Prepare a backup slide with the full `summary.csv` table.
- **Person 3:** slides 4, 5. Make sure the repo's README run steps work from a fresh clone (ask someone outside the team to try). Tag the commit used for the results: `git tag stage1-results`.
- **All:** dry run by **Tue Nov 24** (Thursday Nov 26 is Thanksgiving), with a full rehearsal and timing by **Sun Nov 29**.
- **Prepare answers to the likely questions:**
  - "Isn't this overfitting?" → the placebo and the seeds
  - "Why the 2-year yield?" → it's a free proxy; the paper uses the official Fed surprise series
  - "Could you trade it?" → not shown; the surprise is only known at 2 pm on the day

---

## Week 10 (Nov 30 →): present and hand off
1. Present.
2. Within 2 days, write `docs/stage1_retro.md`: what worked, what broke, what reviewers will attack, and the open questions from the audience.
3. Merge everything to `main`. Stage 2 starts from `stage1-results`; see [PROJECT_PLAN.md](PROJECT_PLAN.md), "Week 10" onward.

---

## Stage 1 checklist
- [ ] Everyone rebuilt the data (222 / 2,142 rows) (Thu Nov 5)
- [ ] Split decided and set in `config.py` (first meeting)
- [ ] `features.csv` + tests merged; CI green (Wed Nov 11)
- [ ] `linear.csv` merged; `gate0.md` says PASS (Wed Nov 11)
- [ ] GP v0 + v1 with 10 seeds; `summary.csv`; `go_no_go.md` (Sun Nov 15)
- [ ] Placebo, Clark-West, data report, 2008-10-08 fix (Sat Nov 21)
- [ ] Numbers frozen and figures final (Sun Nov 22)
- [ ] Slides + rehearsal (Sun Nov 29)
- [ ] `stage1-results` tag; retro written (week 10)

## What carries into Stage 2
`config.py`, `data.py`, the result formats, `linear.py`, `gp.py`, `compare.py`, `placebo.py`, `stats.py` and the tests all carry over. Stage 2 adds the official SF Fed surprise and Ken French industries in the same columns, swaps or adds PySR, and runs on the locked 2016+ test set.
