# Final Architecture and 3-Person Plan

This is **the** plan: where ARCHITECTURE.md or older notes disagree with it, this file wins. It merges the data and weeks 6–9 plan already on `main` (club presentation in week 10) with the publication plan in [ARCHITECTURE.md](ARCHITECTURE.md), which is the deep reference for *why* each choice was made.

**Stage 1 (club presentation) has its own detailed plan: [STAGE1_PLAN.md](STAGE1_PLAN.md).** That file wins for weeks 6–10.

The work happens in two stages, built on one codebase:

| | **Stage 1: Club result** | **Stage 2: Paper** |
|---|---|---|
| When | Weeks 6–10 (Nov 3 – Nov 30, 2026) | Dec 1, 2026 – Jan 25, 2027 |
| Goal | SFIC presentation: does GP beat a linear model on Fed-day sector reactions? | GECCO 2027 submission (abstract **Jan 18**, paper **Jan 25**) |
| Data | **v0**: 2y-yield surprise, SPDR ETFs, 1999–2025 (already built) | **v1**: published SF Fed surprise, Ken French industries, 1994–2025 (v0 kept as a robustness check) |
| Models | GP (gplearn) vs linear regression | GP (PySR + gplearn) vs 8 baselines, 30 seeds, significance tests, placebos |
| Outcome either way | "GP finds readable rules" **or** "GP overfits shocks". Both are valid findings | Same, with publication-grade evidence |

Roles. The leads stay responsible for review, pre-registration, advisor outreach and the intro and related work.

| | Role | Stage 1 piece (unchanged from the current plan) | Stage 2 piece |
|---|---|---|---|
| **Person 1** | GP & evaluation | `src/models/gp.py`, results table + figures | PySR engine, 30-seed runner, formula analysis |
| **Person 2** | Baselines & statistics | `src/models/linear.py` → `results/linear.csv` | 8 baselines, significance tests, placebos, Gate 0 |
| **Person 3** | Data & features | `src/build_features.py` → `features.csv` | v1 data (SF Fed shocks, Ken French), splits with a test lock, data tests |

---

# Part 1: Final architecture

## 1.1 Data flow

```
                  ┌──────────────────────── STAGE 1 (v0, exists) ────────────────────────┐
fetch_data.py ──► data/raw/prices.csv (SPY + 11 SPDR)   data/raw/fred.csv (DGS2, DGS10, VIX, BAA10Y, DFF [+DGS3MO])
fomc_dates.csv ─► build_events.py ──► events.csv   (surprise_bp = Δ2y, day −1 → day 0)
                  build_returns.py ─► returns.csv  (excess_1d, excess_5d vs SPY)
                  build_features.py ► features.csv (vix, slope_2s10s, beta_1y, mom_20d, ...)   ← Person 3
                  └──────────────────────────────────────────────────────────────────────┘
                  ┌──────────────────────── STAGE 2 adds (v1) ────────────────────────────┐
fetch_shocks.py ► data/raw/mps.parquet, usmpd.parquet  (SF Fed surprises)                  ← Person 3
fetch_french.py ► data/raw/french12.parquet             (Ken French 12 industries + market) ← Person 3
build_events.py   + columns mps_bp, mps_orth_bp, path_bp, zlb, events back to 1994
build_returns.py  --source spdr | french12  (SAME output columns)
build_panel.py  ─► panel.parquet = events ⋈ returns ⋈ features  + DATA_CARD.md
                  └──────────────────────────────────────────────────────────────────────┘
splits.py ──► dev folds + LOCKED test ──► models/* ──► runner.py ──► results/ ──► evaluate/* ──► paper/
```

**Design rule:** v1 keeps v0's file names and column names. A model written in Stage 1 runs on v1 data without changes; only a config flag changes (`shock_col`, `source`).

## 1.2 Schemas (the contract between people)

**events** (one row per FOMC announcement). Existing columns are kept; Stage 2 adds the ones marked ➕.

| column | meaning |
|---|---|
| `event_date`, `trade_date` | announcement date; day 0 = first trading day on or after it |
| `event_type` | `fomc_scheduled` / `fomc_unscheduled` |
| `dgs2_before`, `dgs2_after`, `surprise_bp` | v0 shock: 2y yield change, in bp |
| ➕ `mps_bp`, `mps_orth_bp` | SF Fed Bauer-Swanson raw / orthogonalized surprise (bp). **v1 main shock = `mps_orth_bp`** |
| ➕ `target_bp`, `path_bp` | USMPD target and path factors (converted to bp) |
| ➕ `zlb` | fed funds at the zero lower bound (≤ 0.25%) |

**returns** (one row per event × sector). Same columns for both sources.

| column | meaning |
|---|---|
| `event_date`, `trade_date`, `ticker` | `ticker` = SPDR symbol, or the French industry code (`NoDur`, `Durbl`, `Manuf`, `Enrgy`, `Chems`, `BusEq`, `Telcm`, `Utils`, `Shops`, `Hlth`, `Money`, `Other`) |
| `ret_1d`, `ret_5d` | close day −1 → close day 0 / day +4 |
| `spy_1d`, `spy_5d` | benchmark over the same window (SPY, or the French value-weighted market) |
| `excess_1d`, `excess_5d` | **target** = ret − benchmark |
| ➕ `source` | `spdr` or `french12` |

**features** (one row per event × ticker, using **only data up to day −1**).

| column | meaning | stage |
|---|---|---|
| `vix` | VIXCLS on day −1 | 1 |
| `slope_2s10s` | DGS10 − DGS2 on day −1 | 1 |
| `beta_1y` | 252-day beta of the ticker to SPY / market, ending day −1 | 1 |
| `mom_20d` | ticker's 20-day return ending day −1 | 1 |
| `term_10y3m` | DGS10 − DGS3MO on day −1 | 2 |
| `credit` | BAA10Y on day −1 | 2 |
| `ff` | DFF on day −1 | 2 |
| `vol_60d` | ticker's 60-day realized volatility ending day −1 | 2 |
| `beta_2y` | 252-day beta of the ticker to daily ΔDGS2 | 2 |
| `mkt_mom_20d` | market 20-day return ending day −1 | 2 |
| `asof_date` | = day −1 (lets the tests check for look-ahead) | 1 |

**Results** (every model writes the same two files):
- `results/preds/<model>_<config_hash>.csv` with columns `event_date, ticker, horizon, split, fold, seed, y, yhat`
- `results/<model>.csv` with columns `ticker, horizon, coef, tstat, r2_oos, rmse_oos, hit_rate, complexity` (`coef`/`tstat` filled only for linear models). This extends the current `linear.csv` spec.

## 1.3 Model interface (`src/models/base.py`)

```python
class Model(Protocol):
    name: str
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def complexity(self) -> int: ...     # tree nodes (GP), #params (linear), #leaves (trees)
    def describe(self) -> str: ...       # formula / coefficients, for the paper
```

## 1.4 Splits: one decision to make before Wed Nov 11

The current plan trains on events **before 2016** and tests on **2016+**. If Stage 1 looks at 2016+ results and Stage 2 then tunes models, the paper's test set is no longer clean. Reviewers count that as data snooping.

| Option | Stage 1 split | Stage 2 locked test | Trade-off |
|---|---|---|---|
| **A (recommended)** | train 1999–2009, test **2010–2015** | **2016+**, untouched until Dec 29 | Stage 1 has a slightly smaller test window with small ZLB-era surprises; the paper stays clean |
| B | train <2016, test 2016+ (current plan) | 2016+, disclosed as "viewed in a preliminary analysis with a proxy shock" | Simpler now; weaker paper |

Stage 2 dev folds (inside 1994–2015, expanding window): train to 2003 / val 2004–06; to 2006 / 2007–09; to 2009 / 2010–12; to 2012 / 2013–15.

## 1.5 Gates

| Gate | Date | Test | If it fails |
|---|---|---|---|
| **G0 Sanity** | Wed Nov 11 | Linear `ret_1d ~ surprise_bp`: a hawkish surprise lowers the market, and cyclical and long-duration sectors move most (Bernanke-Kuttner pattern) | Stop and debug the data before any GP conclusions |
| **G1 Go/no-go** | Sun Nov 15 | GP vs linear out of sample (Stage 1 split) | Present "GP overfits shocks" at the club; Stage 2 continues as a rigorous benchmark |
| **G2 v1 sanity** | Sun Dec 6 | G0 repeated on v1 data (`mps_orth_bp`, French 12) | Fall back to v0 data for the paper, disclose it, target a poster or workshop |
| **G3 Freeze** | Mon Dec 28 | All configs tagged `freeze-v1` | n/a |
| **G4 Test** | Dec 29 – Jan 4 | Locked test opened **once** | Report whatever comes out |

---

# Part 2: Step-by-step plan

Week numbers follow the current SFIC plan (week 6 = Nov 3–8). Each step has a **Done when** check.

## Week 6 (Nov 3–8): everyone rebuilds the data

**All three**
1. `git pull`, then `pip install -r requirements.txt`.
2. `python -m src.fetch_data`, then `python -m src.build_events`, then `python -m src.build_returns`.
3. Check: `events.csv` has 222 rows × 6 columns, and `returns.csv` has 2,142 rows × 9 columns.
4. Create your own branch: `p1-gp`, `p2-linear`, or `p3-features`.
5. Team decision: **split option A or B** (§1.4). Write it in `docs/data_interface.md`.
- **Done when:** all three have identical row counts and the split decision is written down.

**Person 1 (GP)**
1. Create `src/models/base.py` with the interface in §1.3.
2. Create `src/models/gp.py`: gplearn `SymbolicRegressor` wrapped behind the interface. Functions `add, sub, mul, div, max, min, abs, neg`; `population_size=1000, generations=40, tournament_size=20, parsimony_coefficient=0.001, random_state=seed`.
3. **GP v0:** input = `surprise_bp` only, target = `excess_1d`, one model per ticker plus one pooled model (ticker as a one-hot, or skip pooling for now).
4. Write predictions in the results format of §1.2.
- **Done when:** `python -m src.models.gp` prints OOS R² per ticker on the Stage 1 split.

**Person 2 (linear)**
1. Create `src/models/linear.py`. For each ticker, fit OLS `excess_1d ~ surprise_bp` on the train split (use `statsmodels` to get t-stats).
2. Compute on the test split: `r2_oos = 1 − SSE_model / SSE_trainmean`, plus RMSE and hit rate (sign agreement).
3. Run a pooled regression across all tickers, then repeat everything for `excess_5d`.
4. **Gate 0 extra:** fit the same model on **`ret_1d`** (raw, not excess) and on SPY itself.
- **Done when:** the script runs end to end, even if the numbers aren't final.

**Person 3 (features)**
1. Create `src/build_features.py` (it's a stub now).
2. Load `prices.csv`, `fred.csv` and `events.csv`. For each event, find day −1 as the trading day before `trade_date` **in the price calendar**.
3. `vix`, `slope_2s10s`: the FRED value on day −1. If it's missing, use the last value on or before day −1, never after.
4. `mom_20d`: `price[day−1] / price[day−21] − 1`.
5. `beta_1y`: regress the ticker's daily returns on SPY's over the 252 trading days ending day −1, and take the slope. Leave it NaN if there are fewer than 126 days (XLRE and XLC early on).
6. Add `asof_date` = day −1.
- **Done when:** `features.csv` has one row per (event_date, ticker) matching `returns.csv`.

## Week 7 (Nov 9–15): first real results and go/no-go

**Person 3**
1. **By Wed Nov 11:** PR `features.csv` + `build_features.py`.
2. Add `tests/test_features.py`:
   - `asof_date < trade_date` for every row
   - recompute `mom_20d` and `beta_1y` by hand for 3 random rows and compare
   - the day −1 from the FRED calendar equals the day −1 from the price calendar (or the code falls back correctly)
3. Add `pytest` to requirements and a GitHub Action that runs `pytest` on every PR.
- **Done when:** the PR is merged and CI is green.

**Person 2**
1. **By Wed Nov 11:** PR `linear.py` + `results/linear.csv` (columns `ticker, horizon, coef, tstat, r2_oos, rmse_oos`, plus `hit_rate`).
2. **Gate 0 report** (a short markdown in `results/gate0.md`):
   - SPY slope on `surprise_bp` (expect negative)
   - which sectors have the largest |coef| (expect XLK, XLY, XLF and cyclicals; XLU and XLP smallest)
3. Add a second baseline, `linear + interactions`: `excess_1d ~ surprise_bp + surprise_bp·vix + surprise_bp·slope_2s10s + surprise_bp·1{surprise>0}`, using features from Person 3.
- **Done when:** `gate0.md` says pass or fail, with the numbers.

**Person 1**
1. **GP v1:** inputs = `surprise_bp, vix, slope_2s10s, beta_1y, mom_20d`, joined from `features.csv`.
2. Run **10 seeds** (not 1) and report the median and spread of OOS R². Never report the single best seed.
3. For the best formula by train-period validation (hold out the last 3 years of train), print it simplified with `sympy`.
4. **Sun Nov 15 go/no-go:** a table comparing GP v0, GP v1, linear and linear + interactions on OOS R², RMSE and hit rate per ticker.
- **Done when:** the team has the table and has made the go/no-go call.

## Week 8 (Nov 16–22): results, figures, and starting Stage 2 data

**Person 1**
1. Final Stage 1 results table and 2 figures: (a) OOS R² by ticker for each model; (b) one GP formula's response surface (surprise × VIX).
2. Write the "GP model" section (½ page): setup, seeds, what the formulas say.
3. Start Stage 2: `pip install pysr`, run the Julia setup, and run a smoke test on 100 rows.

**Person 2**
1. "Baseline" section + 1 figure: a surprise vs excess scatter by sector with fit lines.
2. Write `src/evaluate/stats_tests.py`:
   - `event_level_losses(preds)`: mean squared error across the tickers of each event, giving one number per event. The tickers on one Fed day are not independent.
   - `clark_west(e_small, e_big, yhat_small, yhat_big)`: `f = e_small² − (e_big² − (yhat_small − yhat_big)²)`, regressed on a constant with Newey-West standard errors, one-sided.
   - `diebold_mariano(l1, l2)`: HAC variance with the Harvey-Leybourne-Newbold correction.
   - `holm(pvals)`, a Wilcoxon test across seeds, and the Vargha-Delaney A12 effect size.
3. `tests/test_stats.py`: check against a toy example computed by hand.

**Person 3**
1. "Data & features" section + slides. **Fix the 2008-10-08 SPY check** (an open item from the current plan; that was the coordinated emergency cut, announced before the open).
2. Start v1. Create `src/fetch_shocks.py`:
   - download `https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx` and `https://www.frbsf.org/wp-content/uploads/USMPD.xlsx` to `data/raw/`
   - save the SHA-256 and download date to `data/raw/manifest.json`
   - **open the files and record the first and last meeting date covered** in `docs/DATA_CARD.md`. The web pages don't say.
   - parse to `mps.parquet` (`event_date, mps_bp, mps_orth_bp`) and `usmpd.parquet`; USMPD is in percentage points, so ×100 to get bp
- **Done when:** both parquet files load and their dates match `fomc_dates.csv` for 1999–2025 (list any mismatches).

## Week 9 (Nov 23–29): slides, plus Stage 2 groundwork

**All:** slides draft and rehearsal. Each person owns 2 slides on their piece.

**Person 1**
1. Create `src/models/gp_pysr.py` behind the same interface:
   - `binary_operators=["+","-","*","/","max","min"]`, `unary_operators=["abs","tanh"]`
   - `maxsize=25, maxdepth=8, niterations=200, populations=30`
   - `deterministic=True, parallelism="serial", random_state=seed`
   - keep the **whole Pareto front**, and choose the expression with the lowest **validation** error (ties go to the smaller one)
2. **Synthetic recovery test** (this goes in the paper): plant `y = −0.04·s·(1 + 0.05·max(0, vix−20)) + 0.01·s·slope + noise` in data the size of the real set. Over 10 seeds at 4 noise levels, how often does PySR recover the structure?

**Person 2**
1. Create `src/evaluate/placebo.py`: shuffle the surprise across events (keeping each event's tickers together) and rerun any model. Expect OOS R² ≤ 0.
2. Start the extra baselines: `ElasticNet` (all features + all surprise × feature products) and `LightGBM` (depth 2–3, early stopping).

**Person 3**
1. Create `src/fetch_french.py`: the `12_Industry_Portfolios_daily` zip (value-weighted block, ÷100, −99.99 → NaN) and `F-F_Research_Data_Factors_daily` (market = Mkt-RF + RF).
2. Extend `fomc_dates.csv` back to **1994-02-04**, cross-checked against the SF Fed file.
3. `build_events.py`: left-join `mps_bp`, `mps_orth_bp`, `target_bp`, `path_bp`; add `zlb`. Keep every v0 column.
4. `build_returns.py --source french12`, with the same output columns.

## Week 10 (Nov 30 – Dec 6): present, then Stage 2 in full

**All:** present at the club.

**Person 3**
1. `build_features.py` for the French panel, plus the Stage 2 features (`term_10y3m, credit, ff, vol_60d, beta_2y, mkt_mom_20d`). Add `DGS3MO` to `FRED_SERIES`.
2. Create `src/build_panel.py`: events ⋈ returns ⋈ features → `panel.parquet`, plus `docs/DATA_CARD.md` (rows per year and per ticker, missingness, summary stats).
3. Create `src/splits.py`:
   - the dev folds from §1.4
   - `locked_test()` raises unless the env var `UNLOCK_TEST=1` is set, and appends user, time and git SHA to `results/test_unlock.log`
   - standardization stats computed inside the training fold only
4. `tests/test_panel.py`: 12 rows per event, no NaN in the main columns, no event in both train and val.
5. Make the pipeline produce an **identical file hash** when run twice.

**Person 2**
1. **Gate 2:** repeat Gate 0 on v1 (`mps_orth_bp` + French 12). Put v0 and v1 in one table.
2. Add the last baselines:
   - smooth-transition regression in VIX (`scipy.optimize.least_squares`)
   - a random forest or `econml` causal forest
   - a small MLP (16–32 units, early stopping)
   - the zero and historical-mean baselines
3. **Equal budget:** every model gets at most 8 hyperparameter settings scored on validation.

**Person 1**
1. Create `src/runner.py`: `python -m src.runner configs/<name>.yaml --folds dev --seeds 0-29`.
   - hash the config, loop folds × seeds, and write the predictions
   - append to `results/registry.csv` (hash, git SHA, time, versions, wall-clock)
   - skip runs that are already done (resumable)
2. Create the `configs/` YAMLs: `pysr_pooled.yaml`, `pysr_per_ticker.yaml`, `gplearn_pooled.yaml`, one per baseline.
3. Time one full PySR fit. If 30 seeds × 4 folds × the configs comes to more than ~48 laptop-hours, ask the leads or advisor for a shared machine.

## Weeks 11–12 (Dec 7–20): finals. Keep runs going overnight; light hands-on work

| Person 1 | Person 2 | Person 3 |
|---|---|---|
| Overnight: PySR pooled + per-ticker, 4 dev folds × 30 seeds, parsimony grid picked on val | Overnight: all baselines on dev folds | Robustness YAMLs: v0 data (Δ2y + SPDR), `mps_bp` raw, 10/17 industries, include unscheduled, ex-ZLB, `excess_5d` |
| Formula stability: for each term (`s·vix`, `s·slope`, `max(0,vix−c)`), the share of seeds whose chosen formula contains it | Placebos on the chosen PySR config (30 shuffles) | Paper "Data" section draft + timing figure (day −1 close → 2 pm statement → day 0 close) |

## Week 13 (Dec 21–28): finish dev work and freeze

**Person 1**
1. Create `src/evaluate/interpret.py`:
   - SymPy-simplified LaTeX formulas
   - response surfaces (surprise × VIX) for 4 sectors
   - a channel check: does the formula's sign agree with theory? (banks vs slope; tech vs path surprise; utilities as bond-like)
2. Write the paper "Method" section.

**Person 2**
1. A dev results table for every model, plus the significance tests on event-level losses.
2. Write the paper "Protocol" section.

**Person 3**
1. Run the robustness grid on dev.
2. Create the anonymized code archive skeleton. GECCO 2027 expects code with the submission.

**All, Mon Dec 28:** pick one final config per model family (best on dev validation), then `git tag freeze-v1`. From here on, only logged bug fixes.

## Dec 29 – Jan 4: open the test set once

1. One named person runs `UNLOCK_TEST=1 python -m src.runner configs/final/*.yaml --test`. Train on 1994–2015 and predict 2016+. GP gets 30 seeds.
2. Person 2 runs the significance tests and placebos on the test predictions.
3. Person 1 runs interpretation on the final formulas.
4. Person 3 runs the illustrative backtest: long the top 3 / short the bottom 3 predicted sectors on Fed days, with 5 and 10 bp costs. Frame it as "if you knew the surprise", since there's no intraday data.
- **Rule:** whatever the result, report it. No new configs after seeing the test.

## Jan 5 – Jan 25: write and submit

| Dates | Person 1 | Person 2 | Person 3 | Leads |
|---|---|---|---|---|
| Jan 5–10 | Results figures, formula table | Results tables, significance text | Data card appendix, reproducibility check from a clean clone | Intro, related work, threats to validity, AI-use disclosure paragraph |
| **Jan 11** | — | — | — | **Full draft to advisor** |
| Jan 12–17 | Revise method + results | Revise protocol | Anonymize code + supplement | Re-run novelty searches; edit |
| **Jan 18** | | | | **Submit 200-word abstract (mandatory)** |
| Jan 19–24 | Final checks | Final checks | Final checks | 8-page limit, anonymization, checklist below |
| **Jan 25** | | | | **Submit paper + code** |

**Pre-submission checklist**
- [ ] Main shock is the published SF Fed series; Δ2y appears only as a robustness check
- [ ] No-look-ahead tests pass; data card in the supplement
- [ ] Test set opened exactly once (`test_unlock.log` has 1 entry)
- [ ] All baselines given an equal tuning budget
- [ ] 30 GP seeds, reported as medians and spread; no "best seed" claims
- [ ] Clark-West / DM on event-level losses, Holm-adjusted
- [ ] Placebos reported
- [ ] Number of configs tried reported (from `registry.csv`)
- [ ] Formulas simplified, with sizes
- [ ] Backtest framed as illustrative, with costs
- [ ] AI-tool use disclosed in Methods (required by GECCO 2027)
- [ ] No names, no SFIC or UCSD, self-citations in third person, code archive anonymized

## After submission

| When | What |
|---|---|
| Mar 19 | GECCO decision. If accepted: camera-ready Apr 9, register by May 10, plan travel to Kraków (Jul 12–16, in person only) |
| Mar 21 | EvoStar late-breaking abstract with **different** content (e.g. the SFIC portfolio case study, or CPI/payroll shocks) |
| Late Mar | If rejected or downgraded to a poster: a GECCO workshop version (Symbolic Regression / ECXAI) using the reviews |
| ~Mar 15 | IEEE CEC 2027 deadline falls **before** the GECCO decision, so CEC can't serve as a fallback for the same paper. Submit there only with clearly different work (e.g. CPI/payroll shocks). **Never submit the same paper to two venues at once** |
| Later | Journal version (GPEM / Applied Soft Computing); public package with a Zenodo DOI |

---

## Pivot rules
- **Gate 0 fails:** stop and fix the data first. Nothing built on a broken pipeline counts.
- **GP loses to linear (Gate 1):** the club gets an honest "GP overfits shocks" result, and the paper becomes a rigorous benchmark (poster or workshop target).
- **PySR can't recover the planted formula:** that becomes a finding about the limits of GP at this sample size.
- **v1 data blocked (Gate 2):** write the paper on v0, disclose the proxy shock, and aim for a poster.
- **Draft not ready Jan 11:** submit a 4-page poster-only paper on Jan 25.
