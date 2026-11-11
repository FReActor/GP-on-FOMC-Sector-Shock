# Stage 1 Plan (final version, revised 2026-11-11)

**Final presentation: weekend of Sat/Sun 11-21/22.** This version replaces the earlier task table. With 10 days left, there are only three principles:

1. **The critical path is on Charles** (merging code, linear baseline, main GP runs, results, figures). Abhi (Person 3) has finished the features and joins the GP work, sharing the v1 compute and interpretation; Snata (Person 2) does slides only. No teammate task blocks the critical path: if something is late or missing, the presentation still works.
2. **Results freeze on Sun 11-15.** After that, no new models: only figures, slides and rehearsal.
3. **A negative result is still a result.** "The surprise alone cannot predict sector reactions; GP overfits, and so does the linear model" is a complete conclusion we can present. The features are done, so GP v1 is back on the main track, but if v1 doesn't produce results, A + B can carry the presentation on their own.

---

## Scope

| | Content | Status | Can it be cut? |
|---|---|---|---|
| **Track A** | GP v0 vs linear vs zero forecast, `surprise_bp` only, Option A split (train 1999–2009, test 2010–2015), all horizons, metric `r2_oos` | GP v0 done; linear done (Charles) | No, this is the core of the presentation |
| **Track B** | The formulas GP learned (sympy) + why this is overfitting (the XLE 0.022/surprise_bp² example) | `results/gp_formulas.csv` already exists | No, it's the best slide we have |
| **Track C** | GP v1: add the pre-event features from `features.csv`, compared with linear (same features) | Code written; features are on main (PR #3), each person generates them locally with build_features.py | Only cut if there are no results by Saturday, see the rules below |

**Track C rules:**
- Abhi's `build_features.py` was merged into main via PR #3. `features.csv` is not in git: each person generates it locally with `python src/build_features.py`, and both use the same `config.py`.
- GP v1 runs on two machines, **starting Thu night**: Charles runs pooled (all tickers together); Abhi runs per-ticker (one model per ticker). Same seeds and generation settings, written in config, sized to finish overnight.
- Results go into `results/` **by Sat 11-14 noon**. If Abhi's per-ticker run isn't finished, we skip it; pooled is enough for the main conclusion.
- If neither side has usable results by Saturday: cut C, present only A + B, and move "adding pre-event features" to Future work.

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

## Day-by-day schedule

| Date | Charles (critical path) | Abhi (Person 3) | Snata (Person 2) |
|---|---|---|---|
| **Wed 11-11** | Merge `charles` into main; write `src/models/linear.py` → `results/linear.csv`; run `compare.py`; message the team | `features-data-report` is already merged into main (PR #3), no new PR needed | Read the lit review, start the Motivation slide |
| **Thu 11-12** | Generate features.csv locally, write linear_interact; put the v1 settings in config; **start pooled GP v1 Thu night** | Pull the latest main; first rebuild the data locally (`fetch_data` → `build_events` → `build_returns` → `build_features`), since `data/` is not in git; do a small test run of `gp.py` on your machine; **start per-ticker GP v1 Thu night** | Motivation + literature, 2 slides (draft) |
| **Fri 11-13** | Check pooled v1; start the plotting script `src/plots.py` | Check per-ticker v1; start going through the v1 formulas: which features GP uses most | Keep going |
| **Sat 11-14** | **v1 results in `results/` by noon**, run `compare.py`; (if C is cut) robustness checks, e.g. dropping the 2008 / 2020 events | **Per-ticker results in `results/` by noon**; write the feature-usage frequency table | — |
| **Sun 11-15** | **Freeze results + set the storyline** (the old go/no-go). Create the shared Google Slides with slide placeholders; send `results/` and the figures to the team | Start the GP v1 slides | Start the Baseline slide once linear.csv arrives |
| **Mon 11-16** | Write the Results slides (GP vs linear vs zero, formula slide) | GP v1 slides + Data & Method slides | Baseline slide |
| **Tue 11-17** | Finish Results slides; write Conclusion / Future work | **Data & Method slides due** | **Motivation + literature slides due** |
| **Wed 11-18** | Put all slides together, unify the formatting | **GP v1 slides due** | **Baseline slide due** |
| **Thu 11-19** | **First rehearsal** (everyone, online, timed); note changes | Join rehearsal | Join rehearsal |
| **Fri 11-20** | Revise slides; second rehearsal (optional); send to leads | Revise own slides | Revise own slides |
| **Sat/Sun 11-21/22** | **Present** | Present own part | Present own part |

**Rule:** if a teammate's slides aren't in by the deadline, Charles makes a short version from existing material and doesn't chase further. Everyone presents only the part they did.

---

## Teammate tasks (small, independent, non-blocking)

**Abhi (Person 3, joining GP)**
1. `build_features.py` merged into main (PR #3): done.
2. Run **per-ticker GP v1** on your own machine (same `gp.py` and config, only the pooled/per-ticker switch changes), output to `results/gp_v1_per_ticker.csv`. First rebuild the data locally (`fetch_data` → `build_events` → `build_returns` → `build_features`), since `data/` is not in git. Start Thu night, **due Sat 11-14 noon**. Non-blocking: if it doesn't finish, we use Charles's pooled results only.
3. Interpret v1: which features GP uses most, 2–3 example formulas, and whether it beats surprise-only. **Due Sun 11-15.**
4. Slides: Data & Method, 1–2 slides (due Tue 11-17); GP v1 results, 1–2 slides (due Wed 11-18).

**Snata (Person 2)**
1. Motivation + literature, 2 slides (what the question is, why GP, Chen & Yeh 2000, Stasinakis 2013, Neely 1997 positive / Allen & Karjalainen 1999 negative). Material: the lit review doc. **Due Tue 11-17.**
2. Baseline slide (linear model method + results table + one chart), using the `results/linear.csv` Charles sends on Sun 11-15. The Colab work can be presented as "first exploration". **Due Wed 11-18.**

---

## Slide outline (about 12 slides, prepare for 12–15 minutes)

| # | Slide | Made by | Presented by |
|---|---|---|---|
| 1 | Title + members | Charles | Charles |
| 2 | Motivation: how do Fed rate surprises hit each sector? Can we predict it? | Snata | Snata |
| 3 | Literature: GP results in finance are mixed | Snata | Snata |
| 4 | Data: 222 FOMC events, 11 sector ETFs, 1999–2025, pre-event features | Abhi | Abhi |
| 5 | Method: surprise definition, excess returns, train/test split, r2_oos | Abhi | Abhi |
| 6 | Models: zero forecast / linear / GP (what GP is, one tree diagram) | Charles | Charles |
| 7 | Baseline results: linear r2_oos | Snata | Snata |
| 8 | Main result: GP vs linear vs zero, by horizon (bar chart) | Charles | Charles |
| 9 | What GP learned: example formulas + why it's overfitting | Charles | Charles |
| 10 | A trap: the r2_oos of +0.08 is an artifact of the benchmark | Charles | Charles |
| 11 | GP v1: results with pre-event features + the features GP uses most | Abhi | Abhi |
| 12 | Conclusion + limitations + Future work (more GP settings, intraday data, quant portfolio application) | Charles | Charles |

**Core conclusion (already supported by Tracks A/B):** using only the rate surprise, sector excess returns are essentially unpredictable out of sample. The formulas GP finds look good on the training set and get worse on the test set, which is overfitting; the linear model shows no stable signal either. This means we need more information (features) or a better surprise measure, and that GP needs extra care in small-sample event studies.

---

## To confirm

- Presentation length and format (how many minutes, whether everyone presents): ask the leads.

---

## What carries into Stage 2
`config.py`, `data.py`, the result formats, `linear.py`, `gp.py`, `compare.py`, `placebo.py`, `stats.py` and the tests all carry over. Stage 2 adds the official SF Fed surprise and Ken French industries in the same columns, swaps or adds PySR, and runs on the locked 2016+ test set.
