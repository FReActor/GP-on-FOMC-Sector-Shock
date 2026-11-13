# Stage 2 handoff: state, methods, and the exact remaining procedure

This is the single document a new agent or teammate needs to pick up Stage 2 and finish it.
Read it fully, then `docs/STAGE2.md` (design and decisions) and `results/v1/STAGE2_REPORT.md`
(latest numbers). Everything below refers to branch `stage2-v1`.

---

## 0. Ground rules (do not break these)

1. **The 2016+ test set is locked.** Never run `UNLOCK_TEST=1 python -m src.v1.final` and never call
   `splits.locked_test()` until the team has frozen the model list. It is a one-time evaluation;
   every unlock is logged. All work below uses the dev folds only.
2. **No AI attribution lines** in commits or PRs (no "Co-Authored-By: ..."). No "Person 3" wording in commits.
3. **Ask the user before every `git push`.** Commit locally freely.
4. **Explain in plain language before changing code** when results are bad or the direction is unclear.
5. **Never kill processes you did not start.** The user runs their own jobs on this machine (e.g.
   `level2.py`, `features_v3.py`, `src/experiments.py ksweep`). Stop only your own process trees, by PID.
6. Every new model or rule must be **fixed before it is run** and **reported even if it loses**.
   Record each addition as a numbered deviation in `docs/STAGE2.md` §6. The paper reports how many models were tried.
7. Windows machine: 14 cores, 16 GB RAM (often only 2–5 GB free). Each PySR (Julia) worker needs
   ~0.7–1.2 GB. Use `--julia-jobs 4` or `5`. Run long jobs in the background and log to a file.
   Plug the laptop in (on battery, Windows throttles the CPU to ~68%).

---

## 1. The project in one paragraph

SFIC Fall 2026 research project, targeting a **GECCO 2027 paper** (abstract Jan 18, paper Jan 25,
2027; main track is the goal, poster or SymReg workshop is the fallback). Question: **can genetic
programming / symbolic regression (GP/SR) predict how US industries react on the day of an
unexpected Federal Reserve policy surprise, and does it find structure beyond the textbook model
(the surprise hurts high-market-beta industries more)?** It is an FOMC event study (Bernanke-Kuttner 2005 style):
one row per (FOMC meeting × industry); the input is the high-frequency rate surprise plus each
industry's characteristics known before the meeting; the target is the industry's return that day.

---

## 2. Data pipeline (all built, all working)

```
python -m src.v1.fetch    # SF Fed USMPD (Bauer-Swanson), Ken French 12/49 industries daily + monthly BE/ME and size, FRED macro
python -m src.v1.build    # panels -> data/processed/v1/<universe>/panel.csv ; docs/DATA_CARD.md
python -m src.v1.gate     # sanity gate -> results/v1/gate2.md (49/49 industries react negatively to hawkish surprises)
```

- **Shock `mps_bp`**: first principal component of ED1–ED4 Eurodollar-futures changes in the USMPD
  "Monetary Events" window, scaled so ED4 moves one-for-one; positive = hawkish. Correlation with
  the official Bauer-Swanson MPS: 0.975. Robustness: official `MPS_ORTH` (variant `orth`).
- **Universe**: Ken French 49 industries (`ff49`, main), 12 industries (`ff12`, robustness). Daily value-weighted returns.
- **Targets**: `ret_1d` (raw event-day return) and `excess_1d` (minus the CRSP market return).
  Winsorized at 0.5/99.5% on training rows only.
- **Features (as of the day before)**: `beta_mkt`, `beta_2y` (rolling betas), `fomc_beta_raw/exc`
  (the industry's shrunk slope on past FOMC surprises, earlier events only, shrink n/(n+20)),
  `bm`, `size` (Fama-French timing), `mom_20d`, `vol_60d`, `vix`, `term_10y3m`, `slope_2s10s`,
  `credit`, `ff`, `mkt_mom_20d`, flags `zlb` (zero lower bound) and `hawk` (s > 0). Also
  `hf_sp500_pct` (30-minute S&P move, used only by the `*_jk` models).
- Scheduled meetings only (variant `unsched` adds unscheduled ones). About 280 events.

## 3. Evaluation protocol (fixed)

- **Dev folds** (`splits.dev_folds`): expanding window; train 1994→2003 / 2006 / 2009 / 2012,
  validate the following 3 years (2004–2015 total). **Inner split** for tuning = last 3 training years.
- **Metrics** (`src/v1/report.py`): `r2_oos` vs each industry's training mean (Campbell-Thompson);
  `xs_r2` (event-demeaned, cross-sectional only); `rank_ic` (Spearman across industries per event).
- **Significance** (one loss per event, because industries on a Fed day share one shock):
  Clark-West vs the mean; paired one-sided t-tests vs predicting 0 (`zero_p`) and vs `linear_fomc` (`vs_lin_p`).
- **Placebo**: variant `placeboK` permutes the shock across events (`rng(1000+K)`); a real effect must beat the placebo distribution.
- Stochastic models report the **seed-averaged forecast**.
- **Inputs z-scored on training rows and clipped at ±3 SD** (`models.CLIP`).

## 4. Code map

| file | role |
|---|---|
| `src/v1/settings.py` | paths, universes, targets, seeds, shrinkage constants |
| `src/v1/models.py` | every model: `fit(name, train, ...) -> (predict, info)`; `Design` does z-scoring |
| `src/v1/run.py` | runs models on the dev folds in parallel; caches each (model, fold, seed) under `data/processed/v1/cache/<universe>_<target>_<variant>/<model>/<config-hash>/`; writes `results/v1/<universe>_<target>[_<variant>]/preds/<model>.csv` and `runs.csv` |
| `src/v1/report.py` | metrics, placebo table, combos, formulas, figures → `results/v1/STAGE2_REPORT.md`, `dev_summary.csv` |
| `src/v1/final.py` | one-time locked test (do not run) |
| `tests/test_v1.py` | 25 tests; whole suite 54 tests, `pytest -q` ~1.5 min |
| `scripts/overnight_dev.sh` | the last overnight run script (template for new runs) |

Cache rule: the hash in `run.config_hash` covers every setting that affects a result, so a
rerun skips finished work and a changed setting never reuses stale results. **If you change a
model's behavior, add the new setting to its hash** (only for the models it affects, or you
invalidate other caches).

## 5. Models (all implemented)

Sensitivity form used by most SR models: **y = base + s10 · g(x)**, where s10 = surprise / 10 bp
and g is the industry's response "per 10 bp". Implemented in PySR by passing s10 as sample
`weights` and a custom Julia loss `SENS_LOSS` scoring (g·w − y)². This is the paper's main methodological device.

| model | what it is |
|---|---|
| `zero`, `mean` | predict 0; industry training mean (the r2 benchmark) |
| `linear`, `linear_pooled` | a + b·s per industry / pooled |
| **`linear_beta`** | s, s·beta_mkt, beta_mkt — **the champion to beat** |
| `linear_fomc` | s, s·fomc_beta, fomc_beta |
| `linear_char` | s × {beta_mkt, fomc_beta, bm, size} + main effects |
| `linear_interact_pooled` | s × every characteristic and state variable |
| `linear_jk` | linear_beta + s·cbi, where cbi = 1 if rates and stocks moved the same way in the 30-min window (Jarociński-Karadi 2020 "information shock"); uses same-day info → reported separately |
| `enet`, `hgb`, `rf`, `mlp` | ElasticNet; gradient boosting; random forest; NN 32-16-8 ×5 (Gu-Kelly-Xiu 2020 black boxes) |
| `gp_pooled` | gplearn GP, free formula (5 seeds) |
| `gp_sens` | gplearn, sensitivity form, Keijzer linear scaling (custom fitness `_ls_mse`) |
| `pysr_pooled` | PySR, free formula |
| `pysr_sens` | PySR sensitivity model, size by 1-SE rule on the inner split |
| `pysr_mdl` | sensitivity model, one search on the full fold, formula by minimum description length (N = events) |
| `pysr_bag` | 8 event-bootstrap bags, each bag's formula picked on its out-of-bag events, averaged; 40 iterations per bag (`BAG_ITERS`) |
| `pysr_anchor` | base = linear_beta, bagged PySR search for the **residual** sensitivity; "g = 0" always a candidate |
| `pysr_mv` | pysr_anchor + per-industry sensitivity offsets on out-of-bag residuals, shrunk n/(n+20). **Derived from the pysr_anchor cache** (no extra search) |
| `pysr_jk` | pysr_anchor with `cbi` as an input |
| `pysr_enet` | FEAT-style: every Pareto-front formula s10·g_k becomes a feature; ElasticNet weights them |
| `combo_linear`, `combo_sr`, `combo_all` | equal-weight forecast combinations fixed in advance (Rapach-Strauss-Zhou 2010), built in `report.py` |

Abandoned (documented): PySR `TemplateExpressionSpec` with 49 per-industry parameters (did not finish 40 iterations in 30 min; needs `loss_function_expression`, not `loss_function`).

## 6. Results so far (dev 2004–2015, ff49; full tables in `results/v1/STAGE2_REPORT.md`)

| model | raw `ret_1d` r2_oos | excess `excess_1d` r2_oos |
|---|---|---|
| zero | −0.038 | +0.008 |
| **linear_jk** | **+0.099** | +0.009 |
| **linear_beta** | +0.079 (zero_p 0.018) | **+0.018** (zero_p 0.029) |
| linear_char / combo_linear | +0.073 / +0.073 | +0.017 / +0.018 (zero_p 0.014) |
| pysr_sens = pysr_mdl | +0.061 (constant sensitivity, ≈ −0.65 %/10 bp) | +0.004 |
| pysr_jk | +0.024 (fold 1 +0.145) | +0.015 |
| pysr_mv / pysr_anchor | −0.171 / −0.183 (fold 1 **+0.163** / +0.159) | +0.013 / +0.009 (fold 1 +0.043) |
| pysr_bag | −0.479 | +0.001 |
| pysr_enet | −0.538 | −0.017 |
| rf / mlp / hgb / enet | −0.12 / −0.38 / −0.23 / −0.12 | +0.012 / +0.008 / −0.010 / +0.011 |

**Interpretation:**
- The surprise really predicts Fed-day industry returns, and simple beta-based models win.
- Every single-search SR model picks a constant sensitivity: it rediscovers the average Fed effect.
- The anchored / bagged SR models find **real extra structure** (sensitivity rises with VIX, shifts at the zero lower bound). They beat beta in fold 1 (2004–06) but **blow up in fold 2 (2007–09)** and fold 3 (2010–12). Two causes:
  1. **Extreme VIX in 2008:** clipping inputs at ±3 SD is not enough when g multiplies VIX and a large surprise.
  2. **`zlb` never varies before 2008:** it is constant in folds 1–2 training data, so any formula term in `zlb` is untested noise that fires when rates hit zero.
- Black boxes fail. That points to the data limits, not to GP in particular.

**Placebo status:**
- Placebos 0–4 done for linear_beta, linear_fomc, linear_char, linear_jk, pysr_anchor, pysr_mv and pysr_mdl.
- Exception: **placebo4 excess_1d pysr_mv and pysr_mdl failed** (BrokenProcessPool, a worker died, likely low RAM). Rerun them.
- Older placebos 0–19 exist for the original models.

**`gp_sens` main run:** started then stopped (the run was stopped from the app at about 02:20). Not finished.

## 7. Uncommitted changes on `stage2-v1` (commit these first)

`src/v1/models.py`, `run.py`, `report.py`, `tests/test_v1.py`, `docs/STAGE2.md` (deviation 7),
`README.md`, `docs/HANDOFF.md`, `scripts/overnight_dev.sh`, regenerated `results/v1/*` (preds are git-ignored).

```
pytest -q                                   # must show 54 passed
git add -A && git commit -m "Stage 2: hybrid SR models, new baselines, forecast combinations, dev results"
# ask the user before: git push origin stage2-v1
```

---

## 8. Remaining procedure (do in this order)

### Step A: finish the interrupted run (cheap)
```
python -m src.v1.run --variant placebo4 --targets excess_1d --models pysr_anchor,pysr_mv,pysr_mdl --seeds 1 --julia-jobs 4
python -m src.v1.run --models gp_sens --seeds 3 --jobs 8
python -m src.v1.report
```

### Step B: extrapolation guards (approved by the user; apply to all SR models)
Implement in `src/v1/models.py`. Pre-specify, document as deviation 8.
1. **Drop dead inputs.** Before each search, remove from the input list any column whose training-fold
   standard deviation is ~0. Use the raw column, e.g. `zlb` before 2008, `cbi` if constant. The fold's
   inputs are recorded in `info["inputs"]`. Implement once (e.g. `live_inputs(x_train, inputs)`) and use it in `fit_pysr`,
   `fit_pysr_mdl`, `fit_pysr_bag`, `fit_pysr_enet`, `fit_gp_sens`, `fit_gp`.
2. **Clamp the sensitivity.** After fitting g (each bag member separately), record g's 1st–99th
   percentile on the rows it was trained on. At prediction, clip g to that range. For anchored models,
   clip the residual sensitivity g the same way. Add a test: a formula of `vix` predicts within the training range when `vix = 1000`.
3. Add `"guards": {"dead_inputs": True, "clamp_g": [0.01, 0.99]}` to `config_hash` for SR models (new caches).
4. Rerun, ff49, both targets:
```
python -m src.v1.run --models pysr_sens,pysr_mdl --seeds 5 --julia-jobs 5
python -m src.v1.run --models pysr_anchor,pysr_mv,pysr_bag --seeds 2 --julia-jobs 5
python -m src.v1.run --models pysr_jk --seeds 1 --julia-jobs 5
python -m src.v1.run --models gp_sens --seeds 3 --jobs 8
for k in 0 1 2 3 4; do python -m src.v1.run --variant placebo$k --models pysr_anchor,pysr_mv,pysr_mdl --seeds 1 --julia-jobs 4; done
python -m src.v1.report
```
Time: ~5 h on a free CPU (one full PySR search ≈ 1–7 min depending on load; a bagged task = 8 searches of 40 iterations).

### Step C: better tuning and a "can't lose much" blend (if B does not beat linear_beta overall)
1. **Blocked cross-validation for formula selection.** Replace the "last 3 years" inner split for SR models with K = 5 contiguous year-blocks over the whole training fold. For each block: fit on the others, score on the block; average event-level losses. Choose formula size (or the bag-selection rule) on the pooled CV loss. Then refit on the full fold. This raises tuning events from ~24 to all training events.
2. **Shrink toward the anchor:** forecast = linear_beta + λ · s10 · g(x), with λ ∈ {0, 0.25, 0.5, 0.75, 1} chosen by the same blocked CV. New model `pysr_shrink`. λ = 0 reproduces linear_beta, so it cannot lose much beyond CV noise.
3. **Stacking:** non-negative weights on {linear_beta, linear_fomc, linear_jk, guarded pysr_anchor} chosen by blocked CV, summing to 1. New combo `stack_cv` computed inside the fold, never on validation rows.
4. Optional **regime SR**: give the search a pre-specified calm/stress flag (VIX above its training-fold 80th percentile) as an input, so it can learn separate sensitivities.

### Step D: more cross-sectional data (free, highest payoff for the paper)
Ken French daily portfolio sets (same site and parser as the industries):
25 size×B/M, 25 size×momentum, 25 size×operating profitability, 25 size×investment, 10 momentum deciles.
- Add each as a universe in `settings.UNIVERSES`. Build characteristics: betas, fomc_beta and mom/vol work as-is. bm and size come from the sort, or set them to missing and z-score to 0.
- Run the champion models plus the guarded SR models on a pooled "ff_all" universe (about 150 portfolios × 280 events).
- Report per universe and pooled.

### Step E: second central bank (the transfer test; strongest main-track argument)
- ECB surprises: the Euro Area Monetary Policy Event-Study Database (Altavilla et al. 2019, JME). It has free high-frequency OIS/Bund changes in the press-release and press-conference windows. Build an `mps` analogue with PC1, scaled like the Fed one.
- Ken French daily European portfolios (Developed-markets "Europe" 25 size×B/M and size×momentum).
- **Protocol:** formulas are found on Fed data only, then evaluated unchanged on ECB events (re-estimate only the linear scale a, b). Fair comparison: linear_beta re-estimated the same way. This "discovered on the Fed, transfers to the ECB" result is the main-track headline.

### Step F: freeze and run the locked test (team decision)
1. With the team, freeze the model list in `src/v1/final.py` (`FROZEN`) and the seeds (`FINAL_SEEDS`, ≥ 10).
2. `UNLOCK_TEST=1 python -m src.v1.final`, once. Archive the log.

### Step G: final statistics and robustness
- 10+ seeds and 20 placebos for the headline models.
- Diebold-Mariano tests (event-level, HAC) vs linear_beta.
- Correct for the number of models tried: Model Confidence Set (Hansen-Lunde-Nason 2011), or report a Holm-adjusted p.
- Robustness table: `--variant orth`, `--variant unsched`, `--universe ff12`, 5-day target (`ret_5d`, `excess_5d`).
- Interpretation: the fold-4 formulas, the share of bags using each input (`report.sr_section`), and the economic reading (VIX: risk-taking channel; ZLB: forward-guidance era).

### Step H: paper (GECCO, 8 pages, ACM format)
Track: Real-World Applications, or Genetic Programming if the method contributions are strong.
1. Intro: the question and the one-line result.
2. Related work: SR/GP in finance; event studies (Bernanke-Kuttner 2005, Bauer-Swanson 2023, Jarociński-Karadi 2020); SR methods (PySR, Cranmer 2023; Keijzer 2003; FEAT; MDL selection, Bartlett et al. 2024; bagging GP).
3. Method: sensitivity-form SR via weighted custom loss; anchored residual SR; bagging with out-of-bag selection; extrapolation guards; blocked CV and shrinkage.
4. Setup: data, folds, locked test, placebos.
5. Results: main table, fold breakdown, placebo figure, formulas, transfer to the ECB (if done).
6. Discussion: when GP adds value (cross-sectional heterogeneity, regimes) and when it does not (tiny signal, regime breaks).
7. Limitations and number of models tried.

**Decision rule for the venue:**
- **Main track:** guarded or shrunk SR beats linear_beta on the dev folds and on the locked test, passes placebos, and transfers to the ECB.
- **Poster or SymReg workshop:** SR ties linear_beta and beats the black boxes.
- **Workshop "negative result" paper:** SR loses. Frame it as "why GP fails on regime-shifting financial events, and the guards that fix it."

## 9. Known pitfalls

- PySR custom loss: `loss_function` evaluates the tree (`eval_tree_array`). With `TemplateExpressionSpec` parameters, use `loss_function_expression` instead.
- `ProcessPoolExecutor` children on Windows survive a stopped parent; stop the whole tree with `taskkill //PID <pid> //T //F`, only for your own run.
- Bash reads scripts incrementally; do not edit a running `.sh` file. Copy it, edit the copy, and restart.
- gplearn `max_samples < 1` zeroes some sample weights, which breaks the weights-as-surprise trick; `gp_sens` sets `max_samples = 1.0`.
- `report.formulas` needs a `program` column only for gplearn models; `gp_sens` stores a readable `formula` instead.
- Seeds: the CLI `--seeds N` means seeds 0..N−1. Placebos use `--seeds 1`.
- `results/v1/**/preds/` is git-ignored; caches live in `data/processed/v1/cache/` (also ignored).
