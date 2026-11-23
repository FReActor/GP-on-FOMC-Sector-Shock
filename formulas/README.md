# GP formulas

Every formula the GP evolved in Stage 1, one row per run (model x scope x ticker x horizon x seed). Generated from `results/` by `python -m src.export_formulas`; do not edit by hand.

- [GP1_surprise_only.md](GP1_surprise_only.md): GP1, input `surprise_bp` only
- [GP2_features.md](GP2_features.md): GP2, `surprise_bp` + `vix`, `slope_2s10s`, `beta_1y`, `mom_20d`
- [gp_formulas.csv](gp_formulas.csv): all runs, with the raw gplearn `program`, the sympy `formula`, its `latex`, `r2_train` (1999-2009), `r2_test` (2010-2015), `hit_rate`

Pooled = one formula for all sectors; per-ticker = one per sector ETF. 10 seeds per cell; parsimony picked per run on the 2007-2009 validation years.

| model | scope | horizon | runs | constant | median length | median R² train | median R² test |
|---|---|---|---:|---:|---:|---:|---:|
| GP1 | per-ticker | 1d | 90 | 74% | 2 | -0.004 | +0.046 |
| GP1 | per-ticker | 5d | 90 | 67% | 2 | -0.001 | -0.007 |
| GP1 | pooled | 1d | 10 | 100% | 1 | -0.034 | +0.080 |
| GP1 | pooled | 5d | 10 | 100% | 1 | -0.009 | +0.004 |
| GP2 | per-ticker | 1d | 90 | 56% | 2 | -0.001 | +0.060 |
| GP2 | per-ticker | 5d | 90 | 18% | 4 | +0.013 | -0.082 |
| GP2 | pooled | 1d | 10 | 80% | 1 | -0.032 | +0.074 |
| GP2 | pooled | 5d | 10 | 50% | 2.5 | -0.007 | +0.008 |
