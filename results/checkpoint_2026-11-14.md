# Sat 11-14 results checkpoint (Track C, GP v1)

Split A, test 2010-2015, metric `r2_oos` (vs each ticker's train mean). Pooled GP v1 (Charles) and
per-ticker GP v1 (Abhi, branch `gp-scope-interpret`) are both in `results/`; `python -m src.compare`
combines them into `gp_v1` in `results/summary.csv`. GP numbers are medians over 10 seeds.

## Test r2_oos by model

| horizon | ticker | zero | linear | linear_interact | gp_v0 | gp_v1 |
|---|---|---|---|---|---|---|
| 1d | POOLED | +0.080 | +0.076 | +0.056 | +0.080 | +0.074 |
| 1d | XLB | -0.008 | -0.023 | -0.211 | -0.226 | -0.428 |
| 1d | XLE | +0.005 | -0.070 | -0.092 | -0.407 | -0.403 |
| 1d | XLF | +0.027 | -0.007 | +0.247 | +0.047 | +0.059 |
| 1d | XLI | +0.043 | +0.044 | +0.095 | +0.053 | +0.071 |
| 1d | XLK | +0.103 | +0.009 | -1.141 | +0.052 | +0.065 |
| 1d | XLP | +0.147 | +0.001 | +0.186 | +0.115 | +0.140 |
| 1d | XLU | +0.175 | -0.048 | -0.098 | +0.104 | +0.072 |
| 1d | XLV | +0.037 | -0.011 | -0.038 | -0.006 | +0.039 |
| 1d | XLY | +0.249 | -0.036 | -0.211 | +0.210 | +0.186 |
| 5d | POOLED | +0.010 | +0.010 | +0.000 | +0.004 | +0.008 |
| 5d | XLB | -0.004 | -0.003 | -0.129 | -0.005 | -0.360 |
| 5d | XLE | +0.003 | -0.007 | -0.118 | -0.018 | -0.013 |
| 5d | XLF | +0.002 | +0.057 | -0.152 | -0.178 | -0.069 |
| 5d | XLI | +0.001 | -0.069 | -0.214 | -0.001 | -0.001 |
| 5d | XLK | +0.194 | +0.017 | -0.086 | +0.123 | +0.235 |
| 5d | XLP | -0.004 | -0.003 | -0.033 | -0.003 | -0.124 |
| 5d | XLU | +0.000 | -0.009 | -0.221 | +0.005 | -0.081 |
| 5d | XLV | +0.017 | -0.060 | -0.166 | -0.040 | -0.052 |
| 5d | XLY | -0.007 | +0.043 | -0.071 | -0.103 | -0.192 |

## Per-ticker cells that beat the zero forecast (out of 9 per horizon)

| model | 1d | 5d | median (model - zero), 1d | median (model - zero), 5d |
|---|---|---|---|---|
| linear | 1 | 4 | -0.075 | -0.009 |
| linear_interact | 3 | 0 | -0.097 | -0.154 |
| gp_v0 | 2 | 2 | -0.043 | -0.021 |
| gp_v1 | 3 | 1 | -0.038 | -0.071 |

## Reading

- Pooled: NO-GO stands (`results/go_no_go.md`). GP v1 +0.074 < linear +0.076 < zero +0.080 at 1d;
  at 5d every model is within 0.01 of zero.
- Per ticker: GP v1 beats zero in 4 of 18 cells (XLF 1d, XLI 1d, XLV 1d, XLK 5d), about what noise
  gives. Its median gap to zero is negative at both horizons, and it blows up on XLB (-0.43 / -0.36).
- Adding the pre-event features does not help either model class out of sample: linear_interact is
  worse than linear, and GP v1 is not reliably better than GP v0.
- Track C is kept, as a negative result: "features don't rescue it". The positive per-ticker
  numbers (e.g. XLY 1d +0.19) are the zero-forecast artifact from slide 10, not skill.
- Feature usage and example formulas: `results/gp_interpretation.md` (Abhi).
