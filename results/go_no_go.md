# Go / no-go (pooled, excess_1d, test set)

Split A. Rule stated in docs/STAGE1_PLAN.md before looking: GO if pooled GP v1's median
r2_oos beats both `linear` and `linear_interact`, and at least 6 of 10 seeds beat `linear`.

| model | r2_oos |
|---|---|
| gp_v1 (median of 10 seeds) | +0.0738 |
| linear | +0.0761 |
| linear_interact | +0.0557 |
| zero forecast (reference) | +0.0805 |

Seeds beating linear: 1 / 10

**Decision: NO-GO**
