# Gate 2: does the Stage 2 data show the known pattern?

Dev years only (1994-2015), scheduled meetings, 176 events. Shock `mps_bp` (positive = hawkish). Returns in percent; slopes per +10 bp surprise; HC1 robust t-stats.

| regression | slope per 10bp | t | R2 | n |
|---|---|---|---|---|
| (a) S&P 500, high-frequency window (%) | -0.481 | -6.027 | +0.197 | 160 |
| (b) market, daily close-to-close (%) | -0.582 | -2.709 | +0.080 | 176 |
| (d) market daily (%), Stage 1 2y proxy | -0.132 | -0.649 | +0.006 | 176 |

**Gate 2: PASS** (rule: (a) and (b) negative with t < -2).

## By industry (daily, close day -1 to close day 0)

- raw return slope negative for 12 of 12 industries
- excess return (the prediction target) has |t| > 2 for 2 of 12 industries

| industry | raw slope/10bp | raw t | excess slope/10bp | excess t | excess R2 | proxy raw t |
|---|---|---|---|---|---|---|
| Money | -0.769 | -1.955 | -0.187 | -0.932 | +0.011 | -0.180 |
| Durbl | -0.767 | -3.938 | -0.186 | -1.703 | +0.016 | -0.769 |
| Other | -0.700 | -2.793 | -0.118 | -1.910 | +0.020 | -0.953 |
| Manuf | -0.690 | -3.072 | -0.108 | -2.147 | +0.019 | -0.879 |
| Telcm | -0.628 | -3.020 | -0.046 | -0.481 | +0.001 | -1.309 |
| Chems | -0.595 | -3.117 | -0.014 | -0.144 | +0.000 | -0.886 |
| BusEq | -0.589 | -2.521 | -0.008 | -0.074 | +0.000 | -0.823 |
| Enrgy | -0.575 | -2.782 | +0.006 | +0.046 | +0.000 | -0.325 |
| Shops | -0.571 | -3.129 | +0.011 | +0.114 | +0.000 | -1.180 |
| Utils | -0.439 | -2.511 | +0.143 | +1.038 | +0.008 | -1.044 |
| NoDur | -0.396 | -3.090 | +0.186 | +1.649 | +0.020 | -0.555 |
| Hlth | -0.389 | -1.985 | +0.193 | +2.211 | +0.025 | -0.616 |

## How to read this

- (a) vs (b): the market reacts strongly inside the 30-100 minute window; over the full day other news is added, so the daily slope is noisier. The models must work with daily data.
- `proxy raw t` uses Stage 1's 2y-yield surprise on the same events, for comparison.
- The excess-return columns are the honest preview: they show how much of each industry's reaction is *different from the market's*, which is what the models try to predict.
