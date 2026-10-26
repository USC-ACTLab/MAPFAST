# Headroom over always bcp2

1647 instances; unsolved runs count 300 s. The oracle always picks the fastest solver.

| Always bcp2 (min) | Oracle (min) | Savings (min) | Savings (%) | 95% bootstrap CI (%) | Instances with a saving | Instances holding 50% / 80% / 90% of it |
|---|---|---|---|---|---|---|
| 348.7 | 324.9 | 23.8 | 6.8% | 3.4%–11.0% | 276 | 4 / 11 / 19 |

## Whether bcp2 solves it

| bcp2 ... | Instances | Another solver faster | Savings (min) | Share |
|---|---|---|---|---|
| solves it | 1644 | 273 | 17.3 | 73% |
| times out | 3 | 3 | 6.5 | 27% |

## Winning solver

Instances on which each solver beats bcp2, what it saves there, and what the two-solver portfolio {bcp2, solver} would save with a perfect choice between the two.

| Solver | Wins | Savings (min) | Share | Median win (s) | {bcp2, solver} saves (min) |
|---|---|---|---|---|---|
| cbsh2_rtc_chbp | 103 | 13.3 | 56% | 0.13 | 18.2 |
| cbsh2_rtc | 48 | 6.9 | 29% | 0.02 | 9.4 |
| cbsh2_rtc_ch | 58 | 3.0 | 13% | 0.03 | 18.2 |
| cbsh2_rtc_bp | 66 | 0.6 | 2% | 0.03 | 6.6 |
| reloc | 1 | 0.0 | 0% | 0.01 | 0.0 |
| cbs | 0 | 0.0 | 0% | 0.00 | 0.0 |
| ecbs | 0 | 0.0 | 0% | 0.00 | 0.0 |

## Agent count

| Agents | Instances | Always bcp2 (min) | Savings (min) | Savings (% of bcp2) | Share |
|---|---|---|---|---|---|
| 10 | 142 | 5.2 | 0.2 | 3.1% | 1% |
| 20 | 172 | 1.4 | 0.0 | 0.8% | 0% |
| 30 | 139 | 4.4 | 0.0 | 0.1% | 0% |
| 40 | 147 | 15.8 | 1.9 | 11.9% | 8% |
| 50 | 123 | 13.6 | 0.0 | 0.2% | 0% |
| 60 | 114 | 6.5 | 0.1 | 1.5% | 0% |
| 70 | 101 | 9.1 | 0.0 | 0.4% | 0% |
| 80 | 106 | 18.0 | 0.1 | 0.8% | 1% |
| 90 | 95 | 15.9 | 0.2 | 1.0% | 1% |
| 100 | 93 | 13.9 | 0.1 | 0.8% | 0% |
| 120 | 96 | 29.0 | 0.1 | 0.4% | 1% |
| 140 | 89 | 34.9 | 0.5 | 1.6% | 2% |
| 160 | 70 | 38.8 | 1.6 | 4.1% | 7% |
| 180 | 52 | 29.3 | 1.7 | 5.9% | 7% |
| 200 | 50 | 46.9 | 1.4 | 3.0% | 6% |
| 250 | 32 | 28.0 | 5.6 | 19.9% | 23% |
| 300 | 18 | 20.1 | 2.7 | 13.2% | 11% |
| 350 | 4 | 8.1 | 1.2 | 14.2% | 5% |
| 400 | 3 | 7.5 | 4.5 | 59.6% | 19% |
| 450 | 1 | 2.4 | 1.8 | 77.0% | 8% |

## Maps (the 15 with the most savings)

| Map | Instances | Always bcp2 (min) | Savings (min) | Savings (% of bcp2) | Share |
|---|---|---|---|---|---|
| warehouse-20-40-10-2-2 | 75 | 13.2 | 9.0 | 67.9% | 38% |
| warehouse-10-20-10-2-2 | 78 | 10.9 | 5.1 | 46.9% | 22% |
| empty-48-48 | 61 | 8.2 | 3.2 | 39.7% | 14% |
| Berlin_1_256 | 73 | 12.3 | 2.1 | 17.2% | 9% |
| room-64-64-8 | 15 | 6.4 | 1.9 | 28.9% | 8% |
| Paris_1_256 | 96 | 19.0 | 1.3 | 6.7% | 5% |
| den520d | 83 | 21.5 | 0.5 | 2.4% | 2% |
| random-64-64-10 | 79 | 2.4 | 0.2 | 9.7% | 1% |
| empty-32-32 | 64 | 12.9 | 0.2 | 1.2% | 1% |
| maze-128-128-1 | 2 | 5.0 | 0.1 | 2.4% | 1% |
| random-32-32-10 | 55 | 7.8 | 0.0 | 0.4% | 0% |
| warehouse-10-20-10-2-1 | 71 | 9.0 | 0.0 | 0.2% | 0% |
| warehouse-20-40-10-2-1 | 68 | 10.0 | 0.0 | 0.1% | 0% |
| brc202d | 64 | 23.3 | 0.0 | 0.0% | 0% |
| empty-16-16 | 38 | 7.8 | 0.0 | 0.1% | 0% |

## Size of the win

| Saving per instance | Instances | Savings (min) | Share |
|---|---|---|---|
| ≤ 0.01 s | 67 | 0.0 | 0% |
| 0.01–0.1 s | 96 | 0.1 | 0% |
| 0.1–1 s | 65 | 0.4 | 2% |
| 1–10 s | 28 | 1.7 | 7% |
| 10–60 s | 13 | 4.7 | 20% |
| > 60 s | 7 | 16.9 | 71% |
