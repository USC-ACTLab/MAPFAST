# Headroom over always bcp2

16813 instances; unsolved runs count 300 s. The oracle always picks the fastest solver.

| Always bcp2 (min) | Oracle (min) | Savings (min) | Savings (%) | 95% bootstrap CI (%) | Instances with a saving | Instances holding 50% / 80% / 90% of it |
|---|---|---|---|---|---|---|
| 3224.4 | 2827.2 | 397.2 | 12.3% | 10.5%–14.4% | 3004 | 45 / 110 / 191 |

## Whether bcp2 solves it

| bcp2 ... | Instances | Another solver faster | Savings (min) | Share |
|---|---|---|---|---|
| solves it | 16775 | 2966 | 235.8 | 59% |
| times out | 38 | 38 | 161.3 | 41% |

## Winning solver

Instances on which each solver beats bcp2, what it saves there, and what the two-solver portfolio {bcp2, solver} would save with a perfect choice between the two.

| Solver | Wins | Savings (min) | Share | Median win (s) | {bcp2, solver} saves (min) |
|---|---|---|---|---|---|
| cbsh2_rtc_chbp | 1323 | 258.7 | 65% | 0.19 | 377.3 |
| cbsh2_rtc_bp | 714 | 69.1 | 17% | 0.04 | 230.1 |
| cbsh2_rtc | 446 | 58.9 | 15% | 0.02 | 230.9 |
| cbsh2_rtc_ch | 518 | 10.4 | 3% | 0.02 | 291.1 |
| reloc | 3 | 0.0 | 0% | 0.00 | 0.1 |
| cbs | 0 | 0.0 | 0% | 0.00 | 0.0 |
| ecbs | 0 | 0.0 | 0% | 0.00 | 4.7 |

## Agent count

| Agents | Instances | Always bcp2 (min) | Savings (min) | Savings (% of bcp2) | Share |
|---|---|---|---|---|---|
| 10 | 1615 | 47.1 | 21.8 | 46.2% | 5% |
| 20 | 1549 | 47.6 | 27.2 | 57.2% | 7% |
| 30 | 1496 | 87.1 | 13.5 | 15.5% | 3% |
| 40 | 1377 | 92.4 | 2.3 | 2.5% | 1% |
| 50 | 1277 | 81.0 | 0.7 | 0.9% | 0% |
| 60 | 1212 | 79.8 | 5.7 | 7.1% | 1% |
| 70 | 1162 | 128.8 | 0.7 | 0.5% | 0% |
| 80 | 1091 | 112.2 | 1.6 | 1.4% | 0% |
| 90 | 1036 | 135.1 | 6.9 | 5.1% | 2% |
| 100 | 958 | 121.1 | 7.6 | 6.3% | 2% |
| 120 | 893 | 201.6 | 13.0 | 6.5% | 3% |
| 140 | 808 | 278.0 | 14.4 | 5.2% | 4% |
| 160 | 700 | 339.1 | 22.3 | 6.6% | 6% |
| 180 | 586 | 349.0 | 39.7 | 11.4% | 10% |
| 200 | 452 | 322.0 | 38.2 | 11.9% | 10% |
| 250 | 289 | 264.1 | 32.4 | 12.3% | 8% |
| 300 | 177 | 258.4 | 49.0 | 18.9% | 12% |
| 350 | 83 | 153.6 | 43.6 | 28.4% | 11% |
| 400 | 35 | 74.7 | 39.6 | 52.9% | 10% |
| 450 | 16 | 49.7 | 17.2 | 34.5% | 4% |
| 500 | 1 | 1.8 | 0.0 | 0.0% | 0% |

## Maps (the 15 with the most savings)

| Map | Instances | Always bcp2 (min) | Savings (min) | Savings (% of bcp2) | Share |
|---|---|---|---|---|---|
| warehouse-20-40-10-2-2 | 937 | 204.9 | 110.0 | 53.7% | 28% |
| Berlin_1_256 | 827 | 185.4 | 67.3 | 36.3% | 17% |
| warehouse-20-40-10-2-1 | 829 | 146.6 | 46.3 | 31.6% | 12% |
| empty-48-48 | 722 | 83.0 | 41.7 | 50.2% | 11% |
| maze-32-32-4 | 216 | 87.7 | 39.6 | 45.1% | 10% |
| maze-128-128-1 | 43 | 52.5 | 22.1 | 42.0% | 6% |
| warehouse-10-20-10-2-2 | 823 | 104.4 | 20.5 | 19.6% | 5% |
| random-64-64-10 | 721 | 51.8 | 13.6 | 26.2% | 3% |
| den520d | 761 | 168.6 | 12.0 | 7.1% | 3% |
| empty-32-32 | 617 | 95.0 | 7.9 | 8.3% | 2% |
| warehouse-10-20-10-2-1 | 679 | 80.1 | 6.2 | 7.8% | 2% |
| Paris_1_256 | 854 | 158.3 | 5.5 | 3.4% | 1% |
| room-64-64-8 | 190 | 44.9 | 1.9 | 4.2% | 0% |
| w_woundedcoast | 629 | 190.3 | 1.1 | 0.6% | 0% |
| random-32-32-10 | 511 | 44.1 | 0.4 | 0.9% | 0% |

## Size of the win

| Saving per instance | Instances | Savings (min) | Share |
|---|---|---|---|
| ≤ 0.01 s | 709 | 0.0 | 0% |
| 0.01–0.1 s | 948 | 0.6 | 0% |
| 0.1–1 s | 690 | 3.9 | 1% |
| 1–10 s | 408 | 22.7 | 6% |
| 10–60 s | 141 | 53.4 | 13% |
| > 60 s | 108 | 316.6 | 80% |
