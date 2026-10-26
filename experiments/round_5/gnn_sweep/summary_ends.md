# MAPFAST-GNN hyperparameter sweep (edges=ends)

2 of 2 configurations finished, ranked by validation speedup over always using bcp2 (the test columns are only reported; with k-fold cross-validation, the folds' range is in brackets). Validation: always bcp2 2875.7 min, perfect selector 2237.4 min.

| Rank | Configuration | Epoch | Threshold | Val. speedup | Val. accuracy | Test speedup | Test accuracy | ms / instance |
|---|---|---|---|---|---|---|---|---|
| 1 | `sage_h128_l4_lr0.001_ends_runtime` | 37 | 0.25 | 1.0946 (1.052–1.156) | 0.7847 | 1.0820 | 0.7930 | 36.33 |
| 2 | `sage_h128_l4_lr0.001_ends_ce` | 14 | 0.5 | 1.0723 (1.041–1.113) | 0.7860 | 1.0836 | 0.7942 | 22.19 |
