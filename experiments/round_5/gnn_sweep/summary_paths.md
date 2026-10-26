# MAPFAST-GNN hyperparameter sweep (edges=paths)

2 of 2 configurations finished, ranked by validation speedup over always using bcp2 (the test columns are only reported; with k-fold cross-validation, the folds' range is in brackets). Validation: always bcp2 2875.7 min, perfect selector 2237.4 min.

| Rank | Configuration | Epoch | Threshold | Val. speedup | Val. accuracy | Test speedup | Test accuracy | ms / instance |
|---|---|---|---|---|---|---|---|---|
| 1 | `sage_h128_l4_lr0.001_paths_runtime` | 27 | 0 | 1.0858 (1.060–1.119) | 0.7396 | 1.0908 | 0.7711 | 37.30 |
| 2 | `sage_h128_l4_lr0.001_paths_ce` | 14 | 1 | 1.0741 (1.047–1.117) | 0.7860 | 1.0707 | 0.7966 | 32.70 |
