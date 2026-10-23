# MAPFAST-GNN hyperparameter sweep (edges=paths)

8 of 8 configurations finished, ranked by validation speedup over always using bcp2 (the test columns are only reported). Validation: always bcp2 336.2 min, perfect selector 301.3 min.

| Rank | Configuration | Epoch | Threshold | Val. speedup | Val. accuracy | Test speedup | Test accuracy | ms / instance |
|---|---|---|---|---|---|---|---|---|
| 1 | `sage_h128_l4_lr0.001_paths_runtime` | 25 | 0.25 | 1.0417 | 0.8171 | 0.9918 | 0.8318 | 6.55 |
| 2 | `sage_h128_l4_lr0.001_paths_ce` | 12 | 0 | 1.0371 | 0.8208 | 0.9877 | 0.8300 | 6.68 |
| 3 | `gin_h128_l4_lr0.001_paths_ce` | 33 | 0.25 | 1.0319 | 0.8244 | 0.9129 | 0.8349 | 6.40 |
| 4 | `gin_h128_l8_lr0.001_paths_runtime` | 33 | 0 | 1.0302 | 0.7801 | 0.9784 | 0.7948 | 11.31 |
| 5 | `sage_h128_l8_lr0.001_paths_runtime` | 24 | 0.25 | 1.0258 | 0.8165 | 1.0032 | 0.8324 | 13.52 |
| 6 | `gin_h128_l4_lr0.001_paths_runtime` | 37 | 0 | 1.0233 | 0.7679 | 0.9823 | 0.7917 | 6.36 |
| 7 | `gin_h128_l8_lr0.001_paths_ce` | 15 | 0.5 | 1.0213 | 0.8196 | 0.9877 | 0.8355 | 6.31 |
| 8 | `sage_h128_l8_lr0.001_paths_ce` | 28 | 1 | 1.0151 | 0.8214 | 1.0004 | 0.8342 | 10.20 |
