# MAPFAST-GNN hyperparameter sweep

16 of 16 configurations finished, ranked by validation speedup over always using bcp2 (the test columns are only reported). Validation: always bcp2 336.2 min, perfect selector 301.3 min.

| Rank | Configuration | Epoch | Threshold | Val. speedup | Val. accuracy | Test speedup | Test accuracy | ms / instance |
|---|---|---|---|---|---|---|---|---|
| 1 | `sage_h128_l4_lr0.001_paths_runtime` | 25 | 0.25 | 1.0417 | 0.8171 | 0.9918 | 0.8318 | 6.55 |
| 2 | `sage_h128_l4_lr0.001_ends_runtime` | 28 | 0 | 1.0408 | 0.7947 | 0.9783 | 0.8124 | 6.16 |
| 3 | `sage_h128_l4_lr0.001_paths_ce` | 12 | 0 | 1.0371 | 0.8208 | 0.9877 | 0.8300 | 6.68 |
| 4 | `sage_h128_l8_lr0.001_ends_runtime` | 29 | 0 | 1.0338 | 0.7819 | 0.9751 | 0.7936 | 10.64 |
| 5 | `gin_h128_l4_lr0.001_ends_runtime` | 35 | 0 | 1.0329 | 0.7606 | 0.9961 | 0.7851 | 5.84 |
| 6 | `gin_h128_l4_lr0.001_paths_ce` | 33 | 0.25 | 1.0319 | 0.8244 | 0.9129 | 0.8349 | 6.40 |
| 7 | `gin_h128_l8_lr0.001_paths_runtime` | 33 | 0 | 1.0302 | 0.7801 | 0.9784 | 0.7948 | 11.31 |
| 8 | `sage_h128_l8_lr0.001_paths_runtime` | 24 | 0.25 | 1.0258 | 0.8165 | 1.0032 | 0.8324 | 13.52 |
| 9 | `gin_h128_l4_lr0.001_paths_runtime` | 37 | 0 | 1.0233 | 0.7679 | 0.9823 | 0.7917 | 6.36 |
| 10 | `gin_h128_l8_lr0.001_ends_runtime` | 37 | 0 | 1.0227 | 0.7649 | 0.9530 | 0.7766 | 10.91 |
| 11 | `gin_h128_l8_lr0.001_paths_ce` | 15 | 0.5 | 1.0213 | 0.8196 | 0.9877 | 0.8355 | 6.31 |
| 12 | `sage_h128_l8_lr0.001_ends_ce` | 16 | 0.25 | 1.0197 | 0.8196 | 0.9368 | 0.8324 | 12.71 |
| 13 | `sage_h128_l4_lr0.001_ends_ce` | 14 | 0 | 1.0194 | 0.8171 | 0.9632 | 0.8312 | 4.41 |
| 14 | `sage_h128_l8_lr0.001_paths_ce` | 28 | 1 | 1.0151 | 0.8214 | 1.0004 | 0.8342 | 10.20 |
| 15 | `gin_h128_l8_lr0.001_ends_ce` | 17 | 0.25 | 1.0149 | 0.8177 | 1.0008 | 0.8330 | 10.58 |
| 16 | `gin_h128_l4_lr0.001_ends_ce` | 13 | 0.25 | 1.0103 | 0.8183 | 0.9820 | 0.8312 | 5.28 |
