# MAPFAST-GNN hyperparameter sweep (edges=ends)

8 of 8 configurations finished, ranked by validation speedup over always using bcp2 (the test columns are only reported). Validation: always bcp2 336.2 min, perfect selector 301.3 min.

| Rank | Configuration | Epoch | Threshold | Val. speedup | Val. accuracy | Test speedup | Test accuracy | ms / instance |
|---|---|---|---|---|---|---|---|---|
| 1 | `sage_h128_l4_lr0.001_ends_runtime` | 28 | 0 | 1.0408 | 0.7947 | 0.9783 | 0.8124 | 6.16 |
| 2 | `sage_h128_l8_lr0.001_ends_runtime` | 29 | 0 | 1.0338 | 0.7819 | 0.9751 | 0.7936 | 10.64 |
| 3 | `gin_h128_l4_lr0.001_ends_runtime` | 35 | 0 | 1.0329 | 0.7606 | 0.9961 | 0.7851 | 5.84 |
| 4 | `gin_h128_l8_lr0.001_ends_runtime` | 37 | 0 | 1.0227 | 0.7649 | 0.9530 | 0.7766 | 10.91 |
| 5 | `sage_h128_l8_lr0.001_ends_ce` | 16 | 0.25 | 1.0197 | 0.8196 | 0.9368 | 0.8324 | 12.71 |
| 6 | `sage_h128_l4_lr0.001_ends_ce` | 14 | 0 | 1.0194 | 0.8171 | 0.9632 | 0.8312 | 4.41 |
| 7 | `gin_h128_l8_lr0.001_ends_ce` | 17 | 0.25 | 1.0149 | 0.8177 | 1.0008 | 0.8330 | 10.58 |
| 8 | `gin_h128_l4_lr0.001_ends_ce` | 13 | 0.25 | 1.0103 | 0.8183 | 0.9820 | 0.8312 | 5.28 |
