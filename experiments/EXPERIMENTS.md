# Experiments

What changed between rounds (datasets, models, losses) and the results. Logs and graphs are in each round's folder.

| Folder | What |
| --- | --- |
| [round_0_pilot/](round_0_pilot/) | Pipeline check on the small `movingai_pilot/nine-solvers` dataset |
| [dataset_build/](dataset_build/) | Dataset builds from MAPF Benchmarking |
| [round_1/](round_1/) | MAPFAST vs MAPFASTv2, cross-entropy |
| [round_2/](round_2/) | Relative speed loss, best epoch, no SIPP, auxiliary outputs, Jev baseline |
| [round_3/](round_3/) | 300 s, up to 500 agents; MAPFAST-GNN |
| [round_4/](round_4/) | MAPFAST-GNN on full-resolution graphs with agent nodes, runtime loss |
| [round_5/](round_5/) | Ten solvers (bcp1, lazycbs), MAPFASTv2 retrained, MAPFAST-GNN with 5-fold cross-validation |
| [headroom/](headroom/) | Where the oracle saves time over always-bcp2 (`tools/headroom.py`) |
| [gnn_sweep/](gnn_sweep/) | MAPFAST-GNN hyperparameter sweeps, one folder per dataset |
| [gnn_architecture/](gnn_architecture/) | MAPFAST-GNN architecture figure (TikZ source) |

Datasets are `datasets/<benchmark>/<portfolio>/[<variant>/]`.

## Dataset: `movingai_10-200-agents_60s` (rounds 1–2)

- 22,205 instances, 10–200 agents, 60 s limit; 15,594 are solved by one of bcp2, cbs, cbsh2_rtc and reloc, and both portfolios keep exactly these.
- `four-solvers` (MAPFAST: bcp2, cbs, cbsh2_rtc, reloc) and `nine-solvers` (all nine; `eight-solvers` drops SIPP).
- bcp2 alone solves 15,569 of the 15,594.

## Round 1: MAPFAST vs MAPFASTv2, cross-entropy

Same 1,557 test instances for both.

| | Total runtime (min) | Always bcp2 | Oracle |
| --- | --- | --- | --- |
| MAPFAST | 93.8 | 75.0 | 73.0 |
| MAPFASTv2 (nine solvers) | 81.2 | 75.0 | 48.3 |

- **Both lose to always-bcp2:** 17–19 picks of solvers that time out cost about 15 min, more than any gains. Cross-entropy treats a 0.04 s loss like a 60 s timeout.
- **Little headroom:** a perfect four-solver selector saves 2 min; 22 of MAPFASTv2's 27 possible minutes come from SIPP, which is not optimal.
- Training always kept the last epoch; accuracy is misleading (always-bcp2 scores 88.5% / 72.3%).

## Round 2: relative speed loss, best epoch, no SIPP, auxiliary outputs, Jev

Changes from round 1 (same instances and split):

- **Loss:** 1 − expected relative speed score of the softmax; a solver scores (1 + fastest time) / (1 + its time), 0 if it fails.
- **Best epoch** by validation score is tested.
- **No SIPP:** MAPFASTv2 uses eight solvers.
- **Ablation:** with and without MAPFAST's auxiliary outputs (finish prediction, pairwise comparison).
- **Jev baseline** (TypeSafe AI's general-purpose model): zero-shot and with 20 training examples from the same map.

| | Total runtime (min) |
| --- | --- |
| Oracle (eight / four solvers) | 69.5 / 73.0 |
| Always bcp2 | 75.0 |
| MAPFAST, MAPFASTv2 | 75.0, 75.0 |
| MAPFAST_aux, MAPFASTv2_aux | 75.7, 76.9 |
| Jev, 20 examples (four / eight) | 82.1 / 79.8 |
| Jev, zero-shot (four / eight) | 424.8 / 383.7 |
| Best other single solver (cbsh2_rtc_chbp) | 372.8 |

- **MAPFAST and MAPFASTv2 always pick bcp2;** the loss is flat from epoch 0. Without SIPP a perfect selector saves only 5.5 of 75 min (7%); other solvers win on 289 instances by a median 0.04 s.
- **Auxiliary outputs** move away from bcp2 and lose time; their losses were still falling at 10 epochs.
- **Jev** with examples is close to bcp2 but behind every trained model; zero-shot is 5–6× slower than bcp2.

## Round 3: harder instances, MAPFAST-GNN

Changes:

- **Dataset `movingai_10-500-agents_300s`:** the round 1 benchmark's unsolved runs rerun at **300 s**, plus **250–500 agents**. Per scenario file and solver, larger agent counts are skipped after the first failure. Only 651 of the 7,403 new 250–500-agent instances are solved by any optimal solver. Portfolios: `four-solvers`, `eight-solvers`.
- **MAPFAST, MAPFASTv2** unchanged (cross-entropy); **MAPFAST_aux, MAPFASTv2_aux** trained for 15 epochs.
- **MAPFAST-GNN** (`models/gnn.py`), eight solvers, MAPFASTv2's split:

  ![MAPFAST-GNN architecture](gnn_architecture/gnn_architecture.png)

  - Input: the map as a grid graph (at most 64 × 64 blocks) with per-block features (free space, starts, goals, shortest-path cells, path overlap) and global ones (agents, density, path lengths).
  - **Regret loss:** expected seconds lost to the fastest solver; a timeout costs 300 s.
  - **Fallback:** keeps the default solver (bcp2) unless the top pick's probability passes a threshold tuned on validation.
  - Epochs, thresholds and configurations are chosen by validation speedup over always-bcp2. The sweep covers GCN/GraphSAGE/GIN, width 32–256, depth 2–8, two learning rates, cross-entropy or regret loss.
- **Jev** with 20 examples only.

Results (1,647 test instances, failures counted as 300 s):

| | Total runtime (min) | Non-bcp2 picks (right) |
| --- | --- | --- |
| Oracle (eight / four solvers) | 324.9 / 339.3 | 276 / 157 |
| Always bcp2 | 348.7 | |
| MAPFAST-GNN (GraphSAGE, width 64, depth 2, cross-entropy, threshold 0.5) | 355.3 | 16 (8) |
| Jev, 20 examples (four / eight) | 400.6 / 443.8 | 94 (53) / 182 (54) |
| MAPFASTv2, MAPFASTv2_aux | 423.1, 465.1 | 183 (51), 217 (55) |
| MAPFAST, MAPFAST_aux | 479.3, 431.5 | 176 (80), 160 (64) |
| Best other single solver (cbsh2_rtc_chbp) | 2,162.1 | |

- **Nothing beats always-bcp2.** On the test set the oracle saves only 7% (eight solvers) or 3% (four), far below the preliminary 21%.
- **MAPFAST/v2 no longer collapse to bcp2** (unlike round 2) but most of their non-bcp2 picks are wrong, and wrong picks at 300 s cost minutes each.
- **MAPFAST-GNN:** all 72 regret-loss configurations stay at always-bcp2 (speedup 1.000). The 72 cross-entropy ones reach validation speedups of 1.039–1.044 but test speedups of 0.94–1.02, so the selection by validation does not transfer; the selected one makes 16 non-bcp2 picks.

## Round 4: MAPFAST-GNN with agent nodes, runtime loss

Changes from round 3's MAPFAST-GNN (same dataset `movingai_10-500-agents_300s/eight-solvers` and split):

- **No coarsening:** a node per free cell (k = 1; up to 96,603 cells on orz900d).
- **Agent nodes:** one per agent, with edges to its start and goal cells, so that which start goes with which goal is kept; features: path length, detour (path / Manhattan distance) and how often other agents' paths visit its path. Cells: starts, goals, paths through them. Cells and agents are pooled separately; each kind of edge has its own weights.
- **Path edges (ablation):** `ends` joins agents to their start and goal only; `paths` also to every cell of their path, so agents whose paths meet are 2 hops apart (with `ends`, information travels one cell per layer).
- **Runtime loss:** predicts each solver's log(1 + seconds) (a timeout only as a lower bound) and picks the fastest; the fallback rule leaves bcp2 only for a predicted speedup above a threshold tuned on validation. Cross-entropy is kept for comparison; the regret loss is dropped.
- **Sweep:** GraphSAGE/GIN, width 128, depth 4/8, `ends`/`paths`, runtime or cross-entropy loss (16 configurations); the best of each edge set is compared with always-bcp2, the oracles and round 3's models; no Jev.

Results (same 1,647 test instances):

| | Total runtime (min) | Test speedup over always-bcp2 |
| --- | --- | --- |
| Oracle (eight / four solvers) | 324.9 / 339.3 | 1.073 / 1.028 |
| Always bcp2 | 348.7 | 1 |
| MAPFAST-GNN `paths` (GraphSAGE, depth 4, runtime loss, threshold 0.25) | 351.5 | 0.992 |
| MAPFAST-GNN round 3 | 355.3 | 0.981 |
| MAPFAST-GNN `ends` (GraphSAGE, depth 4, runtime loss, threshold 0) | 356.4 | 0.978 |
| MAPFAST (round 3) | 479.3 | 0.728 |

- **Still nothing beats always-bcp2.** `paths` is the closest model so far (2.8 min behind), but the gaps between the GNNs are a few minutes, within what the choice of configuration changes.
- **Selection by validation is noise:** all 16 configurations reach validation speedups of 1.010–1.042 but test speedups of 0.913–1.003 (mean 0.977); across configurations, validation and test speedup are uncorrelated (r = −0.04). Choosing the epoch and threshold among 200 combinations on 1,646 validation instances fits that set's noise. The validation set also has more headroom (oracle 1.116× bcp2, test 1.073×).
- Averaged over configurations, the runtime loss beats cross-entropy (test 0.982 vs 0.971), `paths` beats `ends` (0.981 vs 0.973) and GraphSAGE beats GIN (0.980 vs 0.974); depth 8 is no better than 4.

## Round 5: ten solvers, MAPFASTv2 retrained, k-fold cross-validation

Changes from round 4 (same instances and test set):

- **Ten solvers:** `movingai_10-500-agents_300s/ten-solvers` is `eight-solvers` plus bcp1 and lazycbs (benchmarked on the same instances at 300 s).
- **MAPFAST-GNN with 5-fold cross-validation** over the training and validation instances: the epoch and threshold are chosen on all out-of-fold predictions (about 15,200 instances instead of 1,646), and the test picks average the five networks. Only GraphSAGE, depth 4 (`ends`/`paths`, runtime or cross-entropy loss: 4 configurations).
- **CNN baselines:** MAPFASTv2 and MAPFASTv2_aux (15 epochs) retrained on ten solvers; MAPFAST and MAPFAST_aux are round 3's (their four-solver dataset is unchanged); round 4's `paths` model as the GNN reference. No Jev.

Results (same 1,647 test instances):

| | Total runtime (min) | Test speedup over always-bcp2 | Non-bcp2 picks (fastest) |
| --- | --- | --- | --- |
| Oracle (ten / eight / four solvers) | 287.1 / 324.9 / 339.3 | 1.215 / 1.073 / 1.028 | |
| MAPFAST-GNN `paths` (runtime loss, threshold 0) | 319.7 | 1.091 | 228 (66), lazycbs 75 (41) |
| MAPFAST-GNN `ends` (runtime loss, threshold 0.25) | 322.2 | 1.082 | 67 (37), lazycbs 54 (35) |
| Always bcp2 | 348.7 | 1 | |
| MAPFAST-GNN round 4 (`paths`, eight solvers) | 351.5 | 0.992 | 22 (7) |
| MAPFASTv2, MAPFASTv2_aux | 368.5, 435.2 | 0.946, 0.801 | 269 (96), 237 (76) |
| MAPFAST, MAPFAST_aux (round 3) | 479.3, 431.5 | 0.728, 0.808 | 176 (80), 160 (64) |

- **MAPFAST-GNN beats always-bcp2 for the first time**, by 8–9%, taking 43–47% of the oracle's savings; all four configurations do (test 1.071–1.091).
- **The gain is mostly lazycbs.** It adds the headroom (ten-solver oracle 37.8 min below the eight-solver one) and its wins are predictable: open maps (empty-16/32/48, random-32/64-10) at 100–200 agents. Four fifths of `ends`'s non-bcp2 picks are lazycbs; the two GNNs agree on 51 of their 54/75 lazycbs picks. bcp1 is fastest on 48 test instances but no model exploits it.
- **Cross-validated selection transfers:** pooled out-of-fold speedups (1.072–1.095) match test (1.071–1.091) and every fold of every configuration beats bcp2 (1.041–1.156), unlike round 4's single validation set. This is confounded with the larger headroom (validation oracle 1.285×, round 4 1.116×).
- **MAPFASTv2 improves (423.1 → 368.5)** by also picking lazycbs (52, 38 fastest) but still loses on its 215 cbsh2_rtc_ch/chbp picks (57 fastest).
