# Experiments

What changed between rounds (datasets, models, losses) and the results. Logs and graphs are in each round's folder.

| Folder | What |
|---|---|
| [round_0_pilot/](round_0_pilot/) | Pipeline check on the small `movingai_pilot/nine-solvers` dataset |
| [dataset_build/](dataset_build/) | Dataset builds from MAPF Benchmarking |
| [round_1/](round_1/) | MAPFAST vs MAPFASTv2, cross-entropy |
| [round_2/](round_2/) | Relative speed loss, best epoch, no SIPP, auxiliary outputs, Jev baseline |
| [round_3/](round_3/) | 300 s, up to 500 agents; MAPFAST-GNN |
| [gnn_sweep/](gnn_sweep/) | MAPFAST-GNN hyperparameter sweeps, one folder per dataset |

Datasets are `datasets/<benchmark>/<portfolio>/[<variant>/]`.

## Dataset: `movingai_10-200-agents_60s` (rounds 1–2)

- 22,205 instances, 10–200 agents, 60 s limit; 15,594 are solved by one of bcp2, cbs, cbsh2_rtc and reloc, and both portfolios keep exactly these.
- `four-solvers` (MAPFAST: bcp2, cbs, cbsh2_rtc, reloc) and `nine-solvers` (all nine; `eight-solvers` drops SIPP).
- bcp2 alone solves 15,569 of the 15,594.

## Round 1: MAPFAST vs MAPFASTv2, cross-entropy

Same 1,557 test instances for both.

| | Total runtime (min) | Always bcp2 | Oracle |
|---|---|---|---|
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
|---|---|
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

- **Dataset `movingai_10-500-agents_300s`:** the round 1 benchmark's unsolved runs rerun at **300 s**, plus **250–500 agents**. Per scenario file and solver, larger agent counts are skipped after the first failure. Portfolios: `four-solvers`, `eight-solvers`.
- **MAPFAST, MAPFASTv2** unchanged (cross-entropy); **MAPFAST_aux, MAPFASTv2_aux** trained for 15 epochs.
- **MAPFAST-GNN** (`models/gnn.py`), eight solvers, MAPFASTv2's split:
  - Input: the map as a grid graph (at most 64 × 64 blocks) with per-block features (free space, starts, goals, shortest-path cells, path overlap) and global ones (agents, density, path lengths).
  - **Regret loss:** expected seconds lost to the fastest solver; a timeout costs 300 s.
  - **Fallback:** keeps the default solver (bcp2) unless the top pick's probability passes a threshold tuned on validation.
  - Epochs, thresholds and configurations are chosen by validation speedup over always-bcp2. The sweep covers GCN/GraphSAGE/GIN, width 32–256, depth 2–8, two learning rates, cross-entropy or regret loss.
- **Jev** with 20 examples only.

Benchmark headroom (preliminary, before the out-of-memory reruns; instances some optimal solver solves, failures counted as 300 s, no SIPP):

| Agents | Instances | Always bcp2 (min) | Oracle (min) | Oracle saves |
|---|---|---|---|---|
| ≤ 30 | 4,658 | 172 | 110 | 36% |
| 31–100 | 8,104 | 715 | 689 | 4% |
| 101–200 | 3,425 | 1,437 | 1,255 | 13% |
| 201–300 | 470 | 555 | 419 | 25% |
| 301–400 | 132 | 305 | 137 | 55% |
| 401–500 | 49 | 212 | 63 | 70% |

- About **21% headroom** overall, against 7% at 60 s; most of it from the cbsh2_rtc variants.
- Only 651 of the 7,403 new 250–500-agent instances are solved by any optimal solver.

Results: _to fill in_.
