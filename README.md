# MAPFAST

Picks the fastest Multi-Agent Path Finding (MAPF) solver for a problem instance. A CNN reads an image of the map with each agent's shortest path drawn on it and predicts which solver in a portfolio will finish first. Based on:

> Jingyao Ren, Vikraman Sathiyanarayanan, Eric Ewing, Baskin Senbaslar and Nora Ayanian. "MAPFAST: A Deep Algorithm Selector for Multi Agent Path Finding using Shortest Path Embeddings." In *Proceedings of the 20th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2021)*, pp. 1055–1062. [arXiv:2102.12461](https://arxiv.org/abs/2102.12461)

| Selector | Model and solver portfolio |
|---|---|
| MAPFAST | The paper's CNN, on `bcp2`, `cbs`, `cbsh2_rtc`, `reloc` (its BCP, CBS, CBSH and SAT) |
| MAPFASTv2 | The same network, on `bcp2`, `cbs`, `ecbs`, `cbsh2_rtc`, `cbsh2_rtc_ch`, `cbsh2_rtc_bp`, `cbsh2_rtc_chbp`, `reloc` |
| MAPFAST-GNN | A graph neural network on the map's grid graph, trained on the seconds a pick loses and falling back to the default solver unless confident; MAPFASTv2's portfolio |
| Baseline | a general-purpose model, TypeSafe AI's [Jev](https://docs.typesafe.ai/), given the same choice |

The solvers come from these libraries, built and benchmarked by the MAPF Benchmarking repository (all forks with small changes):

| Solvers | Library |
|---|---|
| `cbs`, `ecbs` (and `mapf_prioritized_sipp`, left out of MAPFASTv2) | [libMultiRobotPlanning](https://github.com/whoenig/libMultiRobotPlanning) (W. Hönig), [fork](https://github.com/mcapoor/libMultiRobotPlanning) |
| `cbsh2_rtc`, `cbsh2_rtc_ch`, `cbsh2_rtc_bp`, `cbsh2_rtc_chbp` | [CBSH2-RTC-CHBP](https://github.com/bshen95/CBSH2-RTC-CHBP) (B. Shen et al.), [fork](https://github.com/mcapoor/CBSH2-RTC-CHBP) |
| `bcp2` | [BCP2-MAPF](https://github.com/ed-lam/bcp2-mapf) (E. Lam; needs Gurobi), [fork](https://github.com/mcapoor/bcp2-mapf) |
| `reloc` | [reLOC](https://github.com/surynek/reLOC) (P. Surynek), [fork](https://github.com/mcapoor/reLOC) |

The problem instances are the [MovingAI MAPF benchmarks](https://movingai.com/benchmarks/mapf.html). All solvers return optimal solutions except `ecbs` (bounded-suboptimal) and `mapf_prioritized_sipp` (neither optimal nor complete).

Results and the history of each experiment are in [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md).

## Layout

| Folder | Contents |
|---|---|
| `models/` | `MAPFAST.py` and `utils.py` (model), `main.py` (train/test), `jev_baseline.py` (Jev baseline), `gnn.py` and `train_gnn.py` (GNN selector) |
| `tools/` | `benchmark_dataset.py`, `derive_dataset.py` (build datasets), `build_graphs.py` (GNN graphs), `plots.py`, `plot_sweep.py` (evaluate), `analysis.py` (the paper's evaluation, with its custom score) |
| `scripts/` | Slurm jobs: `generate_benchmark.sh`, `train.sh`, `compare.sh`, `sweep_gnn.sh` |
| `datasets/` | `<benchmark>/<portfolio>/[<variant>/]`, e.g. `movingai_10-500-agents_300s/eight-solvers/gnn`: `config.json`, solver times, agents, maps and images |
| `experiments/` | Logs, graphs and predictions of each round |

Run everything from this folder.

## Usage

1. **Build the datasets** (needs the MAPF Benchmarking repository set up on the cluster, at `$BENCH`, default `~/MAPF-Benchmarking`). This benchmarks every solver on every map (up to 500 agents, 300 s limit) as a MAPF Benchmarking experiment, then writes `datasets/<experiment>/four-solvers` and `nine-solvers`:
   ```bash
   ./scripts/generate_benchmark.sh movingai_10-500-agents_300s
   ```
2. **Make variants** without re-rendering the images, e.g. dropping a solver or adding the auxiliary outputs:
   ```bash
   python tools/derive_dataset.py datasets/movingai_10-500-agents_300s/nine-solvers --name movingai_10-500-agents_300s/eight-solvers --drop mapf_prioritized_sipp
   ```
3. **Train, test and compare.** This submits one training job per model, the Jev baselines (needs `TYPESAFE_API_KEY`) and a comparison plot, all written to `experiments/round_N/`:
   ```bash
   ./scripts/sweep_gnn.sh datasets/movingai_10-500-agents_300s/eight-solvers     # MAPFAST-GNN, once
   TRAIN_TIME=24:00:00 OUT=experiments/round_3 ./scripts/compare.sh MAPFAST=datasets/movingai_10-500-agents_300s/four-solvers \
       MAPFASTv2=datasets/movingai_10-500-agents_300s/eight-solvers MAPFAST-GNN=@datasets/movingai_10-500-agents_300s/eight-solvers/gnn
   ```
   - `scripts/train.sh` creates the Python environment (`pytorch.venvsource/`) on its first run.
   - To run one model by hand: `python models/main.py -C <dataset>` to train, add `-T 0` to test, then `python tools/plots.py -C <dataset>`.

## Notes

- **Configs:** each dataset's `config.json` sets its solver portfolio, the loss (cross-entropy; `score`, the relative speed score; or `custom`, the paper's custom score), the outputs and the time limit. Paths in it are relative to the dataset folder.
- **Auxiliary outputs:** the original authors report that MAPFAST's auxiliary outputs (finish prediction, pairwise comparison) do not help (MAPFASTER, IROS 2022; "No Panacea in Planning").
