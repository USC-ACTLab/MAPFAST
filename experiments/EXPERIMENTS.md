# Experiments

Each round has its own folder: Slurm logs (`<job name>_<job id>.out/.err`), per-model graphs (`<job>_plots/`), comparison graphs (`*_vs_*/`) and, where copied, the test predictions (`predictions/<model>/`).

| Folder | What |
|---|---|
| [round_0_pilot/](round_0_pilot/) | First runs on the small `nine_solvers` dataset (Oct 7) |
| [dataset_build/](dataset_build/) | Building `full_four_solvers` and `full_nine_solvers` from MAPF Benchmarking (Oct 9) |
| [round_1/](round_1/) | MAPFAST vs MAPFASTv2, cross-entropy (Oct 15) |
| [round_2/](round_2/) | Relative speed loss, best epoch, no SIPP, auxiliary outputs, Jev baseline |

## Round 0: pilot (Oct 7)

- Job 7094489: MAPFASTv2 on the small 9-solver dataset (342 test instances).
- Job 7094486: MAPFAST. It was cancelled after 20 minutes.

This round only checked that the training, test and plotting steps run end to end on the cluster.

![Round 0 runtime](round_0_pilot/7094489_plots/runtime.png)

## Dataset build (Oct 9)

- Job 7095349 merged 9 MAPF Benchmarking runs: 22,205 instances, 60 s limit.
- 15,594 of them are solved by one of bcp2, cbs, cbsh2_rtc or reloc. Both datasets keep exactly these instances:
  - `full_four_solvers`: those four solvers.
  - `full_nine_solvers`: all nine solvers.
- bcp2 alone solves 15,569 of the 15,594.

## Round 1: MAPFAST vs MAPFASTv2, cross-entropy (Oct 15)

- Jobs 7427439 (MAPFAST: bcp2, cbs, cbsh2_rtc, reloc) and 7427440 (MAPFASTv2: all nine solvers). Comparison: job 7427441.
- All runs use a 60 s limit and the same 1,557 test instances.

| | Total runtime (min) | Always bcp2 | Oracle |
|---|---|---|---|
| MAPFAST | 93.8 | 75.0 | 73.0 |
| MAPFASTv2 | 81.2 | 75.0 | 48.3 |

![Round 1 runtime relative to the oracle and bcp2](round_1/MAPFAST_vs_MAPFASTv2_7427441/compare_runtime.png)

| MAPFAST | MAPFASTv2 |
|---|---|
| ![MAPFAST runtime](round_1/MAPFAST_7427439_plots/runtime.png) | ![MAPFASTv2 runtime](round_1/MAPFASTv2_7427440_plots/runtime.png) |
| ![MAPFAST confusion](round_1/MAPFAST_7427439_plots/confusion.png) | ![MAPFASTv2 confusion](round_1/MAPFASTv2_7427440_plots/confusion.png) |
| ![MAPFAST loss](round_1/MAPFAST_7427439_plots/loss.png) | ![MAPFASTv2 loss](round_1/MAPFASTv2_7427440_plots/loss.png) |

Problems:

- **Both lose to always-bcp2.** bcp2 solves every test instance, so a wrong pick that times out costs 60 s for nothing. 17 (v1) and 19 (v2) such picks cost about 15 min each, more than any gains.
- **Cross-entropy ignores the cost of mistakes.** Losing by 0.04 s counts the same as a 60 s timeout.
- **Little room to improve for v1.** A perfect selector saves only 2 min over bcp2, unlike the paper, where BCP failed on 13% of instances.
- **Most of v2's room comes from a non-optimal solver.** 22 of v2's 27 possible minutes come from `mapf_prioritized_sipp`, which doesn't guarantee optimal paths. Its picks saved 10.4 min; the cbsh2 picks lost 16.7.
- **Training always kept epoch 9.** v2's validation loss was lowest around epochs 2–3.
- **Accuracy is misleading.** Always picking bcp2 scores 88.5% (v1) and 72.3% (v2), so the two models' accuracies can't be compared.

## Round 2: relative speed loss, best epoch, no SIPP, auxiliary outputs, Jev baseline

Changes:

- **Loss:** 1 − expected relative speed score of the softmax. A solver scores (1 + fastest time) / (1 + its time), or 0 if it fails. This is analysis.py's custom score divided by the fastest solver's (`Training.loss: score`).
- **Best epoch:** the checkpoint with the best validation score of its picks, saved as `model_<name>_best.pth`, is the one tested.
- **No SIPP:** v2 uses eight solvers, all optimal (`derive_dataset.py --drop mapf_prioritized_sipp`).
- **Ablation:** each of v1 and v2 is trained with and without MAPFAST's auxiliary outputs: solver finish prediction and pairwise comparison (`--aux`). The README says these didn't help in follow-up papers.
- **General-purpose baseline:** answers the reviewer objection that bespoke selectors are obsolete now that general-purpose models exist. It uses TypeSafe AI's Jev, a hosted model built for structured decisions (`jev_baseline.py`).
  - For each test instance Jev gets the map, agent count, start-goal distances and coordinates, and picks the fastest solver as a Choice question.
  - It runs twice: zero-shot (`_jev0`) and with 20 training instances from the same map plus every solver's time (`_jev20`), its best shot without training.
  - **Selector time** is recorded for both kinds of model (Jev's per-request latency; MAPFAST's inference time in the test log), since it adds to the total runtime before any solver starts.
- **Same instances and split as round 1,** so results compare directly.
- **New graph** `compare_total_runtime.png`: every model, both oracles and every solver.

Runs, with `TYPESAFE_API_KEY` set:

```bash
./scripts/compare.sh   # MAPFAST, MAPFASTv2, MAPFAST_aux, MAPFASTv2_aux, and Jev0/Jev20 on each portfolio's test instances
```

The Jev jobs start once the first model of their portfolio has been tested (`Jev0_MAPFAST`, `Jev20_MAPFAST`, `Jev0_MAPFASTv2`, `Jev20_MAPFASTv2`).

Results: _to fill in_. `compare.sh` writes the logs, graphs and predictions to `experiments/round_2/` on the cluster; copy that folder back here and link `compare_total_runtime.png` and `compare_runtime.png`.
