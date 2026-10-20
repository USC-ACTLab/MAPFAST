#!/bin/bash
#SBATCH -J round3_setup
#SBATCH -t 1:00:00
#SBATCH --mem=16G
#SBATCH -o /oscar/home/mcapoor/MAPFASTv2/experiments/round_3/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=milan_capoor@brown.edu
# Round 3 (see experiments/EXPERIMENTS.md). One-off: delete after the round. From MAPFAST on the cluster:
#   sbatch experiments/round_3/round3.sh
# This job submits the benchmark extension, then submits itself again ("models") to run
# once the datasets exist, which derives the variants and submits the GNN sweep and the comparison.

set -eo pipefail
cd ~/MAPFASTv2
export BENCH=~/MAPF-Benchmarking
export TYPESAFE_API_KEY=$(tr -d '[:space:]' < typesafe_key.txt)
D=datasets/movingai_10-500-agents_300s

if [ "$1" != models ]; then
    # 1. the 300 s / 500-agent extension and its datasets; the first attempt ran with 32G per shard, so redo the runs
    #    the out-of-memory killer ended (and those whose reason was not recorded yet)
    RETRY=1 ./scripts/generate_benchmark.sh extend movingai_10-200-agents_60s 2>&1 | tee experiments/round_3/benchmark_jobs.txt
    DATASETS=$(sed -n 's/^Datasets: *job \([0-9]*\).*/\1/p' experiments/round_3/benchmark_jobs.txt)
    sbatch -J round3_models --dependency=afterok:$DATASETS experiments/round_3/round3.sh models
    exit
fi

# 2. the dataset variants, the MAPFAST-GNN sweep and the comparison
source pytorch.venvsource/bin/activate
python tools/derive_dataset.py $D/nine-solvers --name ${D#datasets/}/eight-solvers --drop mapf_prioritized_sipp
python tools/derive_dataset.py $D/eight-solvers --name ${D#datasets/}/eight-solvers/gnn
python tools/derive_dataset.py $D/four-solvers --name ${D#datasets/}/four-solvers/aux --aux --epochs 15
python tools/derive_dataset.py $D/eight-solvers --name ${D#datasets/}/eight-solvers/aux --aux --epochs 15

./scripts/sweep_gnn.sh $D/eight-solvers 2>&1 | tee experiments/round_3/gnn_jobs.txt
SELECT=$(sed -n 's/^select: job \([0-9]*\).*/\1/p' experiments/round_3/gnn_jobs.txt)

OUT=experiments/round_3 TRAIN_TIME=24:00:00 JEV_EXAMPLES=20 ./scripts/compare.sh \
    MAPFAST=$D/four-solvers \
    MAPFASTv2=$D/eight-solvers \
    MAPFAST_aux=$D/four-solvers/aux \
    MAPFASTv2_aux=$D/eight-solvers/aux \
    MAPFAST_GNN=@$D/eight-solvers/gnn \
    2>&1 | tee experiments/round_3/compare_jobs.txt

PLOT=$(sed -n 's/^Comparison: job \([0-9]*\).*/\1/p' experiments/round_3/compare_jobs.txt)

# the comparison already waits for the models it trains; hold it until the sweep has chosen MAPFAST-GNN too
scontrol hold $PLOT
sbatch -J round3_release --dependency=afterok:$SELECT -t 0:05:00 --wrap "scontrol release $PLOT"
