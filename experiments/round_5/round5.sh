#!/bin/bash
#SBATCH -J round5_setup
#SBATCH -t 0:30:00
#SBATCH --mem=8G
#SBATCH -o /oscar/home/mcapoor/MAPFASTv2/experiments/round_5/logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=milan_capoor@brown.edu
# Round 5 (see experiments/EXPERIMENTS.md). One-off: delete after the round. From MAPFAST on the cluster:
#   mkdir -p experiments/round_5/logs && sbatch experiments/round_5/round5.sh
# This job benchmarks bcp1 and lazycbs on round 3's instances (MAPF Benchmarking's experiment $EXPERIMENT, round 3's
# plus their runs), then submits itself again ("models") to run once they are merged: it builds the ten-solver dataset
# (round 3's eight-solvers, the same instances and split, plus bcp1 and lazycbs) and submits the MAPFAST-GNN sweep on it
# with 5-fold cross-validation, and the comparison, which trains MAPFASTv2 and MAPFASTv2_aux on it and plots them,
# round 3's MAPFAST and MAPFAST_aux (whose four-solver dataset has not changed), round 4's best MAPFAST-GNN (paths)
# and this round's best configuration of each kind of edges. The comparison is held until a last run of this job
# ("select") has picked those (summary_<edges>.md).

set -eo pipefail
cd ~/MAPFASTv2
BENCH=~/MAPF-Benchmarking
EXPERIMENT=movingai_10-500-agents_300s_bcp1-lazycbs
D=datasets/movingai_10-500-agents_300s
SWEEP=experiments/round_5/gnn_sweep
export LOGS=experiments/round_5/logs
mkdir -p $LOGS

case "$1" in
"")
    EXPERIMENT=$EXPERIMENT $BENCH/benchmark.sh --timeout 300 --reuse movingai_10-500-agents_300s --solvers bcp1 lazycbs \
        2>&1 | tee $LOGS/benchmark_jobs.txt
    MERGE=$(sed -n 's/^Merge: job \([0-9]*\).*/\1/p' $LOGS/benchmark_jobs.txt)
    sbatch -J round5_models --dependency=afterok:$MERGE experiments/round_5/round5.sh models
    ;;
models)
    source pytorch.venvsource/bin/activate
    # the instances of eight-solvers (one of MAPFAST's four solvers solves them), whose images nine-solvers has
    mkdir -p $D/ten-solvers
    ln -sfn ../nine-solvers/images $D/ten-solvers/images
    python tools/benchmark_dataset.py $BENCH/experiments/$EXPERIMENT --benchmarks $BENCH/benchmarks \
        --name ${D#datasets/}/ten-solvers --timeout 300 --augmentation 6 --jobs 1 --solved-by bcp2 cbs cbsh2_rtc reloc \
        --solvers bcp1 bcp2 cbs cbsh2_rtc cbsh2_rtc_bp cbsh2_rtc_ch cbsh2_rtc_chbp ecbs lazycbs reloc
    python tools/derive_dataset.py $D/ten-solvers --name ${D#datasets/}/ten-solvers/aux --aux --epochs 15
    for edges in ends paths; do
        python tools/derive_dataset.py $D/ten-solvers --name ${D#datasets/}/ten-solvers/gnn-$edges
    done

    # five folds take about five times as long as round 4's configurations
    OUT=$SWEEP VARIANT=gnn-round5 FOLDS=5 TRAIN_TIME=24:00:00 ./scripts/sweep_gnn.sh $D/ten-solvers 2>&1 | tee $LOGS/gnn_jobs.txt
    SELECT=$(sed -n 's/^select: job \([0-9]*\).*/\1/p' $LOGS/gnn_jobs.txt)

    OUT=experiments/round_5 TRAIN_TIME=24:00:00 JEV_EXAMPLES= ./scripts/compare.sh \
        MAPFAST_GNN_ends=@$D/ten-solvers/gnn-ends \
        MAPFAST_GNN_paths=@$D/ten-solvers/gnn-paths \
        MAPFAST_GNN_round4=@$D/eight-solvers/gnn-paths \
        MAPFASTv2=$D/ten-solvers \
        MAPFASTv2_aux=$D/ten-solvers/aux \
        MAPFAST_round3=@$D/four-solvers \
        MAPFAST_aux_round3=@$D/four-solvers/aux \
        2>&1 | tee $LOGS/compare_jobs.txt
    PLOT=$(sed -n 's/^Comparison: job \([0-9]*\).*/\1/p' $LOGS/compare_jobs.txt)

    # the comparison already waits for MAPFASTv2 and MAPFASTv2_aux; hold it until the GNNs are selected too
    scontrol hold $PLOT
    sbatch -J round5_select --dependency=afterok:$SELECT experiments/round_5/round5.sh select $PLOT
    ;;
select)
    source pytorch.venvsource/bin/activate
    for edges in ends paths; do
        python models/train_gnn.py select $SWEEP --where edges=$edges --dataset-out $D/ten-solvers/gnn-$edges
    done
    scontrol release $2
    ;;
esac
