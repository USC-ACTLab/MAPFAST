#!/bin/bash
#SBATCH -J round4_setup
#SBATCH -t 0:30:00
#SBATCH --mem=8G
#SBATCH -o /oscar/home/mcapoor/MAPFASTv2/experiments/round_4/logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=milan_capoor@brown.edu
# Round 4 (see experiments/EXPERIMENTS.md). One-off: delete after the round. From MAPFAST on the cluster:
#   mkdir -p experiments/round_4/logs && sbatch experiments/round_4/round4.sh
# This job submits the MAPFAST-GNN sweep on round 3's eight-solver dataset (cell graphs with agent nodes, agents joined
# to their start and goal only, "ends", or to their whole path, "paths"), then submits itself again ("compare") to run
# once the sweep has ended: it picks the best configuration of each kind of edges (summary_<edges>.md) and plots both
# against always-bcp2, the oracles and round 3's MAPFAST-GNN and MAPFAST.

set -eo pipefail
cd ~/MAPFASTv2
D=datasets/movingai_10-500-agents_300s
SWEEP=experiments/round_4/gnn_sweep
export LOGS=experiments/round_4/logs
mkdir -p $LOGS

if [ "$1" != compare ]; then
    OUT=$SWEEP VARIANT=gnn-round4 ./scripts/sweep_gnn.sh $D/eight-solvers 2>&1 | tee $LOGS/gnn_jobs.txt
    SELECT=$(sed -n 's/^select: job \([0-9]*\).*/\1/p' $LOGS/gnn_jobs.txt)
    sbatch -J round4_compare --dependency=afterok:$SELECT experiments/round_4/round4.sh compare
    exit
fi

source pytorch.venvsource/bin/activate
for edges in ends paths; do
    [ -f $D/eight-solvers/gnn-$edges/config.json ] || python tools/derive_dataset.py $D/eight-solvers --name ${D#datasets/}/eight-solvers/gnn-$edges
    python models/train_gnn.py select $SWEEP --where edges=$edges --dataset-out $D/eight-solvers/gnn-$edges
done

# every model only plots its existing predictions; round 3's MAPFAST brings the four-solver oracle
OUT=experiments/round_4 JEV_EXAMPLES= ./scripts/compare.sh \
    MAPFAST_GNN_ends=@$D/eight-solvers/gnn-ends \
    MAPFAST_GNN_paths=@$D/eight-solvers/gnn-paths \
    MAPFAST_GNN_round3=@$D/eight-solvers/gnn \
    MAPFAST_round3=@$D/four-solvers \
    2>&1 | tee $LOGS/compare_jobs.txt
