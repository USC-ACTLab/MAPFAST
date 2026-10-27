#!/bin/bash
#SBATCH -J round6
#SBATCH -t 2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH -o /oscar/home/mcapoor/MAPFASTv2/experiments/round_6/logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=milan_capoor@brown.edu
# Round 6: round 5's MAPFAST-GNN networks without the fallback rule (always the top output). One-off. From MAPFAST:
#   mkdir -p experiments/round_6/logs && sbatch experiments/round_6/round6.sh
# The fallback only acts on selection and test picks, so no network is retrained: train_gnn.py reuses round 5's
# per-fold outputs and only reselects the epoch with threshold 0 (CPU, minutes).

set -eo pipefail
cd ~/MAPFASTv2
source pytorch.venvsource/bin/activate
D=datasets/movingai_10-500-agents_300s/ten-solvers
OLD=experiments/round_5/gnn_sweep
SWEEP=experiments/round_6/gnn_sweep

for config in "$OLD"/*/; do
	mkdir -p "$SWEEP/$(basename "$config")"
	cp "$config"/fold*of5.npz "$SWEEP/$(basename "$config")/"
done
for i in $(seq 0 $(($(python models/train_gnn.py list) - 1))); do
	python -u models/train_gnn.py train $D --config-index $i --out $SWEEP --folds 5 --no-fallback
done

declare -A best
for edges in ends paths; do
	python tools/derive_dataset.py $D --name ${D#datasets/}/gnn-$edges-no-fallback
	best[$edges]=$(python models/train_gnn.py select $SWEEP --where edges=$edges --dataset-out $D/gnn-$edges-no-fallback |
		tee /dev/stderr | sed -n 's/^Best: \([^ ]*\).*/\1/p')
done

python tools/plots.py \
	--run MAPFAST_GNN_ends $D/gnn-ends $OLD/sage_h128_l4_lr0.001_ends_runtime/results.json \
	--run MAPFAST_GNN_ends_no_fallback $D/gnn-ends-no-fallback $SWEEP/${best[ends]}/results.json \
	--run MAPFAST_GNN_paths $D/gnn-paths $OLD/sage_h128_l4_lr0.001_paths_runtime/results.json \
	--run MAPFAST_GNN_paths_no_fallback $D/gnn-paths-no-fallback $SWEEP/${best[paths]}/results.json \
	--run MAPFASTv2 $D experiments/round_5/logs/MAPFASTv2_7883079.out \
	-o experiments/round_6/comparison
