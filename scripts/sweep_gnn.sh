#!/bin/bash
# Hyperparameter sweep of MAPFAST-GNN (models/gnn.py, models/train_gnn.py) on one dataset: build its graphs from the
# MovingAI files (tools/build_graphs.py) unless they exist, train every configuration of models/train_gnn.py's GRID (one
# GPU task of a Slurm job array each), then rank them by validation speedup and keep the best one's test predictions in
# the dataset's variant <dataset>/gnn/, which compare.sh can plot next to the other models (NAME=@<dataset>/gnn).
# usage, from the repository folder: ./scripts/sweep_gnn.sh DATASET
#   e.g. ./scripts/sweep_gnn.sh datasets/movingai_10-500-agents_300s/eight-solvers
#   OUT        folder for the logs, the trained configurations and summary.md (default experiments/gnn_sweep/<dataset
#              path below datasets/>)
#   BENCH      MAPF Benchmarking folder, for its benchmarks/ (default ~/MAPF-Benchmarking)
#   SIDE       most blocks along each side of the coarsened map grids (default 64)
#   EPOCHS     epochs per configuration (default 40)
#   PARALLEL   most configurations training at once (default 8)
#
# This script only submits jobs. Train a model with scripts/train.sh first, so that the virtual environment exists.

set -e

# Exits unless the dataset and the virtual environment exist
check_setup() {
	[ -f "$DATASET/config.json" ] || { echo "no config.json in $DATASET" >&2; exit 1; }
	[ -f pytorch.venvsource/bin/activate ] || { echo "no pytorch.venvsource/: run scripts/train.sh once first" >&2; exit 1; }
}

# Submits the job that builds the dataset's graphs, unless they exist; prints its id (nothing if they exist)
submit_graphs() {
	[ -f "$GRAPHS" ] && { echo "graphs: $GRAPHS exists" >&2; return; }
	local id
	id=$(sbatch --parsable -J "build_graphs_$NAME" -o "$OUT/%x_%j.out" -e "$OUT/%x_%j.err" \
		--nodes=1 --cpus-per-task=16 --time=4:00:00 --mem=32G \
		--wrap "source pytorch.venvsource/bin/activate && python -u tools/build_graphs.py $DATASET --benchmarks $BENCH/benchmarks --side $SIDE --jobs 16")
	echo "graphs: job $id builds $GRAPHS" >&2
	echo "$id"
}

# Submits one array task per hyperparameter configuration, after the graph job if there is one; prints the array's id
submit_sweep() {
	local after=() configs id
	[ -z "$1" ] || after=(--dependency="afterok:$1")
	configs=$(python -u models/train_gnn.py list)
	id=$(sbatch --parsable -J "gnn_$NAME" "${after[@]}" --array="0-$((configs - 1))%$PARALLEL" \
		-o "$OUT/%x_%A_%a.out" -e "$OUT/%x_%A_%a.err" \
		-p gpu --gres=gpu:1 --nodes=1 --cpus-per-task=4 --time=3:00:00 --mem=24G \
		--wrap "source pytorch.venvsource/bin/activate && python -u models/train_gnn.py train $DATASET --config-index \$SLURM_ARRAY_TASK_ID --out $OUT --side $SIDE --epochs $EPOCHS")
	echo "sweep: job array $id, $configs configurations, at most $PARALLEL at once, logs $OUT/gnn_${NAME}_${id}_<index>.out" >&2
	echo "$id"
}

# Submits the job that ranks the configurations once the array has ended (whether or not every task succeeded) and
# writes the best one's predictions to <dataset>/gnn/
submit_select() {
	local id
	id=$(sbatch --parsable -J "select_gnn_$NAME" --dependency="afterany:$1" -o "$OUT/%x_%j.out" -e "$OUT/%x_%j.err" \
		--time=0:15:00 --mem=8G --cpus-per-task=1 \
		--mail-type=END,FAIL --mail-user=milan_capoor@brown.edu \
		--wrap "set -e; source pytorch.venvsource/bin/activate
[ -f $DATASET/gnn/config.json ] || python -u tools/derive_dataset.py $DATASET --name ${REL}/gnn
python -u models/train_gnn.py select $OUT --dataset-out $DATASET/gnn
python -u tools/analysis.py -C $DATASET/gnn")
	echo "select: job $id, ranking in $OUT/summary.md, best predictions in $DATASET/gnn/" >&2
	echo "Then compare: ./scripts/compare.sh ... MAPFAST-GNN=@$DATASET/gnn" >&2
}

# run from the repository folder, whose paths (datasets/, models/, tools/, ...) every job uses
cd "$(dirname "$0")/.."
DATASET="${1:?usage: ./scripts/sweep_gnn.sh DATASET}"
DATASET="${DATASET%/}"
# its path below datasets/, and that path as one word for the job names
REL="${DATASET#datasets/}"
NAME="${REL//\//_}"
OUT="${OUT:-experiments/gnn_sweep/$REL}"
BENCH=$(readlink -f "${BENCH:-$HOME/MAPF-Benchmarking}")
SIDE="${SIDE:-64}"
EPOCHS="${EPOCHS:-40}"
PARALLEL="${PARALLEL:-8}"
GRAPHS="$DATASET/graphs_s$SIDE.pkl"

check_setup
mkdir -p "$OUT"
source pytorch.venvsource/bin/activate
graphs_job=$(submit_graphs)
sweep_job=$(submit_sweep "$graphs_job")
submit_select "$sweep_job"
