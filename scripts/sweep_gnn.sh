#!/bin/bash
# MAPFAST-GNN sweep: graphs (unless built) -> one GPU job per GRID config -> select the best into <dataset>/$VARIANT.
# usage: ./scripts/sweep_gnn.sh DATASET   (env: OUT, LOGS, BENCH, EPOCHS, FOLDS, VARIANT, TRAIN_TIME, PARALLEL)

set -e

check_setup() {
	[ -f "$DATASET/config.json" ] || { echo "no config.json in $DATASET" >&2; exit 1; }
	[ -f pytorch.venvsource/bin/activate ] || { echo "no pytorch.venvsource/: run scripts/train.sh once first" >&2; exit 1; }
}

# prints the job id, or nothing if the graphs exist
submit_graphs() {
	[ -f "$GRAPHS" ] && { echo "graphs: $GRAPHS exists" >&2; return; }
	local id
	id=$(sbatch --parsable -J "build_graphs_$NAME" -o "$LOGS/%x_%j.out" -e "$LOGS/%x_%j.err" \
		--nodes=1 --cpus-per-task=16 --time=4:00:00 --mem=32G \
		--wrap "source pytorch.venvsource/bin/activate && python -u tools/build_graphs.py $DATASET --benchmarks $BENCH/benchmarks --jobs 16")
	echo "graphs: job $id builds $GRAPHS" >&2
	echo "$id"
}

submit_sweep() {
	local after=() configs id
	[ -z "$1" ] || after=(--dependency="afterok:$1")
	configs=$(python -u models/train_gnn.py list)
	id=$(sbatch --parsable -J "gnn_$NAME" "${after[@]}" --array="0-$((configs - 1))%$PARALLEL" \
		-o "$LOGS/%x_%A_%a.out" -e "$LOGS/%x_%A_%a.err" \
		-p gpu --gres=gpu:1 --nodes=1 --cpus-per-task=4 --time=${TRAIN_TIME:-12:00:00} --mem=48G \
		--wrap "source pytorch.venvsource/bin/activate && python -u models/train_gnn.py train $DATASET --config-index \$SLURM_ARRAY_TASK_ID --out $OUT --epochs $EPOCHS --folds $FOLDS")
	echo "sweep: job array $id, $configs configurations, at most $PARALLEL at once, logs $LOGS/gnn_${NAME}_${id}_<index>.out" >&2
	echo "$id"
}

# afterany: rank whatever configurations finished
submit_select() {
	local id
	id=$(sbatch --parsable -J "select_gnn_$NAME" --dependency="afterany:$1" -o "$LOGS/%x_%j.out" -e "$LOGS/%x_%j.err" \
		--time=0:15:00 --mem=8G --cpus-per-task=1 \
		--mail-type=END,FAIL --mail-user=milan_capoor@brown.edu \
		--wrap "set -e; source pytorch.venvsource/bin/activate
[ -f $DATASET/$VARIANT/config.json ] || python -u tools/derive_dataset.py $DATASET --name ${REL}/$VARIANT
python -u models/train_gnn.py select $OUT --dataset-out $DATASET/$VARIANT")
	echo "select: job $id, ranking in $OUT/summary.md, best predictions in $DATASET/$VARIANT/" >&2
	echo "Then compare: ./scripts/compare.sh ... MAPFAST-GNN=@$DATASET/$VARIANT" >&2
}

cd "$(dirname "$0")/.."
DATASET="${1:?usage: ./scripts/sweep_gnn.sh DATASET}"
DATASET="${DATASET%/}"
REL="${DATASET#datasets/}"
NAME="${REL//\//_}"
OUT="${OUT:-experiments/gnn_sweep/$REL}"
LOGS="${LOGS:-$OUT}"
BENCH=$(readlink -f "${BENCH:-$HOME/MAPF-Benchmarking}")
EPOCHS="${EPOCHS:-40}"
FOLDS="${FOLDS:-1}"
PARALLEL="${PARALLEL:-8}"
VARIANT="${VARIANT:-gnn}"
GRAPHS="$DATASET/graphs.pkl"

check_setup
mkdir -p "$OUT" "$LOGS"
source pytorch.venvsource/bin/activate
graphs_job=$(submit_graphs)
sweep_job=$(submit_sweep "$graphs_job")
submit_select "$sweep_job"
