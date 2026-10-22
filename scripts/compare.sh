#!/bin/bash
# Train and test several models with train.sh (one job each), run the Jev baseline (models/jev_baseline.py) on their
# test instances, then plot them all together in $OUT/<name>_vs_<name>..._<jobid>/. Each model is one of
#   NAME=DATASET     train DATASET with train.sh
#   NAME=@DATASET    only plot DATASET, whose predictions exist (e.g. MAPFAST-GNN's, from sweep_gnn.sh)
# For every solver portfolio among the trained models and every K of JEV_EXAMPLES (default 20; empty: no baseline),
# one job asks Jev about the test instances of the first model with that portfolio, with K training instances of the
# same map per request, into the variant <dataset>/jev<K>/, plotted as Jev<K>_<model>. It needs TYPESAFE_API_KEY.
# Logs go to $OUT/<name>_<jobid>.out, and the comparison job copies every model's config and predictions to
# $OUT/predictions/<name>/.
#
# usage, from the repository folder: OUT=experiments/round_N ./scripts/compare.sh NAME=DATASET ...
#   TRAIN_TIME   time limit of the training jobs (default: train.sh's)
#   JEV_BUDGET   most USD per Jev job (default 3)
# At most eight models and baselines can be plotted together. This script only submits jobs.

set -e
cd "$(dirname "$0")/.."
[ -n "$OUT" ] && [ $# -gt 0 ] || { echo "usage: OUT=experiments/round_N ./scripts/compare.sh NAME=DATASET ..." >&2; exit 1; }
JEV_EXAMPLES="${JEV_EXAMPLES-20}"
[ -z "$JEV_EXAMPLES" ] || [ -n "$TYPESAFE_API_KEY" ] || { echo "set TYPESAFE_API_KEY, or JEV_EXAMPLES= for no baseline" >&2; exit 1; }
mkdir -p "$OUT"
log=(-o "$OUT/%x_%j.out" -e "$OUT/%x_%j.err")
time_limit=()
[ -z "$TRAIN_TIME" ] || time_limit=(--time="$TRAIN_TIME")
config() { python3 -c "import json, sys; print(json.load(open(sys.argv[1] + '/config.json'))[sys.argv[2]])" "$@"; }

declare -A portfolios
for run in "$@"; do
	dataset=${run#*=}
	[ -f "${dataset#@}/config.json" ] || { echo "no config.json in ${dataset#@}" >&2; exit 1; }
	[[ $dataset == @* ]] || portfolios[$(config "$dataset" mapping)]=1
done
(( $# + ${#portfolios[@]} * $(wc -w <<< "$JEV_EXAMPLES") <= 8 )) || { echo "at most 8 models and baselines can be plotted together" >&2; exit 1; }
portfolios=()

jobs=() names=() plot_args="" copy_args="" after_first=()
# a run to plot: NAME DATASET LOG
plot() {
	names+=("$1")
	plot_args+=" --run $1 $2 $3"
	copy_args+=" && mkdir -p $OUT/predictions/$1 && cp $2/config.json $2/y_pred_true_*.json $OUT/predictions/$1/"
}

for run in "$@"; do
	name=${run%%=*} dataset=${run#*=}
	if [[ $dataset == @* ]]; then
		plot "$name" "${dataset#@}" none
		continue
	fi
	id=$(sbatch --parsable -J "$name" "${log[@]}" --export=ALL,LOG_DIR="$OUT" "${time_limit[@]}" "${after_first[@]}" \
		scripts/train.sh "$dataset")
	# only the first job creates the virtual environment
	[ -f pytorch.venvsource/bin/activate ] || [ ${#jobs[@]} -gt 0 ] || after_first=(--dependency="afterany:$id")
	jobs+=("$id")
	plot "$name" "$dataset" "$OUT/${name}_${id}.out"
	echo "$name ($dataset): job $id"

	portfolio=$(config "$dataset" mapping)
	[ -z "${portfolios[$portfolio]}" ] || continue
	portfolios[$portfolio]=1
	for k in $JEV_EXAMPLES; do
		jev="$dataset/jev$k"
		jev_id=$(sbatch --parsable -J "Jev${k}_$name" "${log[@]}" --dependency="afterok:$id" --time=2:00:00 --mem=8G \
			--cpus-per-task=2 --wrap "set -e; source pytorch.venvsource/bin/activate
python -u tools/derive_dataset.py $dataset --name ${jev#datasets/}
python -u models/jev_baseline.py $jev --test-from $dataset/$(config "$dataset" prediction_output) --examples $k --budget ${JEV_BUDGET:-3} --yes")
		jobs+=("$jev_id")
		plot "Jev${k}_$name" "$jev" none
		echo "Jev${k}_$name (Jev, $k examples, on the test instances of $dataset): job $jev_id"
	done
done

comparison=$(printf '%s_vs_' "${names[@]}")
comparison=${comparison%_vs_}
after=()
[ ${#jobs[@]} -eq 0 ] || after=(--dependency="afterok:$(IFS=:; echo "${jobs[*]}")")
id=$(sbatch --parsable -J "$comparison" "${after[@]}" --time=0:15:00 --mem=4G --cpus-per-task=1 "${log[@]}" \
	--mail-type=END,FAIL --mail-user=milan_capoor@brown.edu \
	--wrap "source pytorch.venvsource/bin/activate && python -u tools/plots.py $plot_args -o $OUT/${comparison}_\$SLURM_JOB_ID $copy_args")
echo "Comparison: job $id, graphs in $OUT/${comparison}_${id}/"
