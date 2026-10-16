#!/bin/bash
# Train, test and analyse several models with train.sh (one job each), run the general-purpose model baseline (Jev, see
# models/jev_baseline.py) on their test instances, then plot them all on the same graphs in
# $OUT/<name>_vs_<name>..._<jobid>/, where OUT is the round's experiments folder (default experiments/round_2).
# Each model is given as one of
#   NAME=DATASET          train DATASET with train.sh
#   NAME=@DATASET         only plot DATASET, whose predictions already exist
# The default is this round's comparison (see experiments/EXPERIMENTS.md): MAPFAST (four solvers) and MAPFASTv2 (eight
# solvers, without mapf_prioritized_sipp), each without and with the auxiliary outputs, all with the relative speed loss.
# Make their datasets first, from the repository folder:
#   python tools/derive_dataset.py datasets/full_four_solvers --name full_four_solvers_score
#   python tools/derive_dataset.py datasets/full_four_solvers --name full_four_solvers_score_aux --aux
#   python tools/derive_dataset.py datasets/full_nine_solvers --name full_eight_solvers_score --drop mapf_prioritized_sipp
#   python tools/derive_dataset.py datasets/full_nine_solvers --name full_eight_solvers_score_aux --drop mapf_prioritized_sipp --aux
#
# Jev baseline: for every solver portfolio among the trained models, and every K of JEV_EXAMPLES, one job asks Jev for
# the fastest solver of each test instance of the first model with that portfolio (once that model has been tested), with
# K training instances of the same map per request (0: zero-shot). Its predictions go to datasets/<dataset>_jev<K>/ and it
# is plotted as Jev<K>_<model name>. It needs TYPESAFE_API_KEY; JEV_EXAMPLES= (empty) leaves the baseline out.
#
# Logs and graphs are named after the models, e.g. $OUT/MAPFAST_<jobid>.out and $OUT/MAPFASTv2_<jobid>_plots/, and the
# comparison job also copies each model's config and test predictions to $OUT/predictions/<name>/.
# usage, from the repository folder: ./scripts/compare.sh [NAME=DATASET ...]
#   e.g. TRAIN_TIME=12:00:00 ./scripts/compare.sh MAPFAST=datasets/full_four_solvers MAPFASTv2=datasets/full_nine_solvers
#   TRAIN_TIME overrides the time limit of the training jobs (train.sh's own otherwise)
#   JEV_EXAMPLES are the numbers of examples per Jev request to run the baseline with (default "0 20")
#   JEV_BUDGET is the most USD each Jev job may spend (default 3)
#   OUT is the folder for this round's logs, graphs and predictions (default experiments/round_2)
# At most eight models and baselines can be plotted together.
#
# This script only submits jobs. If the virtual environment does not exist yet, the other jobs wait for the
# first one, so that only it creates the environment; otherwise all of them train at once.

set -e

DEFAULT_RUNS=(
	MAPFAST=datasets/full_four_solvers_score
	MAPFASTv2=datasets/full_eight_solvers_score
	MAPFAST_aux=datasets/full_four_solvers_score_aux
	MAPFASTv2_aux=datasets/full_eight_solvers_score_aux
)
MAX_PLOTTED=8

# Filled in by the submit_* functions and read by submit_comparison: the job ids to wait for, the names of the plotted
# runs, the --run arguments of plots.py, and the commands that copy each run's predictions into $OUT/predictions/
job_ids=()
names=()
plot_args=""
copy_args=""
# dataset -> id of the job that trains and tests it
declare -A trained

# Exits unless every run's dataset has a config.json and, if the Jev baseline runs, TYPESAFE_API_KEY is set
check_runs() {
	local run dataset
	for run in "$@"; do
		dataset=${run#*=}
		[ -f "${dataset#@}/config.json" ] || { echo "no config.json in ${dataset#@} (see the top of this script)" >&2; exit 1; }
	done
	if [ -n "$JEV_EXAMPLES" ] && [ -z "$TYPESAFE_API_KEY" ]; then
		echo "set TYPESAFE_API_KEY for the Jev baseline, or JEV_EXAMPLES= to leave it out" >&2
		exit 1
	fi
}

# Prints the Jev baselines for the given runs, one NAME=K:DATASET per line: one per portfolio (the solvers of the
# dataset's mapping) and number of examples, on the first trained model with that portfolio, as models with the same
# portfolio are tested on the same instances
plan_jev_runs() {
	local -A seen
	local run dataset portfolio k
	for run in "$@"; do
		dataset=${run#*=}
		[[ $dataset == @* ]] && continue
		portfolio=$(python3 -c "import json, sys; print(','.join(sorted(json.load(open(sys.argv[1]))['Analysis']['mapping'])))" "$dataset/config.json")
		[ -z "${seen[$portfolio]}" ] || continue
		seen[$portfolio]=1
		for k in $JEV_EXAMPLES; do
			echo "Jev${k}_${run%%=*}=$k:$dataset"
		done
	done
}

# Adds a run to the comparison: plotted from DATASET's predictions (with LOG, or none, for the loss graph), which the
# comparison job then copies to $OUT/predictions/NAME/
add_to_comparison() {
	local name=$1 dataset=$2 log_file=$3
	names+=("$name")
	plot_args+=" --run $name $dataset $log_file"
	copy_args+=" && mkdir -p $OUT/predictions/$name && cp $dataset/config.json $dataset/y_pred_true_*.json $OUT/predictions/$name/"
}

# Submits train.sh for each run (NAME=DATASET), or only plots a NAME=@DATASET run. If the virtual environment does
# not exist yet, the other jobs wait for the first one, so that only it creates the environment
submit_training() {
	local after_first=() time_limit=()
	[ -z "$TRAIN_TIME" ] || time_limit=(--time="$TRAIN_TIME")
	local run name dataset id
	for run in "$@"; do
		name=${run%%=*}
		dataset=${run#*=}
		if [[ $dataset == @* ]]; then
			add_to_comparison "$name" "${dataset#@}" none
			echo "$name (${dataset#@}): plotted from its existing predictions"
			continue
		fi
		id=$(sbatch --parsable -J "$name" "${log[@]}" --export=ALL,LOG_DIR="$OUT" "${time_limit[@]}" "${after_first[@]}" \
			scripts/train.sh "$dataset")
		if [ ${#job_ids[@]} -eq 0 ] && ! [ -f pytorch.venvsource/bin/activate ]; then
			after_first=(--dependency="afterany:$id")
		fi
		job_ids+=("$id")
		trained[$dataset]=$id
		add_to_comparison "$name" "$dataset" "$OUT/${name}_${id}.out"
		echo "$name ($dataset): job $id, log $OUT/${name}_${id}.out, graphs in $OUT/${name}_${id}_plots/"
	done
}

# Submits a Jev baseline job for each NAME=K:DATASET, once DATASET's training job has tested its model: the test
# instances are those of the model's predictions (main.py -T 0), and Jev's go to datasets/<dataset>_jev<K>/
submit_jev() {
	local run name spec examples dataset jev_dataset id
	for run in "$@"; do
		name=${run%%=*}
		spec=${run#*=}
		examples=${spec%%:*}
		dataset=${spec#*:}
		jev_dataset="datasets/$(basename "$dataset")_jev$examples"
		id=$(sbatch --parsable -J "$name" "${log[@]}" --dependency="afterok:${trained[$dataset]}" \
			--time=2:00:00 --mem=8G --cpus-per-task=2 \
			--wrap "set -e; source pytorch.venvsource/bin/activate
predictions=\$(python -c \"import sys; sys.path.insert(0, 'models'); from utils import read_config; print(read_config('$dataset')['Testing']['prediction_output'])\")
python -u tools/derive_dataset.py $dataset --name $(basename "$jev_dataset")
python -u models/jev_baseline.py $jev_dataset --test-from \"\$predictions\" --examples $examples --budget ${JEV_BUDGET:-3} --yes
echo 'Running Analysis...'
python -u tools/analysis.py -C $jev_dataset")
		job_ids+=("$id")
		add_to_comparison "$name" "$jev_dataset" none
		echo "$name (Jev baseline, $examples examples, on the test instances of $dataset): job $id, log $OUT/${name}_${id}.out, predictions in $jev_dataset/"
	done
}

# Submits the job that plots every run on the same graphs once all the other jobs have ended, and copies their
# predictions to $OUT/predictions/
submit_comparison() {
	local comparison after=() id
	comparison=$(printf '%s_vs_' "${names[@]}")
	comparison=${comparison%_vs_}
	[ ${#job_ids[@]} -eq 0 ] || after=(--dependency="afterok:$(IFS=:; echo "${job_ids[*]}")")
	id=$(sbatch --parsable -J "$comparison" "${after[@]}" \
		--time=0:15:00 --mem=4G --cpus-per-task=1 \
		"${log[@]}" \
		--mail-type=END,FAIL --mail-user=milan_capoor@brown.edu \
		--wrap "source pytorch.venvsource/bin/activate && python -u tools/plots.py $plot_args -o $OUT/${comparison}_\$SLURM_JOB_ID $copy_args")
	echo "Comparison: job $id, log $OUT/${comparison}_${id}.out, graphs in $OUT/${comparison}_${id}/"
}

# run from the repository folder, whose paths (datasets/, models/, tools/, ...) every job uses
cd "$(dirname "$0")/.."
OUT="${OUT:-experiments/round_2}"
JEV_EXAMPLES="${JEV_EXAMPLES-0 20}"
log=(-o "$OUT/%x_%j.out" -e "$OUT/%x_%j.err")

runs=("$@")
[ ${#runs[@]} -gt 0 ] || runs=("${DEFAULT_RUNS[@]}")
check_runs "${runs[@]}"
mapfile -t jev_runs < <(plan_jev_runs "${runs[@]}")
total=$(( ${#runs[@]} + ${#jev_runs[@]} ))
[ $total -le $MAX_PLOTTED ] || { echo "$total models and baselines: at most $MAX_PLOTTED can be plotted together (fewer models, or fewer JEV_EXAMPLES)" >&2; exit 1; }

mkdir -p "$OUT"
submit_training "${runs[@]}"
submit_jev "${jev_runs[@]}"
submit_comparison
