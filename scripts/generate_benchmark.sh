#!/bin/bash
# Benchmark with MAPF Benchmarking's benchmark.sh, then build datasets/<experiment>/{four,ten}-solvers (same instances).
# usage: ./scripts/generate_benchmark.sh <experiment> [experiment.py options] | build <experiment>
#   (env: BENCH, TIMEOUT, AGENTS, AUGMENTATION, and benchmark.sh's SHARD_*)

set -e
MAPFAST_SOLVERS=(bcp2 cbs cbsh2_rtc reloc)
# MAPFASTv2: all but mapf_prioritized_sipp
V2_SOLVERS=(bcp1 bcp2 cbs cbsh2_rtc cbsh2_rtc_bp cbsh2_rtc_ch cbsh2_rtc_chbp ecbs lazycbs reloc)
LOG_DIR=experiments/dataset_build

# submit_build [--after JOB] EXPERIMENT
submit_build() {
	local after=()
	[ "$1" != --after ] || { after=(--dependency="afterok:$2"); shift 2; }
	local id
	id=$(sbatch --parsable -J benchmark_dataset "${after[@]}" --nodes=1 --cpus-per-task=16 --time=4:00:00 --mem=32G \
		-o "$LOG_DIR/%x_%j.out" -e "$LOG_DIR/%x_%j.err" --mail-type=END,FAIL --mail-user=milan_capoor@brown.edu \
		--export=ALL,BENCH="$BENCH",TIMEOUT="$TIMEOUT",AUGMENTATION="$AUGMENTATION" scripts/generate_benchmark.sh build-job "$1")
	echo "Datasets: job $id ($LOG_DIR/benchmark_dataset_$id.out) builds datasets/$1/four-solvers and datasets/$1/ten-solvers"
}

build_datasets() {
	local results="$BENCH/experiments/$1"
	export MPLBACKEND=Agg
	[ -f "$results/results.csv" ] || { echo "no merged results in $results" >&2; exit 1; }
	python -u tools/benchmark_dataset.py "$results" --benchmarks "$BENCH/benchmarks" --name "$1/four-solvers" \
		--solvers "${MAPFAST_SOLVERS[@]}" --timeout "$TIMEOUT" --augmentation "$AUGMENTATION" --jobs "$SLURM_CPUS_PER_TASK"
	python -u tools/benchmark_dataset.py "$results" --benchmarks "$BENCH/benchmarks" --name "$1/ten-solvers" \
		--solvers "${V2_SOLVERS[@]}" --solved-by "${MAPFAST_SOLVERS[@]}" --timeout "$TIMEOUT" --augmentation "$AUGMENTATION" --jobs "$SLURM_CPUS_PER_TASK"
}

# Slurm runs a copy of this script, so return to the submit folder
if [ "$1" = build-job ]; then cd "$SLURM_SUBMIT_DIR"; else cd "$(dirname "$0")/.."; fi
BENCH=$(readlink -f "${BENCH:-$HOME/MAPF-Benchmarking}")
TIMEOUT="${TIMEOUT:-300}"
AGENTS="${AGENTS:-10 20 30 40 50 60 70 80 90 100 120 140 160 180 200 250 300 350 400 450 500}"
AUGMENTATION="${AUGMENTATION:-6}"
mkdir -p "$LOG_DIR"
source "$BENCH/.venv/bin/activate"

case "$1" in
	build) submit_build "$2" ;;
	build-job) build_datasets "$2" ;;
	""|-*) echo "usage: ./scripts/generate_benchmark.sh <experiment> [options] | build <experiment>" >&2; exit 1 ;;
	*)
		name=$1
		shift
		out=$(EXPERIMENT="$name" "$BENCH/benchmark.sh" --timeout "$TIMEOUT" --agents $AGENTS "$@")
		echo "$out"
		submit_build --after "$(sed -n 's/^Merge: job \([0-9]*\).*/\1/p' <<< "$out")" "$name"
		;;
esac
