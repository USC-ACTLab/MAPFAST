#!/bin/bash
# Benchmark every solver on every map of MAPF Benchmarking as one of its experiments (its benchmark.sh), and build the
# MAPFAST and MAPFASTv2 datasets from the results, with every training instance augmented:
#   datasets/<experiment>/four-solvers/   MAPFAST:   bcp2, cbs, cbsh2_rtc and reloc (in place of its BCP, CBS, CBSH and SAT)
#   datasets/<experiment>/nine-solvers/   MAPFASTv2: all nine solvers
# Both hold the same instances (those that one of the four MAPFAST solvers solves), so the two models are split,
# trained and tested on exactly the same problems.
#
# usage, from the MAPFAST folder (it only submits jobs):
#   ./scripts/generate_benchmark.sh <experiment> [tools/experiment.py options]   benchmark, then build the datasets
#   ./scripts/generate_benchmark.sh build <experiment>                         only build the datasets
# e.g. ./scripts/generate_benchmark.sh movingai_10-500-agents_300s --reuse movingai_10-200-agents_60s --retry
#
#   BENCH         MAPF Benchmarking folder (default ~/MAPF-Benchmarking)
#   TIMEOUT       time limit of a run, in seconds (default 300), which the datasets charge to unsolved runs
#   AGENTS        agent counts (default 10 20 ... 100 120 ... 200 250 ... 500; a scenario file runs those it has)
#   AUGMENTATION  augmented copies of each training instance (default 6, every flip and rotation)
# and benchmark.sh's SHARDS, SHARD_JOBS, SHARD_MEM and SHARD_TIME. The dataset job's log goes to experiments/dataset_build/.

set -e
MAPFAST_SOLVERS=(bcp2 cbs cbsh2_rtc reloc)
LOG_DIR=experiments/dataset_build

# Submits the dataset job (this script's build-job) for an experiment; with --after ID, once that job has ended
submit_build() {
	local after=()
	[ "$1" != --after ] || { after=(--dependency="afterok:$2"); shift 2; }
	local id
	id=$(sbatch --parsable -J benchmark_dataset "${after[@]}" --nodes=1 --cpus-per-task=16 --time=4:00:00 --mem=32G \
		-o "$LOG_DIR/%x_%j.out" -e "$LOG_DIR/%x_%j.err" --mail-type=END,FAIL --mail-user=milan_capoor@brown.edu \
		--export=ALL,BENCH="$BENCH",TIMEOUT="$TIMEOUT",AUGMENTATION="$AUGMENTATION" scripts/generate_benchmark.sh build-job "$1")
	echo "Datasets: job $id ($LOG_DIR/benchmark_dataset_$id.out) builds datasets/$1/four-solvers and datasets/$1/nine-solvers"
}

# Builds both datasets from the merged results of an experiment (runs in the dataset job)
build_datasets() {
	local results="$BENCH/experiments/$1"
	export MPLBACKEND=Agg
	[ -f "$results/results.csv" ] || { echo "no merged results in $results" >&2; exit 1; }
	python -u tools/benchmark_dataset.py "$results" --benchmarks "$BENCH/benchmarks" --name "$1/four-solvers" \
		--solvers "${MAPFAST_SOLVERS[@]}" --timeout "$TIMEOUT" --augmentation "$AUGMENTATION" --jobs "$SLURM_CPUS_PER_TASK"
	python -u tools/benchmark_dataset.py "$results" --benchmarks "$BENCH/benchmarks" --name "$1/nine-solvers" \
		--solved-by "${MAPFAST_SOLVERS[@]}" --timeout "$TIMEOUT" --augmentation "$AUGMENTATION" --jobs "$SLURM_CPUS_PER_TASK"
}

# a Slurm job runs a copy of this script, so it goes back to the folder it was submitted from
if [ "$1" = build-job ]; then cd "$SLURM_SUBMIT_DIR"; else cd "$(dirname "$0")/.."; fi
BENCH=$(readlink -f "${BENCH:-$HOME/MAPF-Benchmarking}")
TIMEOUT="${TIMEOUT:-300}"
AGENTS="${AGENTS:-10 20 30 40 50 60 70 80 90 100 120 140 160 180 200 250 300 350 400 450 500}"
AUGMENTATION="${AUGMENTATION:-6}"
mkdir -p "$LOG_DIR"
# MAPF Benchmarking's environment has what the dataset job needs (numpy, Pillow with matplotlib)
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
