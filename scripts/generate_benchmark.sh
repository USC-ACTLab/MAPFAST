#!/bin/bash
# Benchmark every solver on every map of MAPF Benchmarking, with both random and even scenarios and agent counts from
# 10 to 200, then build the MAPFAST and MAPFASTv2 datasets from the results, with every training instance augmented:
#   datasets/full_four_solvers/   MAPFAST:   bcp2, cbs, cbsh2_rtc and reloc (in place of its BCP, CBS, CBSH and SAT)
#   datasets/full_nine_solvers/   MAPFASTv2: all nine solvers of benchmark.sh
# Both hold the same instances (those that one of the four MAPFAST solvers solves), so the two models are split,
# trained and tested on exactly the same problems.
#
# usage, from the MAPFAST folder (it only submits jobs):
#   ./scripts/generate_benchmark.sh                   benchmark everything, then build the datasets once it has ended
#   ./scripts/generate_benchmark.sh build <id>...     only build the datasets, from benchmark.sh aggregate jobs that have
#                                                     ended (the "Aggregate jobs" an earlier run printed)
#
#   BENCH          MAPF Benchmarking folder (default ~/MAPF-Benchmarking)
#   AGENTS         agent counts (default 10 20 ... 100 120 ... 200); a map whose scenario files have
#                  fewer agents runs the counts it has
#   BENCH_TIME     time limit of each map's job (default 24:00:00)
#   AUGMENTATION   augmented copies of each training instance (default 6, every flip and rotation)
#
# This is a large job: for each scenario type, one 12-CPU job per map runs all nine solvers with a 60 s
# limit on 25 scenario files per agent count. Run MAPF Benchmarking's ./test_cluster.sh first.
# The dataset job's log goes to experiments/dataset_build/.

set -e
MAPFAST_SOLVERS=(bcp2 cbs cbsh2_rtc reloc)
LOG_DIR=experiments/dataset_build

# Submits one benchmark.sh run per group of maps that can run the same agent counts, and prints the ids of their
# aggregate jobs (which merge each group's results) on stdout; the progress goes to stderr
run_benchmarks() {
	local agents=(${AGENTS:-10 20 30 40 50 60 70 80 90 100 120 140 160 180 200})
	local bench_time="${BENCH_TIME:-24:00:00}"
	[ -x "$BENCH/benchmark.sh" ] || { echo "no benchmark.sh in $BENCH (set BENCH)" >&2; exit 1; }

	# group the maps by the agent counts they can run: benchmark.py needs every scenario file
	# of the type to have at least the largest count
	local -A groups
	local type map_dir map fewest scen n counts a key
	shopt -s nullglob
	for type in random even; do
		for map_dir in "$BENCH"/benchmarks/*/; do
			map=$(basename "$map_dir")
			compgen -G "$map_dir*.map" > /dev/null || continue
			fewest=""
			for scen in "$map_dir"*-"$type"-*.scen; do
				n=$(tail -n +2 "$scen" | grep -c . || true)
				[ -z "$fewest" ] || [ "$n" -lt "$fewest" ] && fewest=$n
			done
			[ -n "$fewest" ] || continue
			counts=()
			for a in "${agents[@]}"; do [ "$a" -le "$fewest" ] && counts+=("$a"); done
			[ ${#counts[@]} -gt 0 ] || { echo "$map ($type): fewer than ${agents[0]} agents, skipped" >&2; continue; }
			key="$type|${counts[*]}"
			groups[$key]="${groups[$key]} $map"
		done
	done

	local maps out id
	for key in "${!groups[@]}"; do
		type=${key%%|*}
		counts=${key#*|}
		maps=${groups[$key]# }
		echo "== $type scenarios, agents $counts: $maps" >&2
		out=$(SBATCH_TIMELIMIT="$bench_time" MAPS="$maps" "$BENCH/benchmark.sh" --scen-type "$type" --agents $counts)
		echo "$out" >&2
		# benchmark.sh prints "Submitted batch job <id> (<map>)" per map, then the aggregate job's "Submitted batch job <id>"
		id=$(echo "$out" | grep -E '^Submitted batch job [0-9]+$' | tail -n 1 | awk '{print $4}')
		[ -n "$id" ] || { echo "no aggregate job submitted for $key" >&2; exit 1; }
		echo "$id"
	done
}

# Submits the job that builds the datasets (this script's build_datasets) from the given aggregate jobs; with --wait,
# once they have ended (without it they must have ended already, and Slurm may no longer know them)
submit_build() {
	local after=()
	if [ "$1" = --wait ]; then
		shift
		after=(--dependency="afterok:$(IFS=:; echo "$*")")
	fi
	mkdir -p "$LOG_DIR"
	local id
	id=$(sbatch --parsable -J benchmark_dataset "${after[@]}" \
		--nodes=1 --cpus-per-task=16 --time=4:00:00 --mem=32G \
		-o "$LOG_DIR/%x_%j.out" -e "$LOG_DIR/%x_%j.err" \
		--mail-type=END,FAIL --mail-user=milan_capoor@brown.edu \
		--export=ALL,BENCH="$BENCH",AUGMENTATION="${AUGMENTATION:-6}" \
		scripts/generate_benchmark.sh build-job "$@")
	echo
	echo "Aggregate jobs: $*"
	echo "Dataset job:    $id ($LOG_DIR/benchmark_dataset_$id.out), builds datasets/full_four_solvers and datasets/full_nine_solvers"
	echo "Then train and compare: TRAIN_TIME=12:00:00 ./scripts/compare.sh MAPFAST=datasets/full_four_solvers MAPFASTv2=datasets/full_nine_solvers"
}

# Merges the given aggregate runs into $BENCH/output/full_<jobid>_*, then builds both datasets from it (runs in the
# dataset job)
build_datasets() {
	local name="full_$SLURM_JOB_ID"
	export MPLBACKEND=Agg

	echo "Merging benchmark runs $* into $BENCH/output/${name}_*..."
	(cd "$BENCH" && python -u tools/aggregate.py "$@" --name "$name")

	echo "Building datasets/full_four_solvers (MAPFAST)..."
	python -u tools/benchmark_dataset.py "$BENCH/output/$name" --benchmarks "$BENCH/benchmarks" \
		--name full_four_solvers --solvers "${MAPFAST_SOLVERS[@]}" \
		--augmentation "${AUGMENTATION:-6}" --jobs "$SLURM_CPUS_PER_TASK"

	echo "Building datasets/full_nine_solvers (MAPFASTv2)..."
	python -u tools/benchmark_dataset.py "$BENCH/output/$name" --benchmarks "$BENCH/benchmarks" \
		--name full_nine_solvers --solved-by "${MAPFAST_SOLVERS[@]}" \
		--augmentation "${AUGMENTATION:-6}" --jobs "$SLURM_CPUS_PER_TASK"

	echo "Train and compare both with: TRAIN_TIME=12:00:00 ./scripts/compare.sh MAPFAST=datasets/full_four_solvers MAPFASTv2=datasets/full_nine_solvers"
}

# run from the repository folder, whose paths (datasets/, tools/, ...) every step uses; a Slurm job runs a copy of
# this script, so it goes back to the folder it was submitted from instead
if [ "$1" = build-job ]; then
	cd "$SLURM_SUBMIT_DIR"
else
	cd "$(dirname "$0")/.."
fi
BENCH=$(readlink -f "${BENCH:-$HOME/MAPF-Benchmarking}")
# MAPF Benchmarking's environment has what every step needs: the aggregate jobs that benchmark.sh submits run python
# from it, and so does the dataset job (numpy, Pillow with matplotlib)
source "$BENCH/.venv/bin/activate"

case "$1" in
	"")
		aggregates=($(run_benchmarks))
		[ ${#aggregates[@]} -gt 0 ] || { echo "no benchmark runs submitted" >&2; exit 1; }
		submit_build --wait "${aggregates[@]}"
		;;
	build)
		shift
		[ $# -gt 0 ] || { echo "usage: ./scripts/generate_benchmark.sh build <aggregate job id>..." >&2; exit 1; }
		submit_build "$@"
		;;
	build-job)
		shift
		build_datasets "$@"
		;;
	*)
		echo "usage: ./scripts/generate_benchmark.sh [build <aggregate job id>...]" >&2
		exit 1
		;;
esac
