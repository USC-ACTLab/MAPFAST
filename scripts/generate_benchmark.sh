#!/bin/bash
# Benchmark every solver on every map of MAPF Benchmarking, with both random and even scenarios, and build the MAPFAST
# and MAPFASTv2 datasets from the results, with every training instance augmented:
#   datasets/<experiment>/four-solvers/   MAPFAST:   bcp2, cbs, cbsh2_rtc and reloc (in place of its BCP, CBS, CBSH and SAT)
#   datasets/<experiment>/nine-solvers/   MAPFASTv2: all nine solvers of benchmark.sh
# named after the experiment of MAPF Benchmarking they come from, e.g. movingai_10-500-agents_300s.
# Both hold the same instances (those that one of the four MAPFAST solvers solves), so the two models are split,
# trained and tested on exactly the same problems.
#
# It goes in two passes, each an experiment of MAPF Benchmarking (experiments/<name>/ there, with its results and
# Slurm logs), to keep the cost of solvers that cannot keep up bounded:
#   1. benchmark.sh runs every solver with a short time limit (INITIAL_TIMEOUT, 60 s) on AGENTS (10 to 200 agents),
#      into the experiment movingai_10-200-agents_60s;
#   2. MAPF Benchmarking's tools/extend_benchmark.py reruns every run that pass 1 did not solve with the full time limit
#      (TIMEOUT, 300 s) and adds EXTRA_AGENTS (250 to 500 agents), going up in agent count per scenario file and
#      solver and stopping at the solver's first failure there (larger counts then count as failures), into the
#      experiment movingai_10-500-agents_300s. It runs as a Slurm job array of SHARDS jobs.
# The datasets are labelled with TIMEOUT, which analysis.py charges to unsolved runs.
#
# usage, from the MAPFAST folder (it only submits jobs):
#   ./scripts/generate_benchmark.sh                   both passes, then build the datasets
#   ./scripts/generate_benchmark.sh extend <experiment>   pass 2 and the datasets, from a finished pass 1 (an experiment
#                                                     of MAPF Benchmarking, e.g. movingai_10-200-agents_60s); submitting
#                                                     it again with the same settings resumes it, skipping the work done
#   ./scripts/generate_benchmark.sh build <experiment>    only build the datasets, from a finished experiment
#
#   BENCH            MAPF Benchmarking folder (default ~/MAPF-Benchmarking)
#   AGENTS           agent counts of pass 1 (default 10 20 ... 100 120 ... 200); a map whose scenario files have
#                    fewer agents runs the counts it has
#   EXTRA_AGENTS     agent counts pass 2 adds (default 250 300 ... 500), on the scenario files with that many agents
#   INITIAL_TIMEOUT  time limit of pass 1, in seconds (default 60)
#   TIMEOUT          time limit of pass 2 and of the datasets, in seconds (default 300)
#   BENCH_TIME       time limit of each map's job in pass 1 (default 24:00:00)
#   SHARDS           jobs of pass 2 (default 20, 12 CPUs each); SHARD_TIME their time limit (default 24:00:00)
#   PASS1, PASS2     names of the two experiments (default movingai_<agents>_<time limit>s)
#   AUGMENTATION     augmented copies of each training instance (default 6, every flip and rotation)
#
# This is a large job. The dataset job's log goes to experiments/dataset_build/ here, the benchmark's to MAPF
# Benchmarking's experiments/<name>/logs/.

set -e
MAPFAST_SOLVERS=(bcp2 cbs cbsh2_rtc reloc)
LOG_DIR=experiments/dataset_build

# Submits one benchmark.sh run per group of maps that can run the same agent counts, all into the experiment
# $PASS1, and prints the ids of their map jobs on stdout; the progress goes to stderr
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
		# benchmark.sh's own --timeout comes first, so this one replaces it; one merge job for all groups (submit_merge)
		out=$(SBATCH_TIMELIMIT="$bench_time" MAPS="$maps" EXPERIMENT="$PASS1" AGGREGATE=0 "$BENCH/benchmark.sh" \
			--scen-type "$type" --agents $counts --timeout "$INITIAL_TIMEOUT")
		echo "$out" >&2
		# benchmark.sh prints "Submitted batch job <id> (<map>)" per map
		ids=$(echo "$out" | grep -E '^Submitted batch job [0-9]+ \(' | awk '{print $4}')
		[ -n "$ids" ] || { echo "no map jobs submitted for $key" >&2; exit 1; }
		echo "$ids"
	done
}

# Submits the job that merges the maps of experiment NAME, once the given map jobs have ended (whether or not they
# finished, as benchmark.sh does); prints its id. Usage: submit_merge NAME MAP_JOB_ID...
submit_merge() {
	local name=$1 logs="$BENCH/experiments/$1/logs"
	shift
	sbatch --parsable -J mapf_benchmark_merge --dependency="afterany:$(IFS=:; echo "$*")" \
		--nodes=1 --cpus-per-task=1 --time=1:00:00 --mem=16G \
		-o "$logs/merge.out" -e "$logs/merge.err" --open-mode=append \
		--wrap "cd '$BENCH' && source .venv/bin/activate && MPLBACKEND=Agg python -u tools/aggregate.py $name"
}

# Submits pass 2 on the experiment SOURCE into the experiment $PASS2 (a job array, then the job that merges its
# shards), after job AFTER if given; prints the merge job's id. Usage: submit_extend SOURCE [AFTER]
submit_extend() {
	local source=$1 after=() name="$PASS2" logs="$BENCH/experiments/$PASS2/logs" array merge
	[ -z "$2" ] || after=(--dependency="afterok:$2")
	mkdir -p "$logs"
	local options="$source --timeout $TIMEOUT --agents $EXTRA_AGENTS --name $name"
	local setup="cd '$BENCH' && module load boost yaml-cpp gurobi && source .venv/bin/activate && export MPLBACKEND=Agg"
	array=$(sbatch --parsable -J benchmark_extend "${after[@]}" --array="0-$((SHARDS - 1))" \
		--nodes=1 --cpus-per-task=12 --time="$SHARD_TIME" --mem=32G \
		-o "$logs/shard%a.out" -e "$logs/shard%a.err" --open-mode=append \
		--wrap "$setup && python -u tools/extend_benchmark.py run $options --shard \$SLURM_ARRAY_TASK_ID --shards $SHARDS --jobs 12")
	merge=$(sbatch --parsable -J benchmark_extend_merge --dependency="afterok:$array" \
		--nodes=1 --cpus-per-task=1 --time=2:00:00 --mem=16G \
		-o "$logs/merge.out" -e "$logs/merge.err" --open-mode=append \
		--wrap "$setup && python -u tools/extend_benchmark.py merge $options")
	echo "Pass 2:         job array $array ($SHARDS shards), merged by job $merge into $BENCH/experiments/$name/" >&2
	echo "                if a shard runs out of time, submit the same command again: it resumes where the shards stopped" >&2
	echo "$merge"
}

# Submits the job that builds the datasets (this script's build_datasets) from an experiment of MAPF Benchmarking;
# with --after ID, once that job has ended
submit_build() {
	local after=()
	if [ "$1" = --after ]; then
		after=(--dependency="afterok:$2")
		shift 2
	fi
	local id
	id=$(sbatch --parsable -J benchmark_dataset "${after[@]}" \
		--nodes=1 --cpus-per-task=16 --time=4:00:00 --mem=32G \
		-o "$LOG_DIR/%x_%j.out" -e "$LOG_DIR/%x_%j.err" \
		--mail-type=END,FAIL --mail-user=milan_capoor@brown.edu \
		--export=ALL,BENCH="$BENCH",TIMEOUT="$TIMEOUT",AUGMENTATION="${AUGMENTATION:-6}" \
		scripts/generate_benchmark.sh build-job "$@")
	echo "Datasets:       job $id ($LOG_DIR/benchmark_dataset_$id.out) builds datasets/$1/four-solvers and datasets/$1/nine-solvers" >&2
}

# Builds both datasets from the merged results of an experiment of MAPF Benchmarking (runs in the dataset job)
build_datasets() {
	local results="$BENCH/experiments/$1"
	export MPLBACKEND=Agg
	[ -f "$results/results.csv" ] || { echo "no merged results in $results" >&2; exit 1; }

	echo "Building datasets/$1/four-solvers (MAPFAST)..."
	python -u tools/benchmark_dataset.py "$results" --benchmarks "$BENCH/benchmarks" \
		--name "$1/four-solvers" --solvers "${MAPFAST_SOLVERS[@]}" --timeout "$TIMEOUT" \
		--augmentation "${AUGMENTATION:-6}" --jobs "$SLURM_CPUS_PER_TASK"

	echo "Building datasets/$1/nine-solvers (MAPFASTv2)..."
	python -u tools/benchmark_dataset.py "$results" --benchmarks "$BENCH/benchmarks" \
		--name "$1/nine-solvers" --solved-by "${MAPFAST_SOLVERS[@]}" --timeout "$TIMEOUT" \
		--augmentation "${AUGMENTATION:-6}" --jobs "$SLURM_CPUS_PER_TASK"

	echo "Train and compare both with: TRAIN_TIME=12:00:00 ./scripts/compare.sh MAPFAST=datasets/$1/four-solvers MAPFASTv2=datasets/$1/nine-solvers"
}

# run from the repository folder, whose paths (datasets/, tools/, ...) every step uses; a Slurm job runs a copy of
# this script, so it goes back to the folder it was submitted from instead
if [ "$1" = build-job ]; then
	cd "$SLURM_SUBMIT_DIR"
else
	cd "$(dirname "$0")/.."
fi
BENCH=$(readlink -f "${BENCH:-$HOME/MAPF-Benchmarking}")
INITIAL_TIMEOUT="${INITIAL_TIMEOUT:-60}"
TIMEOUT="${TIMEOUT:-300}"
EXTRA_AGENTS="${EXTRA_AGENTS-250 300 350 400 450 500}"
SHARDS="${SHARDS:-20}"
SHARD_TIME="${SHARD_TIME:-24:00:00}"
# the experiments of the two passes in MAPF Benchmarking, named after what they hold
agents_first=(${AGENTS:-10})
extra=($EXTRA_AGENTS)
PASS1="${PASS1:-movingai_${agents_first[0]}-$(echo ${AGENTS:-200} | awk '{print $NF}')-agents_${INITIAL_TIMEOUT%.*}s}"
PASS2="${PASS2:-movingai_${agents_first[0]}-${extra[-1]:-$(echo ${AGENTS:-200} | awk '{print $NF}')}-agents_${TIMEOUT%.*}s}"
mkdir -p "$LOG_DIR"
# MAPF Benchmarking's environment has what every step needs: the merge jobs run python
# from it, and so does the dataset job (numpy, Pillow with matplotlib)
source "$BENCH/.venv/bin/activate"

case "$1" in
	"")
		for pass in "$PASS1" "$PASS2"; do
			[ ! -e "$BENCH/experiments/$pass" ] || { echo "$BENCH/experiments/$pass exists: set PASS1 and PASS2 to new names" >&2; exit 1; }
		done
		map_jobs=($(run_benchmarks))
		[ ${#map_jobs[@]} -gt 0 ] || { echo "no benchmark runs submitted" >&2; exit 1; }
		merge=$(submit_merge "$PASS1" "${map_jobs[@]}")
		echo "Pass 1:         ${#map_jobs[@]} map jobs, merged by job $merge into $BENCH/experiments/$PASS1/" >&2
		extended=$(submit_extend "$PASS1" "$merge")
		submit_build --after "$extended" "$PASS2"
		;;
	extend)
		[ -n "$2" ] || { echo "usage: ./scripts/generate_benchmark.sh extend <experiment of MAPF Benchmarking>" >&2; exit 1; }
		[ -f "$BENCH/experiments/$2/results.csv" ] || { echo "no $BENCH/experiments/$2/results.csv" >&2; exit 1; }
		(cd "$BENCH" && MPLBACKEND=Agg python tools/extend_benchmark.py plan "$2" --timeout "$TIMEOUT" --agents $EXTRA_AGENTS) >&2
		extended=$(submit_extend "$2")
		submit_build --after "$extended" "$PASS2"
		;;
	build)
		shift
		[ $# -eq 1 ] || { echo "usage: ./scripts/generate_benchmark.sh build <experiment of MAPF Benchmarking>" >&2; exit 1; }
		submit_build "$1"
		;;
	build-job)
		shift
		build_datasets "$@"
		;;
	*)
		echo "usage: ./scripts/generate_benchmark.sh [extend <experiment> | build <experiment>]" >&2
		exit 1
		;;
esac
