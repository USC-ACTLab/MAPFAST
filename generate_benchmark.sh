#!/bin/bash
# Benchmark every solver on every map of MAPF Benchmarking, with both random and even scenarios and
# agent counts from 10 to 200, then build the MAPFAST and MAPFASTv2 datasets from the results
# (benchmark_dataset.sh: datasets/full_four_solvers and datasets/full_nine_solvers, augmented).
# usage, from the MAPFAST folder (it only submits jobs): ./generate_benchmark.sh
#
#   BENCH          MAPF Benchmarking folder (default ~/MAPF-Benchmarking)
#   AGENTS         agent counts (default 10 20 ... 100 120 ... 200); a map whose scenario files have
#                  fewer agents runs the counts it has
#   BENCH_TIME     time limit of each map's job (default 24:00:00)
#   AUGMENTATION   augmented copies of each training instance (default 6, every flip and rotation)
#
# This is a large job: for each scenario type, one 12-CPU job per map runs all nine solvers with a 60 s
# limit on 25 scenario files per agent count. Run MAPF Benchmarking's ./test_cluster.sh first.

set -e
cd "$(dirname "$0")"
mkdir -p logs

BENCH=$(readlink -f "${BENCH:-$HOME/MAPF-Benchmarking}")
AGENTS=(${AGENTS:-10 20 30 40 50 60 70 80 90 100 120 140 160 180 200})
BENCH_TIME="${BENCH_TIME:-24:00:00}"
[ -x "$BENCH/benchmark.sh" ] || { echo "no benchmark.sh in $BENCH (set BENCH)" >&2; exit 1; }

# the aggregate jobs that benchmark.sh submits run python from this environment
source "$BENCH/.venv/bin/activate"

# group the maps by the agent counts they can run: benchmark.py needs every scenario file
# of the type to have at least the largest count
declare -A groups
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
		for a in "${AGENTS[@]}"; do [ "$a" -le "$fewest" ] && counts+=("$a"); done
		[ ${#counts[@]} -gt 0 ] || { echo "$map ($type): fewer than ${AGENTS[0]} agents, skipped" >&2; continue; }
		key="$type|${counts[*]}"
		groups[$key]="${groups[$key]} $map"
	done
done

aggregates=()
for key in "${!groups[@]}"; do
	type=${key%%|*}
	counts=${key#*|}
	maps=${groups[$key]# }
	echo "== $type scenarios, agents $counts: $maps"
	out=$(SBATCH_TIMELIMIT="$BENCH_TIME" MAPS="$maps" "$BENCH/benchmark.sh" --scen-type "$type" --agents $counts)
	echo "$out"
	# benchmark.sh prints "Submitted batch job <id> (<map>)" per map, then the aggregate job's "Submitted batch job <id>"
	id=$(echo "$out" | grep -E '^Submitted batch job [0-9]+$' | tail -n 1 | awk '{print $4}')
	[ -n "$id" ] || { echo "no aggregate job submitted for $key" >&2; exit 1; }
	aggregates+=("$id")
done

dataset=$(sbatch --parsable --dependency="afterok:$(IFS=:; echo "${aggregates[*]}")" \
	--export=ALL,AUGMENTATION="${AUGMENTATION:-6}" \
	benchmark_dataset.sh "$BENCH" "${aggregates[@]}")

echo
echo "Aggregate jobs: ${aggregates[*]}"
echo "Dataset job:    $dataset (logs/benchmark_dataset_${dataset}.out), builds datasets/full_four_solvers and datasets/full_nine_solvers"
echo "Then train and compare: TRAIN_TIME=12:00:00 ./compare.sh datasets/full_four_solvers datasets/full_nine_solvers"
