#!/bin/bash
# Build the MAPFAST and MAPFASTv2 datasets from MAPF Benchmarking runs
# usage: sbatch benchmark_dataset.sh <MAPF Benchmarking folder> <aggregate job id>...
#
# Merges the given benchmark.sh aggregate runs into <benchmarks folder>/output/full_<jobid>_*,
# then builds from it, with every training instance augmented AUGMENTATION times (default 6):
#   datasets/full_four_solvers/   MAPFAST:   bcp2, cbs, cbsh2_rtc and reloc (in place of its BCP, CBS, CBSH and SAT)
#   datasets/full_nine_solvers/   MAPFASTv2: all nine solvers of benchmark.sh
# Both hold the same instances (those that one of the four MAPFAST solvers solves), so the two models
# are split, trained and tested on exactly the same problems.
# generate_benchmark.sh submits this once the benchmark runs have ended.

#SBATCH -J benchmark_dataset
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=milan_capoor@brown.edu

set -e
BENCH=$(readlink -f "$1")
shift
NAME="full_$SLURM_JOB_ID"
MAPFAST_SOLVERS=(bcp2 cbs cbsh2_rtc reloc)
AUGMENTATION="${AUGMENTATION:-6}"
export MPLBACKEND=Agg

# MAPF Benchmarking's environment has everything both steps need (numpy, Pillow with matplotlib)
source "$BENCH/.venv/bin/activate"

echo "Merging benchmark runs $* into $BENCH/output/${NAME}_*..."
(cd "$BENCH" && python -u tools/aggregate.py "$@" --name "$NAME")

echo "Building datasets/full_four_solvers (MAPFAST)..."
python -u benchmark_dataset.py "$BENCH/output/$NAME" --benchmarks "$BENCH/benchmarks" \
	--name full_four_solvers --solvers "${MAPFAST_SOLVERS[@]}" \
	--augmentation "$AUGMENTATION" --jobs "$SLURM_CPUS_PER_TASK"

echo "Building datasets/full_nine_solvers (MAPFASTv2)..."
python -u benchmark_dataset.py "$BENCH/output/$NAME" --benchmarks "$BENCH/benchmarks" \
	--name full_nine_solvers --solved-by "${MAPFAST_SOLVERS[@]}" \
	--augmentation "$AUGMENTATION" --jobs "$SLURM_CPUS_PER_TASK"

echo "Train and compare both with: TRAIN_TIME=12:00:00 ./compare.sh datasets/full_four_solvers datasets/full_nine_solvers"
