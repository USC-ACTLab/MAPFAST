#!/bin/bash
# Queue round 3 (see experiments/EXPERIMENTS.md): plain Slurm jobs, each waiting for the one before and submitting the
# next once it knows its job ids. Logs in experiments/round_3/logs/.
#   1. once round 2's jobs have ended: MAPF Benchmarking's old output/ and logs/ into its experiments/, and MAPFAST's
#      datasets into datasets/<benchmark>/<portfolio>/;
#   2. the 300 s / 500-agent extension of round 1's benchmark and its datasets (generate_benchmark.sh extend);
#   3. once the datasets exist: the eight-solver dataset, the auxiliary-output variants (15 epochs), the MAPFAST-GNN sweep
#      (sweep_gnn.sh) and the comparison (compare.sh) of MAPFAST, MAPFASTv2, both with auxiliary outputs, MAPFAST-GNN and
#      Jev with 20 examples, held until the sweep has chosen MAPFAST-GNN's best configuration.
# Steps 1 and 2 are safe to run again: what is already moved or extended is skipped.
# usage, from MAPFAST on the cluster login node, after pulling both repositories (git restore datasets, then git pull):
#   ./experiments/round_3/queue.sh
#   TYPESAFE_API_KEY  the Jev API key (default: the contents of typesafe_key.txt in the repository folder, not tracked)
#   ROUND2   job ids step 1 waits for, separated by colons (default: all your queued and running jobs)
#   BENCH    MAPF Benchmarking folder (default ~/MAPF-Benchmarking)
# If a job fails, the jobs after it stay pending as DependencyNeverSatisfied: cancel them (scancel), fix the cause, and
# run the rest of the steps by hand.

set -eo pipefail
cd "$(dirname "$0")/../.."
LOGS=experiments/round_3/logs
mkdir -p "$LOGS"
export BENCH=$(readlink -f "${BENCH:-$HOME/MAPF-Benchmarking}")
if [ -z "$TYPESAFE_API_KEY" ] && [ -f typesafe_key.txt ]; then
    export TYPESAFE_API_KEY=$(tr -d '[:space:]' < typesafe_key.txt)
fi
[ -n "$TYPESAFE_API_KEY" ] || { echo "put the Jev API key in typesafe_key.txt or TYPESAFE_API_KEY (step 3 needs it)" >&2; exit 1; }
ROUND2="${ROUND2-$(squeue --me -h -o %i | paste -sd:)}"
after=()
[ -z "$ROUND2" ] || after=(--dependency="afterany:$ROUND2")

# 1. once round 2 has ended: the old results of MAPF Benchmarking into experiments/, the datasets into their folders
STEP1=$(sbatch --parsable -J round3_organize "${after[@]}" -t 1:00:00 --mem=16G -o "$LOGS/organize.out" <<'EOF'
#!/bin/bash
set -e
(cd "$BENCH" && source .venv/bin/activate && MPLBACKEND=Agg python tools/temp_organize_results.py \
    --apply --force --rename full_7095349=movingai_10-200-agents_60s)
source pytorch.venvsource/bin/activate && python tools/rename_datasets.py --apply --force
EOF
)
echo "Step 1: job $STEP1, after round 2 (${ROUND2:-no jobs})"

# 2. the extension and the datasets; once it knows the dataset job, it queues step 3 after it
STEP2=$(sbatch --parsable -J round3_benchmark --dependency=afterok:$STEP1 -t 0:30:00 -o "$LOGS/benchmark.out" <<'EOF'
#!/bin/bash
set -eo pipefail
LOGS=experiments/round_3/logs
./scripts/generate_benchmark.sh extend movingai_10-200-agents_60s 2>&1 | tee $LOGS/benchmark_jobs.txt
DATASETS=$(sed -n 's/^Datasets: *job \([0-9]*\).*/\1/p' $LOGS/benchmark_jobs.txt)

# 3. the eight-solver dataset, the MAPFAST-GNN sweep and the comparison
sbatch -J round3_models --dependency=afterok:$DATASETS -t 0:30:00 --mem=8G -o $LOGS/models.out <<'MODELS'
#!/bin/bash
set -eo pipefail
LOGS=experiments/round_3/logs
source pytorch.venvsource/bin/activate
D=datasets/movingai_10-500-agents_300s
python tools/derive_dataset.py $D/nine-solvers --name ${D#datasets/}/eight-solvers --drop mapf_prioritized_sipp
python tools/derive_dataset.py $D/eight-solvers --name ${D#datasets/}/eight-solvers/gnn
for portfolio in four-solvers eight-solvers; do
    python tools/derive_dataset.py $D/$portfolio --name ${D#datasets/}/$portfolio/aux --aux --epochs 15
done
./scripts/sweep_gnn.sh $D/eight-solvers 2>&1 | tee $LOGS/gnn_jobs.txt
SELECT=$(sed -n 's/^select: job \([0-9]*\).*/\1/p' $LOGS/gnn_jobs.txt)
# Jev zero-shot was far behind in round 2 (5-6x bcp2's runtime), so only Jev with 20 examples (7 runs, at most 8 plot)
OUT=experiments/round_3 TRAIN_TIME=24:00:00 JEV_EXAMPLES=20 ./scripts/compare.sh \
    MAPFAST=$D/four-solvers MAPFASTv2=$D/eight-solvers MAPFAST_aux=$D/four-solvers/aux MAPFASTv2_aux=$D/eight-solvers/aux \
    MAPFAST-GNN=@$D/eight-solvers/gnn 2>&1 | tee $LOGS/compare_jobs.txt
PLOT=$(sed -n 's/^Comparison: job \([0-9]*\).*/\1/p' $LOGS/compare_jobs.txt)
# the comparison waits for the models it trained; hold it until MAPFAST-GNN's best configuration is chosen too
scontrol hold $PLOT
sbatch -J round3_release --dependency=afterok:$SELECT -t 0:05:00 --wrap "scontrol release $PLOT"
MODELS
EOF
)
echo "Step 2: job $STEP2, after step 1; it queues step 3 (logs in $LOGS/)"
