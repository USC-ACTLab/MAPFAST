#!/bin/bash
# Train, test and analyse MAPFAST (default datasets/three_solvers) and MAPFASTv2 (default datasets/nine_solvers) with
# train.sh, then plot both models on the same graphs in logs/MAPFAST_vs_MAPFASTv2_<jobid>/.
# Logs and graphs are named after the models, e.g. logs/MAPFAST_<jobid>.out and logs/MAPFASTv2_<jobid>_plots/.
# usage, from the repository folder: ./compare.sh [MAPFAST dataset] [MAPFASTv2 dataset]
#   e.g. TRAIN_TIME=12:00:00 ./compare.sh datasets/full_four_solvers datasets/full_nine_solvers
#   TRAIN_TIME overrides the time limit of the training jobs (train.sh's own otherwise)
#
# This script only submits jobs. If the virtual environment does not exist yet, the MAPFASTv2 job
# waits for the MAPFAST job, so that only one of them creates it; otherwise both train at once.

set -e
cd "$(dirname "$0")"
mkdir -p logs
V1_DATASET="${1:-datasets/three_solvers}"
V2_DATASET="${2:-datasets/nine_solvers}"
time_limit=()
[ -z "$TRAIN_TIME" ] || time_limit=(--time="$TRAIN_TIME")

if [ -f pytorch.venvsource/bin/activate ]; then
	after_v1=""
else
	after_v1="--dependency=afterany:"
fi

v1=$(sbatch --parsable -J MAPFAST "${time_limit[@]}" train.sh "$V1_DATASET")
if [ -n "$after_v1" ]; then
	v2=$(sbatch --parsable -J MAPFASTv2 "${time_limit[@]}" "${after_v1}${v1}" train.sh "$V2_DATASET")
else
	v2=$(sbatch --parsable -J MAPFASTv2 "${time_limit[@]}" train.sh "$V2_DATASET")
fi

plot=$(sbatch --parsable -J MAPFAST_vs_MAPFASTv2 \
	--dependency="afterok:${v1}:${v2}" \
	--time=0:15:00 --mem=4G --cpus-per-task=1 \
	-o "logs/%x_%j.out" -e "logs/%x_%j.err" \
	--mail-type=END,FAIL --mail-user=milan_capoor@brown.edu \
	--wrap "source pytorch.venvsource/bin/activate && python -u plots.py \
		--run MAPFAST $V1_DATASET logs/MAPFAST_${v1}.out \
		--run MAPFASTv2 $V2_DATASET logs/MAPFASTv2_${v2}.out \
		-o logs/MAPFAST_vs_MAPFASTv2_\$SLURM_JOB_ID")

echo "MAPFAST   ($V1_DATASET): job $v1, log logs/MAPFAST_${v1}.out, graphs in logs/MAPFAST_${v1}_plots/"
echo "MAPFASTv2 ($V2_DATASET): job $v2, log logs/MAPFASTv2_${v2}.out, graphs in logs/MAPFASTv2_${v2}_plots/"
echo "Comparison:                         job $plot, log logs/MAPFAST_vs_MAPFASTv2_${plot}.out, graphs in logs/MAPFAST_vs_MAPFASTv2_${plot}/"
