#!/bin/bash
# Train, test and analyse MAPFAST (datasets/three_solvers) and MAPFASTv2 (datasets/nine_solvers) with
# train.sh, then plot both models on the same graphs in logs/compare_<jobid>/.
# usage, from the repository folder: ./compare.sh
#
# This script only submits jobs. If the virtual environment does not exist yet, the MAPFASTv2 job
# waits for the MAPFAST job, so that only one of them creates it; otherwise both train at once.

set -e
cd "$(dirname "$0")"
mkdir -p logs

if [ -f pytorch.venvsource/bin/activate ]; then
	after_v1=""
else
	after_v1="--dependency=afterany:"
fi

v1=$(sbatch --parsable -J MAPFAST_training train.sh datasets/three_solvers)
if [ -n "$after_v1" ]; then
	v2=$(sbatch --parsable -J MAPFASTv2_training "${after_v1}${v1}" train.sh datasets/nine_solvers)
else
	v2=$(sbatch --parsable -J MAPFASTv2_training train.sh datasets/nine_solvers)
fi

plot=$(sbatch --parsable -J MAPFAST_compare \
	--dependency="afterok:${v1}:${v2}" \
	--time=0:15:00 --mem=4G --cpus-per-task=1 \
	-o "logs/%j.out" -e "logs/%j.err" \
	--mail-type=END,FAIL --mail-user=milan_capoor@brown.edu \
	--wrap "source pytorch.venvsource/bin/activate && python -u plots.py \
		--run MAPFAST datasets/three_solvers logs/${v1}.out \
		--run MAPFASTv2 datasets/nine_solvers logs/${v2}.out \
		-o logs/compare_\$SLURM_JOB_ID")

echo "MAPFAST   (datasets/three_solvers): job $v1, graphs in logs/${v1}_plots/"
echo "MAPFASTv2 (datasets/nine_solvers):  job $v2, graphs in logs/${v2}_plots/"
echo "Comparison:                         job $plot, graphs in logs/compare_${plot}/"
