#!/bin/bash
# Train, test and analyse MAPFAST on one dataset

#SBATCH -p gpu --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH -J MAPFAST_training
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=milan_capoor@brown.edu

module load cuda

source pytorch.venvsource/bin/activate
# usage: sbatch train.sh [dataset], e.g. sbatch train.sh datasets/nine_solvers
set -e
DATASET="${1:-datasets/three_solvers}"
echo "Training model..."
python main.py -C "$DATASET"
echo "Testing model..."
python main.py -T 0 -C "$DATASET"
echo "Running Analysis..."
python analysis.py -C "$DATASET"
