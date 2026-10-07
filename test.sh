#!/bin/bash
# Test MAPFAST

#SBATCH -p gpu --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH -J MAPFAST_testing
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=milan_capoor@brown.edu

module load cuda

source pytorch.venvsource/bin/activate
echo "Testing model..."
python main.py -T 0 -C "${1:-datasets/three_solvers}"
echo "Running Analysis..."
python analysis.py -C "${1:-datasets/three_solvers}"
