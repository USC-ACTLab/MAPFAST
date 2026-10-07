#!/bin/bash
# Train MAPFAST

#SBATCH -p gpu --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH -J MAPFAST
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

module load cuda

source pytorch.venvsource/bin/activate
python main.py