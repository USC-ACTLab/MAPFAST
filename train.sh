#!/bin/bash
# Train, test, analyse and plot MAPFAST on one dataset (graphs in logs/<jobid>_plots/)
# usage: sbatch train.sh [dataset], e.g. sbatch train.sh datasets/nine_solvers
#
# The first job creates the virtual environment pytorch.venvsource/ (or repairs it, if its torch
# is not the CUDA 12.8 build that the cluster's GPU driver supports). Set PYTHON_MODULE to the
# name `module avail python` shows if python/3.11 is not one, e.g.
#   sbatch --export=ALL,PYTHON_MODULE=<name from module avail python> train.sh datasets/nine_solvers

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

set -e
DATASET="${1:-datasets/three_solvers}"
VENV=pytorch.venvsource
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
# PyTorch wheels built for CUDA 12.8; newer builds need a newer driver than the GPU nodes have
TORCH_INDEX=https://download.pytorch.org/whl/cu128

module load cuda

if ! [ -f "$VENV/bin/activate" ] || ! "$VENV/bin/python" -c "import torch, torchvision, numpy, PIL; assert torch.version.cuda == '12.8'" 2>/dev/null; then
	echo "Setting up $VENV..."
	module load "$PYTHON_MODULE"
	rm -rf "$VENV"
	python -m venv "$VENV"
	"$VENV/bin/pip" install --upgrade pip
	"$VENV/bin/pip" install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"
	"$VENV/bin/pip" install --no-cache-dir numpy pillow
fi
# for plots.py; installed on its own so that an older venv without it is not rebuilt
"$VENV/bin/python" -c "import matplotlib" 2>/dev/null || "$VENV/bin/pip" install --no-cache-dir matplotlib

source "$VENV/bin/activate"

# stop instead of silently training on the CPU
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU: torch ' + torch.__version__; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

echo "Training model..."
python -u main.py -C "$DATASET"
echo "Testing model..."
python -u main.py -T 0 -C "$DATASET"
echo "Running Analysis..."
python -u analysis.py -C "$DATASET"
echo "Plotting..."
python -u plots.py -C "$DATASET" --log "logs/$SLURM_JOB_ID.out"
