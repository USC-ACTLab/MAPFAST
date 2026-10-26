#!/bin/bash
# Train, test and plot MAPFAST on one dataset. Creates pytorch.venvsource/ (CUDA 12.8 torch) if needed.
# usage: sbatch -J NAME scripts/train.sh DATASET   (PYTHON_MODULE overrides python/3.11)

#SBATCH -p gpu --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=24G
#SBATCH -J MAPFAST_training
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=milan_capoor@brown.edu

set -e
DATASET="${1:-datasets/mapfast_paper/four-solvers}"
VENV=pytorch.venvsource
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
# newer CUDA builds need a newer driver than Oscar's GPU nodes have
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
"$VENV/bin/python" -c "import matplotlib" 2>/dev/null || "$VENV/bin/pip" install --no-cache-dir matplotlib

source "$VENV/bin/activate"

# fail rather than train on the CPU
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU: torch ' + torch.__version__; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

echo "Training model..."
python -u models/main.py -C "$DATASET"
echo "Testing model..."
python -u models/main.py -T 0 -C "$DATASET"
echo "Plotting..."
python -u tools/plots.py -C "$DATASET" --log "${LOG_DIR:-logs}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
