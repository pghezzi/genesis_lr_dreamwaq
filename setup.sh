#!/usr/bin/env bash
set -e

# -------- Configuration --------
ENV_NAME=genesis_env
PYTHON_VERSION=3.10

# -------- Ensure conda is available --------
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda or Mambaforge first."
    echo "Arch users often install via AUR: miniconda3 or mambaforge"
    exit 1
fi

# -------- Initialize conda for this shell --------
# Required when running from scripts
source "$(conda info --base)/etc/profile.d/conda.sh"

# -------- Create conda environment --------
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

conda activate $ENV_NAME

# -------- Upgrade pip --------
pip install --upgrade pip setuptools wheel

# -------- Install PyTorch --------
echo "Installing PyTorch..."
# CPU-only (portable default)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA, see: https://pytorch.org/get-started/locally/

# -------- Install Genesis --------
echo "Installing Genesis..."
# Adjust if Genesis requires a different install method
pip install genesis-world

# -------- Install genesis_lr --------
echo "Cloning and installing genesis_lr..."
git clone git@github.com:lupinjia/genesis_lr.git
cd genesis_lr
pip install -e .

echo "Setup complete!"
echo "Activate later with: conda activate $ENV_NAME"

