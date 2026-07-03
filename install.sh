#!/usr/bin/env bash
# OpenVocab-4D installer - Linux (NVIDIA GPU) / macOS (CPU-experimental, slow)
# Usage:  bash install.sh
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"

echo "== OpenVocab-4D setup =="
python3 -m venv "$root/.venv"
py="$root/.venv/bin/python"
"$py" -m pip install --upgrade pip

if [[ "$(uname)" == "Darwin" ]]; then
    echo "-- PyTorch (macOS: CPU/MPS -- experimental, expect very slow inference) --"
    "$py" -m pip install torch torchvision
else
    echo "-- PyTorch (CUDA 12.6) --"
    "$py" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi

echo "-- model repos (pinned) --"
if [[ ! -d "$root/vggt" ]]; then
    git clone https://github.com/facebookresearch/vggt "$root/vggt"
    git -C "$root/vggt" checkout a288dd0f14786c93483e45524328726ab7b1b4ce
fi
if [[ ! -d "$root/sam3" ]]; then
    git clone https://github.com/facebookresearch/sam3 "$root/sam3"
    git -C "$root/sam3" checkout 5dd401d1c5c1d5c3eedff06d41b77af824517619
fi
"$py" -m pip install -e "$root/vggt" -e "$root/sam3" -e "$root"

echo
echo "Done. Two manual steps remain:"
echo "  1. SAM 3 weights are license-gated: accept at https://huggingface.co/facebook/sam3"
echo "     then run:  $root/.venv/bin/hf auth login"
echo "  2. Launch the GUI:  $py $root/app.py"
