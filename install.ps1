# OpenVocab-4D installer - Windows (NVIDIA GPU, 8GB+ VRAM recommended)
# Usage:  powershell -ExecutionPolicy Bypass -File install.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "== OpenVocab-4D setup (Windows) =="
python -m venv "$root\.venv"
$py = "$root\.venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
Write-Host "-- PyTorch (CUDA 12.6) --"
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Write-Host "-- model repos (pinned) --"
if (-not (Test-Path "$root\vggt")) {
    git clone https://github.com/facebookresearch/vggt "$root\vggt"
    git -C "$root\vggt" checkout a288dd0f14786c93483e45524328726ab7b1b4ce
}
if (-not (Test-Path "$root\sam3")) {
    git clone https://github.com/facebookresearch/sam3 "$root\sam3"
    git -C "$root\sam3" checkout 5dd401d1c5c1d5c3eedff06d41b77af824517619
}
& $py -m pip install -e "$root\vggt" -e "$root\sam3" -e $root

Write-Host ""
Write-Host "Done. Two manual steps remain:"
Write-Host "  1. SAM 3 weights are license-gated: accept at https://huggingface.co/facebook/sam3"
Write-Host "     then run:  $root\.venv\Scripts\hf.exe auth login"
Write-Host "  2. Launch the GUI:  $py $root\app.py"
