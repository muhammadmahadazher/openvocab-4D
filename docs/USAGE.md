# Usage guide — run, test, troubleshoot

Short, numbered, no fluff. GUI first, CLI second, then what the outputs mean and what to do
when something breaks.

## 1. First-time setup (once)

| Step | Windows | Linux / macOS |
|---|---|---|
| 1. Clone | `git clone https://github.com/muhammadmahadazher/openvocab-4D && cd openvocab-4D` | same |
| 2. Install | `powershell -ExecutionPolicy Bypass -File install.ps1` | `bash install.sh` |
| 3. SAM 3 license | accept at [hf.co/facebook/sam3](https://huggingface.co/facebook/sam3) (free, instant) | same |
| 4. HF login | `.venv\Scripts\hf.exe auth login` | `.venv/bin/hf auth login` |

The installer creates `.venv/`, installs the right PyTorch, clones pinned VGGT + SAM 3, and
registers the `ov4d` / `ov4d-gui` commands inside that venv.

> ⚠️ **Laptop users:** keep the charger plugged in. Some laptops power the NVIDIA GPU off on
> battery and the pipeline will report "No CUDA GPUs are available".

## 2. Test with the GUI (5–10 min)

1. Activate the venv (`.venv\Scripts\activate` / `source .venv/bin/activate`) → run `ov4d-gui`
2. A browser tab opens (or go to http://127.0.0.1:7860)
3. Input — either:
   - **Video file**: drop in an `.mp4`/`.mov`. Best input: walk *slowly* through a room for
     30–60 s, camera held steady, good lighting, avoid pointing at blank walls
   - **Frames folder**: paste a path to a folder of `.jpg`/`.png` from one scene
4. **Concepts**: comma-separated nouns — `chair, table, monitor, plant`. Anything works:
   `red mug`, `whiteboard`, `power outlet`…
5. **Frame step** 2 (default) — raise to 4–5 for long videos
6. Click **Reconstruct** and watch the log:
   - `[vggt] chunk k/N ... peak VRAM` — geometry (≈26 s per 40-frame chunk)
   - `[track] frame f/S` — detection + 3D tracking (≈1 s/frame)
   - `[fuse] ...` — building the labeled cloud
7. When done: labeled render appears, object table fills in
8. Click **Open 3D viewer (Rerun)**:
   - orbit = drag · pan = right-drag · zoom = scroll
   - left panel → `world/objects/<concept>` → toggle the 👁 icon per concept or per object
   - `world/trajectory` shows the recovered camera path

**Expected first-run extras:** VGGT-1B (~5 GB) and SAM 3 (~3.4 GB) checkpoints download once
into the Hugging Face cache.

## 3. CLI

```bash
ov4d --images path/to/frames --out out/myscene \
     --prompts "chair,table,monitor" --step 2 --render \
     --eval-colmap path/to/colmap/sparse/0     # optional benchmark
```

| Flag | Default | Meaning |
|---|---|---|
| `--images` | — | folder of frames (one scene, ordered) |
| `--out` | — | output folder |
| `--prompts` | — | comma-separated concepts |
| `--step` | 2 | use every Nth frame |
| `--chunk-size` | 40 | VGGT frames per chunk — **lower to 24–32 if you OOM** |
| `--overlap` | 8 | shared frames between chunks (registration quality) |
| `--stages` | vggt,track,fuse | rerun any stage independently |
| `--render` | off | also write `scene_labeled.png` |
| `--eval-colmap` | off | pose-accuracy report vs a COLMAP sparse model |

Stages write intermediate artifacts, so `--stages track,fuse` reuses existing geometry —
handy for trying new prompts without re-running VGGT.

## 4. Outputs explained

| File | What it is | Open with |
|---|---|---|
| `objects.rrd` | full interactive scene (background, objects, cameras, trajectory) | `rerun objects.rrd` |
| `objects.ply` | labeled point cloud, concept-colored | MeshLab, CloudCompare, Blender |
| `labels.npz` | `xyz (N,3)` + `concept_id` + `object_id` + names — for your own code | numpy |
| `cameras.npz` | per-frame extrinsics (3×4, OpenCV cam-from-world) + intrinsics | numpy |
| `depth.npz` | per-frame depth + confidence (fp16) | numpy |
| `report.json` | object counts, per-object stats | any editor |
| `align_report.json` | per-chunk registration residuals | any editor |
| `eval/eval.json` | ATE + rotation error vs COLMAP | any editor |

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` in `[vggt]` | `--chunk-size 24` (or 32); close other GPU apps |
| `No CUDA GPUs are available` | laptop on battery? plug in AC. Then `nvidia-smi` to confirm |
| `GatedRepoError: 401` | accept the SAM 3 license on HF, then `hf auth login` |
| Very slow on macOS | expected — no CUDA; use a smaller video and `--step 5` |
| Objects fragment into several IDs | raise association radius: it's conservative by design; re-ID is on the roadmap |
| Reconstruction looks warped | input violates the static-scene assumption (people moving), or frames have heavy motion blur / rolling shutter |
| GUI stuck at "Extracting frames" | very long video — it caps at 2000 frames; give it a minute |

## 6. For developers

- Stage scripts in `trackA/` run standalone — each has a docstring with usage
- Windows-specific landmines (import order, virtual-drive IO, precision rules) are documented
  in [ARCHITECTURE.md](ARCHITECTURE.md) and as comments at the exact code lines
- The COLMAP parser (`eval_colmap.py`) reads `images.bin` natively — no pycolmap dependency
