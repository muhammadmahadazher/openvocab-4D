# Open-Vocabulary 4D Scene Understanding on an 8GB Laptop GPU

**Text prompt + casual video → persistent, labeled 3D objects.**
Point the pipeline at a video walkthrough and a list of concepts (`"chair, table, monitor"`),
and it produces a metric-consistent 3D reconstruction where every instance of every concept is
segmented, tracked across the full video, and placed in world space — fully local on an
RTX 4060 Laptop GPU (8 GB VRAM).

![Labeled 3D reconstruction of a loft from a phone-style walkthrough](docs/scene_labeled.png)
*488-frame walkthrough of a loft (DigAkust dataset) → 49 persistent 3D objects. Red = chair,
blue = table, green = monitor, yellow = brick wall, purple = plant.*

## Why this is interesting

Two of the last two CVPR Best Papers point the same direction: **one feed-forward transformer
replacing the classical 3D pipeline** ([VGGT](https://github.com/facebookresearch/vggt), CVPR 2025)
and its extension to dynamic scenes (D4RT, CVPR 2026). Meanwhile
[SAM 3](https://github.com/facebookresearch/sam3) (Meta, 2025) made segmentation *promptable by
concept*: every instance of an open-vocabulary noun phrase, in images or video.

This project fuses the two into a working system, and adds the engineering nobody publishes:
making it run in a consumer VRAM budget.

```
video frames ─┬─► VGGT-1B, chunked (40 frames, 8 overlap) ─► cameras + depth + confidence
              │       └─ chunks registered into one world frame:
              │          Sim(3) Umeyama on dense overlap-frame depth correspondences
              └─► SAM 3 (846M) per-frame concept detection
                      └─ 3D tracking-by-detection: greedy association of detections
                         by WORLD-SPACE centroid (viewpoint-invariant), flicker filtering
fusion: masks × depth × cameras ─► per-object labeled point clouds
outputs: objects.rrd (interactive Rerun scene) · objects.ply · labels.npz · eval vs COLMAP
```

## Results (loft walkthrough, 244 frames used, RTX 4060 Laptop 8GB)

### Camera pose accuracy vs COLMAP reference (215 matched frames)

| Metric | Value |
|---|---|
| ATE RMSE (Sim(3)-aligned) | **5.6 % of trajectory extent** |
| ATE median | 4.4 % of extent |
| Rotation error median / p90 | **7.9° / 10.8°** |
| VGGT reconstruction time | **~3.5 min** (8 chunks × ~26 s) |
| COLMAP (reference, offline) | hours, with global bundle adjustment |

![VGGT trajectory vs COLMAP](docs/trajectory_vs_colmap.png)

No bundle adjustment, no loop closure — a feed-forward network plus a closed-form alignment,
within striking distance of an offline SfM pipeline at a fraction of the runtime.

### Open-vocabulary 3D objects

| Concept | Persistent 3D objects |
|---|---|
| chair | 22 |
| table | 10 |
| monitor | 11 |
| brick wall | 3 |
| plant | 3 |

283 k labeled 3D points; median object footprint 10 % of the scene diagonal (objects are compact,
not smeared). Cross-frame identity comes from **world-space centroid association** — matching
objects in 3D instead of 2D makes the tracker viewpoint-invariant for free.

![SAM 3 open-vocabulary instance segmentation](docs/sam3_overlay.png)
*Per-frame SAM 3 detections: every chair / monitor / table instance with its own mask and score.*

### VRAM engineering (the actual contribution)

| Component | Config | Peak VRAM |
|---|---|---|
| VGGT-1B, 25 frames | bf16 weights + autocast | 4.6 GiB |
| VGGT-1B, 49 frames | bf16 weights + autocast | 6.8 GiB |
| VGGT-1B, 40-frame chunks (244-frame video) | bf16 | 6.35 GiB steady |
| SAM 3 image (846M) | bf16 autocast | 3.8–4.2 GiB |
| 3D tracking-by-detection, 5 concepts × 244 frames | | **4.1 GiB, 212 s** |
| SAM 3 / 3.1 *video* tracker | any config tried | ❌ OOM |

Finding worth knowing: the SAM 3/3.1 video trackers cannot fit in 8 GB — the multiplex memory
encoder allocates fixed 1152×1152 multi-channel mask buffers regardless of input resolution
(tried: bf16 backbone with fp32-stable decoder, object caps, 720 px input, CPU offload).
The 3D tracking-by-detection replacement runs in half the memory and is ~100 lines of transparent
numpy.

## Install (Windows / Linux / macOS)

```bash
git clone https://github.com/<user>/openvocab-4d && cd openvocab-4d
# Windows:            powershell -ExecutionPolicy Bypass -File install.ps1
# Linux / macOS:      bash install.sh
```

The script creates a venv, installs the right PyTorch build (CUDA 12.6 on Windows/Linux;
CPU/MPS on macOS — experimental and slow, an NVIDIA GPU with 8 GB+ is the intended target),
clones the pinned VGGT and SAM 3 repos, and installs everything as the `openvocab4d` package.

**One manual step:** SAM 3 weights are license-gated by Meta. Accept once at
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), then `hf auth login`.

## Use it

**GUI** — upload a video, type concepts, click Reconstruct:

```bash
ov4d-gui            # or: python app.py
```

Live pipeline log, labeled-scene render, object table, and a button that opens the full
interactive 3D scene in the Rerun viewer.

**CLI** — same pipeline, scriptable:

```bash
ov4d --images path/to/frames --out out/myscene \
     --prompts "chair,table,monitor" --step 2 --render \
     --eval-colmap path/to/colmap/sparse/0   # optional, if you have a reference

rerun out/myscene/objects.rrd   # explore: toggle world/objects/<concept> in the tree
```

Full technical explanation of every stage: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Single-scene tools (each is also a standalone milestone):
`trackA/milestone1_vggt_stills.py` (photos → 3D + VRAM benchmark) ·
`trackA/milestone2_sam3_concepts.py` (concept masks on any images) ·
`trackA/milestone3_fuse.py` (single-chunk fusion) ·
`trackA/milestone4_video.py` (staged full-video pipeline) ·
`trackA/eval_colmap.py` (pose accuracy).

## Honest limitations

- **Track persistence is short** (longest track ≈ 8 detections): association is deliberately
  conservative (radius 5 % of scene diagonal), so an object revisited later in the walkthrough
  usually gets a new ID. Fix on the roadmap: appearance-embedding re-ID (DINOv3 features per mask).
- **Chunk drift**: Sim(3) chaining accumulates error (per-chunk overlap RMS 0.03–0.23 scene units);
  a pose-graph over chunk constraints would tighten the 5.6 % ATE meaningfully.
- **Static-scene assumption**: VGGT treats the scene as rigid; dynamic objects belong to the
  D4RT line of work (its query-based 4D formulation is the natural upgrade here).
- The loft frames are perspective crops from a 360° rig — mild stitching artifacts propagate
  into depth around crop seams.

## Windows survival notes (hard-won)

<details>
<summary>expand</summary>

- `rerun` must be imported **and** its recording stream created **before** `import torch`,
  otherwise the process dies with STATUS_HEAP_CORRUPTION (torch 2.12+cu126 / rerun 0.23.1).
- Never let native code write to a Google Drive virtual path — write to local temp, then copy.
- VGGT's DPT head injects an fp32 positional embedding: half-precision weights require autocast.
- SAM 3 has undeclared deps on Windows: `triton-windows`, `pycocotools`, `psutil`; it must run
  under bf16 autocast (its perflib hard-casts fused ops); `async_loading_frames` breaks
  thread-local autocast; the multiplex predictor's `start_session` forwards a kwarg its
  `init_state` rejects (shimmed at runtime).
- Laptop dGPUs can be powered off by OEM battery management mid-run — "No CUDA GPUs are
  available" with a clean event log means *check the charger*, not the driver.

</details>

## References

- Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025 (Best Paper) — [repo](https://github.com/facebookresearch/vggt)
- Carion, Gustafson, Hu et al., *SAM 3: Segment Anything with Concepts*, 2025 — [repo](https://github.com/facebookresearch/sam3)
- Zhang et al., *Efficiently Reconstructing Dynamic Scenes One D4RT at a Time*, CVPR 2026 (Best Paper)
- Dataset: DigAkust loft scan (Aspekteins GmbH, Saarbrücken) via Kaggle, with COLMAP reference
- Umeyama, *Least-squares estimation of transformation parameters between two point patterns*, TPAMI 1991
