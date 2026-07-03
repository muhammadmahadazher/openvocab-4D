# Portfolio Pipeline — Step-by-Step Implementation Plan

Two complementary tracks. Track A is the visual flagship (inference, ships fast, demos hard).
Track B is the physical-intelligence credential (real training, benchmark numbers).
Both are 100% local on RTX 4060 8GB. Do A first (1–2 weeks), then B (2–3 weeks).

---

## Track A — "Video → 4D Semantic World" (VGGT + SAM 3)

**One-liner:** feed a casual phone video, get back a navigable 3D scene where every object is
segmented by open-vocabulary text and tracked over time — the CVPR 2025 + 2026 best-paper stack
in one pipeline.

### Architecture
```
phone video ──► frame sampler (N-frame chunks, overlap)
    ├─► VGGT-1B ──► camera intrinsics/extrinsics, depth maps, dense point maps, confidences
    └─► SAM 3 ───► text-prompted concept masks + video tracklets ("chair", "person", "mug"…)
fusion: back-project masks through VGGT depth → per-object 3D point clouds in world frame
    └─► temporal alignment across chunks (shared frames / pose graph)
output: semantically-labeled 4D point cloud + trajectories ──► Rerun / Viser interactive viewer
```

### Steps
1. **Environment** (Python 3.11, CUDA 12.6):
   `torch>=2.7+cu126`, `vggt` (clone facebookresearch/vggt), `sam3` (clone facebookresearch/sam3),
   `rerun-sdk` (viewer), `opencv-python`, `ffmpeg`. All bf16 on the 4060 (Ada = CC 8.9 ✅).
2. **Milestone 1 — VGGT on stills:** run VGGT-1B on 10–20 photos of your room; export cameras +
   point map; view in Rerun. *Measure VRAM vs frame count; find your chunk limit (~30–60).*
3. **Milestone 2 — SAM 3 concept segmentation:** text-prompt segmentation + tracking on the same
   video; visualize mask tracklets.
4. **Milestone 3 — fusion:** back-project SAM 3 masks through VGGT depth/point maps into world
   frame → per-object colored 3D clouds; handle confidence thresholds.
5. **Milestone 4 — chunked video mode:** sliding window with overlapping frames, align chunk
   poses via shared-frame registration (Umeyama/Procrustes on shared points).
6. **Milestone 5 — polish for portfolio:** CLI (`reconstruct.py --video x.mp4 --concepts "chair,laptop"`),
   README with GIFs, hosted Rerun recording, comparison table vs COLMAP runtime, blog-style writeup
   referencing VGGT/D4RT/SAM 3 papers.
7. **Stretch:** swap Depth Anything 3 as an alternative geometry backbone and compare;
   evaluate OpenD4RT checkpoint on dynamic clips (VGGT assumes mostly-static scenes).

### Risks / mitigations
- VGGT is static-scene biased → keep chunks short for dynamic content, or mask dynamic objects
  (SAM 3 person masks) out of geometry estimation — this itself is a nice contribution to show.
- 8GB OOM → reduce frames/chunk, bf16, `torch.cuda.empty_cache()` between models (run VGGT and
  SAM 3 sequentially, not co-resident).

---

## Track B — "Consumer-scale VLA: fine-tune SmolVLA in simulation" (LeRobot)

**One-liner:** fine-tune a 450M vision-language-action model (same VLM + flow-matching action
expert recipe as π0 / Gemini Robotics / GR00T) on simulated manipulation benchmarks and beat/match
classic imitation baselines — no physical robot required.

### Architecture
```
LIBERO / PushT / ALOHA-sim demos (HF Hub datasets, no hardware needed)
    ──► LeRobot training loop
         ├─ baseline 1: ACT (~80M, transformer BC)
         ├─ baseline 2: Diffusion Policy
         └─ SmolVLA-450M: SigLIP vision + SmolLM2 (frozen) + 100M flow-matching action expert (trained)
    ──► evaluation: success-rate on libero_spatial / libero_object / libero_goal suites
    ──► report: success rates, VRAM, throughput; ablation (frozen vs unfrozen backbone)
```

### Steps
1. **Environment:** `pip install "lerobot[pusht,aloha,libero]"` (+ MuJoCo). Verify sim renders.
2. **Milestone 1 — sanity:** train ACT on PushT from the Hub dataset; eval; reproduce reference
   success rate. (Hours on a 4060, proves the whole loop.)
3. **Milestone 2 — baselines on LIBERO:** ACT + Diffusion Policy on 1–2 LIBERO suites; log to W&B.
4. **Milestone 3 — SmolVLA fine-tune:** start from `lerobot/smolvla_base`; freeze the VLM
   backbone, train action expert; bf16, batch 4–8 + gradient accumulation to emulate bs≥32;
   ~50–100 episodes per task is the documented starting point.
5. **Milestone 4 — evaluation & writeup:** success-rate table SmolVLA vs ACT vs Diffusion Policy;
   language-conditioning demo (same policy, different instructions); connect to π0/GR00T lineage
   in the README.
6. **Stretch (real hardware later):** SO-101 arm (~$100–300) — LeRobot supports it natively;
   record 50 episodes by teleop, fine-tune the same checkpoint → real-world demo video.

### Risks / mitigations
- 8GB during SmolVLA FT → freeze backbone (only ~100M trainable), reduce image resolution,
  grad-accum; fall back to Diffusion Policy as flagship if needed (still very presentable).
- 16GB system RAM with MuJoCo + dataloaders → keep num_workers low, stream datasets from disk.

---

## Why this beats other options (decision record)
- π0/openpi & GR00T N1.5: architecture is the industry story, but fine-tuning needs 22.5–70GB —
  SmolVLA is the same story at a size you can actually train. Interviewers care that you trained
  something end-to-end and measured it.
- Isaac Lab: impressive but heavy; LeRobot+MuJoCo gives benchmarkable results on 8GB.
- Pure detection/segmentation projects (YOLO-style): commodity portfolio material in 2026 — the
  field (per CVPR 2026 stats) has moved to 3D/4D, VLMs, and embodied AI; these two tracks sit
  exactly on that frontier.
