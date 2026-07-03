# How OpenVocab-4D works

This document explains the full system end to end: what goes in, what each stage computes,
the math that connects them, and why it is built this way.

## The problem

Given (a) an ordinary video walkthrough of a space and (b) a list of concepts in plain English
(`"chair, table, monitor"`), produce a **3D scene where every instance of every concept is a
persistent, labeled object** — on a consumer 8 GB GPU, with no cloud calls.

Classically this needs a whole toolchain: SfM (COLMAP) for cameras, MVS for dense geometry, a
detector for semantics, a tracker for identity, and glue code. This project replaces that with
two foundation models plus ~600 lines of fusion logic.

## Data flow

```
                 frames (every Nth video frame)
                          │
        ┌─────────────────┴──────────────────┐
        ▼ stage 1: geometry                  ▼ stage 2: semantics
  VGGT-1B on overlapping chunks         SAM 3 (846M) per frame
  (40 frames, 8 shared)                 text prompt → instance masks + scores
        │ per chunk:                         │
        │  camera extrinsics E=[R|t]         │
        │  camera intrinsics K               │
        │  depth map D + confidence C        │
        ▼                                    ▼
  Sim(3) chunk registration            3D tracking-by-detection
  (Umeyama on overlap-frame            (masks → world centroids →
   depth correspondences)               greedy association)
        │                                    │
        └─────────────────┬──────────────────┘
                          ▼ stage 3: fusion
        back-project every tracked mask through D, K, E
                          ▼
   labeled 3D point cloud: (x,y,z, concept, object_id)
   + Rerun scene (objects.rrd) + PLY + eval report
```

## Stage 1 — feed-forward geometry (VGGT, chunked)

**What VGGT does.** VGGT (CVPR 2025 Best Paper) is a 1.2B-parameter transformer that takes a set
of images and directly regresses, in one forward pass: per-camera extrinsics/intrinsics, a dense
depth map per frame, and per-pixel confidence. No feature matching, no triangulation, no bundle
adjustment — the network learned multi-view geometry from data. A 40-frame chunk takes ~26 s on
an RTX 4060 Laptop.

**Why chunks.** Attention over all frames is quadratic; 8 GB VRAM caps a chunk at ~55 square
frames (we use 40 + 8 overlap). Each chunk comes back in its *own* coordinate system with its
*own* arbitrary scale (monocular reconstruction is scale-free).

**Chunk registration.** Consecutive chunks share 8 overlap frames. The same pixel of the same
overlap frame has a 3D position in both chunks' coordinate systems — that gives thousands of
exact 3D↔3D correspondences per seam (we sample 1 500 high-confidence pixels per overlap frame).
A closed-form **Sim(3) Umeyama fit** (rotation R, translation t, scale s) then maps chunk k+1
into the global frame:

- points:  `X_global = s·R·X_local + t`
- cameras: `R_global = R_local·Rᵀ`,  `t_global = s·t_local − R_global·t`
- depth:   `D_global = s·D_local`

Per-seam RMS residual is logged (0.03–0.23 scene units on the loft) — it is the honest measure of
how well chunks agree.

**Precision trick.** Weights in bf16 *plus* autocast: VGGT's DPT depth head internally creates an
fp32 positional embedding; without autocast the fp32 activations crash into bf16 convolutions.
bf16 weights halve the model to 2.4 GB and are why 40-frame chunks fit.

## Stage 2 — open-vocabulary instances with persistent identity

**Detection.** SAM 3 performs *promptable concept segmentation*: given the text "chair", it
returns a mask, box and score for **every chair instance** in the frame (~0.8 s and ~4 GB for all
five concepts). This runs per frame under bf16 autocast (mandatory — SAM 3's perf library
hard-casts fused ops to bf16).

**Why not SAM 3's own video tracker?** We tried, exhaustively: the SAM 3/3.1 video models
allocate fixed 1152×1152 multi-channel memory buffers per tracked-object bucket *regardless of
input resolution*. bf16 backbones, object caps, 720 px inputs, CPU offload — all still OOM on
8 GB. That hardware wall shaped the design:

**3D tracking-by-detection.** For each detection, its mask is intersected with the confident
region of the frame's depth map and back-projected to a **world-space centroid**. Association is
greedy nearest-centroid: a detection joins the track whose (EMA-smoothed) centroid lies within
5 % of the scene diagonal and was seen within the last 40 frames; otherwise it founds a new track.
Tracks observed in fewer than 3 frames are discarded as flicker.

The key idea: because stage 1 gives us *metric world coordinates*, identity can be resolved in 3D,
where an object's position is viewpoint-invariant — the same chair seen from the front and the
back has the same centroid, even though its 2D masks look nothing alike. Pure-2D trackers do not
get this for free.

## Stage 3 — fusion

For every frame f and tracked mask m: keep pixels where depth confidence ≥ the 55th percentile,
back-project through `K_f, E_f`:

```
X_cam = D(u,v) · K⁻¹ · (u, v, 1)ᵀ        X_world = R_fᵀ (X_cam − t_f)
```

and append the points (capped per object per frame to bound RAM) to the track's cloud, colored by
the source frame. The result is `labels.npz` — `(x, y, z, concept_id, object_id)` per point —
plus a Rerun scene with the background cloud, camera trajectory, frusta, and one toggleable
entity per object.

## Evaluation

The loft dataset ships a COLMAP reconstruction (full offline SfM + bundle adjustment) — we treat
it as reference. `eval_colmap.py` parses `images.bin` directly, matches frames by name,
Sim(3)-aligns our trajectory to COLMAP's (scale is unobservable for both), and reports:

- **ATE RMSE 5.6 % of trajectory extent** (median 4.4 %) over 215 frames
- median rotation error 7.9° (p90 10.8°)

For a feed-forward pipeline with closed-form seam alignment (no global optimization) vs. hours of
COLMAP, that is the trade this system makes: ~3.5 minutes of GPU time for ~95 % of the geometry.

## VRAM budget (why every design choice exists)

| Decision | VRAM consequence |
|---|---|
| bf16 VGGT weights + autocast | 1.2B model: 4.9 → 2.4 GB |
| 40-frame chunks with 8 overlap | peak 6.35 GiB (fits, with margin for fragmentation) |
| Models run *sequentially*, never co-resident | max(VGGT, SAM3), not sum |
| SAM 3 image model instead of video tracker | 4.1 GiB instead of OOM |
| Per-object / per-frame point caps in fusion | bounded CPU RAM on long videos |

## Known limitations (and their fixes)

1. **Short track lifetimes** — conservative association re-IDs an object when the camera returns
   to a room. Fix: mask-level appearance embeddings (e.g. DINOv3) as a second association cue.
2. **Seam drift** — Sim(3) chaining accumulates; a pose graph over chunk constraints (or
   VGGT-Omega's longer context) would tighten ATE.
3. **Rigid-world assumption** — moving objects violate VGGT's geometry; D4RT-style dynamic
   reconstruction is the natural upgrade path.
