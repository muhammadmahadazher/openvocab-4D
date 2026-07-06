"""Milestone 3 - Fusion: SAM 3 concept masks x VGGT geometry -> labeled 3D objects.

For every frame of a reconstructed scene (milestone 1 output with --save-depth),
runs SAM 3 text-prompted segmentation, back-projects each instance mask through
the VGGT depth map and camera into world space, then groups per-frame instances
into persistent 3D objects by centroid proximity.

Usage (from CVPR_ex01):
    python trackA/milestone3_fuse.py --geometry out/loft ^
        --prompts "chair,table,monitor,brick wall,plant" --out out/fusion_loft

Outputs:
    objects.rrd    background cloud (gray) + per-concept colored object clouds + cameras
    objects.ply    all object points, colored by concept
    labels.npz     xyz (N,3), concept_id (N), object_id (N), concepts (list)
    report.json    per-concept object/point counts, timings
"""

import argparse
import cv2
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

CONCEPT_COLORS = [(230, 60, 60), (60, 160, 230), (70, 200, 90), (240, 180, 40),
                  (170, 90, 220), (240, 120, 40), (90, 220, 200), (230, 100, 170)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry", required=True,
                   help="milestone-1 output dir containing cameras.npz + depth.npz")
    p.add_argument("--prompts", required=True, help="comma-separated concepts")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--min-score", type=float, default=0.55, help="SAM 3 detection threshold")
    p.add_argument("--conf-percentile", type=float, default=50.0,
                   help="drop 3D points below this depth-confidence percentile")
    p.add_argument("--merge-eps-frac", type=float, default=0.06,
                   help="instance-merge radius as a fraction of the scene diagonal")
    p.add_argument("--min-points", type=int, default=400,
                   help="discard per-frame instance fragments smaller than this")
    p.add_argument("--max-bg-points", type=int, default=800_000,
                   help="subsample the gray background cloud to this size")
    return p.parse_args()


# rerun before torch: reversed order heap-corrupts the process on Windows
# (see milestone1). The .rrd goes to local disk first: rerun's native IO
# fails on Google Drive virtual paths.
ARGS = parse_args()
OUT_DIR = Path(ARGS.out)
OUT_DIR.mkdir(parents=True, exist_ok=True)

import rerun as rr  # noqa: E402

LOCAL_RRD = Path(tempfile.gettempdir()) / f"vggt_m3_{OUT_DIR.name}.rrd"
rr.init("vggt_sam3_fusion", spawn=False)
rr.save(str(LOCAL_RRD))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

_REPO = Path(__file__).resolve().parents[1] / "vggt"
if _REPO.exists() and str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402


def to_numpy_masks(masks):
    m = masks.detach().float().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    m = np.squeeze(m)
    if m.ndim == 2:
        m = m[None]
    return m > 0.5


def resize_mask(mask, dw, dh):
    """Nearest-resize a bool mask to the depth-map grid, matching the
    resize-width-then-center-crop-height preprocessing of milestone 1."""
    mh, mw = mask.shape
    scale = dw / mw
    new_h = max(dh, int(round(mh * scale)))
    arr = cv2.resize(mask.astype(np.uint8), (dw, new_h), interpolation=cv2.INTER_NEAREST_EXACT)
    top = (new_h - dh) // 2
    return arr[top:top + dh, :] > 0


def run_sam3(image_paths, prompts, min_score, dw, dh):
    """Returns per-frame lists of (mask_at_depth_res, score, prompt_idx)."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    t0 = time.perf_counter()
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print(f"[m3] SAM 3 loaded in {time.perf_counter() - t0:.1f}s")

    per_frame = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for f, path in enumerate(image_paths):
            image = Image.open(path).convert("RGB")
            state = processor.set_image(image)
            dets = []
            for pi, prompt in enumerate(prompts):
                output = processor.set_text_prompt(prompt=prompt, state=state)
                masks = to_numpy_masks(output["masks"])
                scores = output["scores"].detach().float().cpu().numpy().reshape(-1)
                for k in range(min(len(scores), len(masks))):
                    if scores[k] >= min_score:
                        dets.append((resize_mask(masks[k], dw, dh), float(scores[k]), pi))
            per_frame.append(dets)
            print(f"[m3] sam3 {f + 1}/{len(image_paths)}: {len(dets)} instances")

    del processor, model
    torch.cuda.empty_cache()
    return per_frame


def main():
    args = ARGS
    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]
    geo = Path(args.geometry)
    cams = np.load(geo / "cameras.npz")
    dep = np.load(geo / "depth.npz")
    extrinsic, intrinsic = cams["extrinsic"], cams["intrinsic"]
    image_paths = [str(p) for p in cams["image_paths"]]
    depth = dep["depth"].astype(np.float32)   # (S,H,W)
    conf = dep["conf"].astype(np.float32)     # (S,H,W)
    S, dh, dw = depth.shape
    assert len(image_paths) == S, "cameras.npz and depth.npz disagree on frame count"
    print(f"[m3] {S} frames @ {dw}x{dh} | prompts: {prompts}")

    t_start = time.perf_counter()
    per_frame = run_sam3(image_paths, prompts, args.min_score, dw, dh)
    sam3_secs = time.perf_counter() - t_start

    # unproject expects (S,H,W,1); depth.npz stores (S,H,W)
    world = unproject_depth_map_to_point_map(depth[..., None], extrinsic, intrinsic)  # (S,H,W,3)
    conf_keep = conf >= np.percentile(conf, args.conf_percentile)
    conf_keep &= np.isfinite(world).all(axis=-1)

    # colors at depth resolution, from the original frames
    frame_rgb = np.stack([
        np.array(Image.open(p).convert("RGB").resize((dw, dh), Image.BILINEAR))
        for p in image_paths
    ])

    # per-frame instances -> 3D fragments
    fragments = []  # (concept_idx, centroid, points, colors, score)
    for f, dets in enumerate(per_frame):
        for mask, score, pi in dets:
            sel = mask & conf_keep[f]
            if sel.sum() < args.min_points:
                continue
            pts = world[f][sel]
            fragments.append((pi, pts.mean(axis=0), pts, frame_rgb[f][sel], score))

    # group fragments into persistent objects by centroid proximity (greedy;
    # approximate - proper cross-frame identity comes from the SAM 3 video
    # tracker in milestone 4)
    valid_pts = world[conf_keep]
    diag = np.linalg.norm(valid_pts.max(axis=0) - valid_pts.min(axis=0))
    eps = args.merge_eps_frac * diag
    objects = []  # dicts: concept, centroid, n, pts list, cols list
    for pi, cen, pts, cols, _score in sorted(fragments, key=lambda x: -len(x[2])):
        target = None
        for obj in objects:
            if obj["concept"] == pi and np.linalg.norm(obj["centroid"] - cen) < eps:
                target = obj
                break
        if target is None:
            objects.append({"concept": pi, "centroid": cen.copy(), "n": len(pts),
                            "pts": [pts], "cols": [cols]})
        else:
            total = target["n"] + len(pts)
            target["centroid"] = (target["centroid"] * target["n"] + cen * len(pts)) / total
            target["n"] = total
            target["pts"].append(pts)
            target["cols"].append(cols)

    # ---- exports ----
    rng = np.random.default_rng(0)
    bg = valid_pts
    bg_cols = frame_rgb[conf_keep]
    if len(bg) > args.max_bg_points:
        idx = rng.choice(len(bg), args.max_bg_points, replace=False)
        bg, bg_cols = bg[idx], bg_cols[idx]
    gray = (bg_cols.astype(np.float32) * 0.25 + 140).clip(0, 255).astype(np.uint8)
    rr.log("world/background", rr.Points3D(bg, colors=gray, radii=0.002), static=True)

    all_xyz, all_concept, all_object = [], [], []
    counts = {p: 0 for p in prompts}
    for j, obj in enumerate(objects):
        pts = np.concatenate(obj["pts"])
        base = np.array(CONCEPT_COLORS[obj["concept"] % len(CONCEPT_COLORS)], dtype=np.float32)
        jitter = rng.uniform(0.85, 1.15)
        color = np.clip(base * jitter, 0, 255).astype(np.uint8)
        name = prompts[obj["concept"]].replace(" ", "_")
        rr.log(f"world/objects/{name}/obj{j:02d}",
               rr.Points3D(pts, colors=np.tile(color, (len(pts), 1)), radii=0.003), static=True)
        all_xyz.append(pts)
        all_concept.append(np.full(len(pts), obj["concept"], dtype=np.int16))
        all_object.append(np.full(len(pts), j, dtype=np.int32))
        counts[prompts[obj["concept"]]] += 1

    for i in range(S):
        R, t = extrinsic[i, :, :3], extrinsic[i, :, 3]
        rr.log(f"world/cams/cam{i:03d}", rr.Transform3D(translation=-R.T @ t, mat3x3=R.T), static=True)
        rr.log(f"world/cams/cam{i:03d}/frustum",
               rr.Pinhole(image_from_camera=intrinsic[i], resolution=[dw, dh],
                          camera_xyz=rr.ViewCoordinates.RDF), static=True)
    rr.disconnect()
    shutil.copy2(LOCAL_RRD, OUT_DIR / "objects.rrd")
    LOCAL_RRD.unlink(missing_ok=True)

    if all_xyz:
        xyz = np.concatenate(all_xyz)
        concept_id = np.concatenate(all_concept)
        object_id = np.concatenate(all_object)
        np.savez_compressed(OUT_DIR / "labels.npz", xyz=xyz.astype(np.float32),
                            concept_id=concept_id, object_id=object_id,
                            concepts=np.array(prompts))
        header = (b"ply\nformat binary_little_endian 1.0\n"
                  + f"element vertex {len(xyz)}\n".encode()
                  + b"property float x\nproperty float y\nproperty float z\n"
                  + b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        rec = np.empty(len(xyz), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        rec["xyz"] = xyz.astype(np.float32)
        rec["rgb"] = np.array([CONCEPT_COLORS[c % len(CONCEPT_COLORS)] for c in concept_id],
                              dtype=np.uint8)
        with open(OUT_DIR / "objects.ply", "wb") as fh:
            fh.write(header)
            fh.write(rec.tobytes())

    per_object = [{"object": j, "concept": prompts[o["concept"]], "points": int(o["n"])}
                  for j, o in enumerate(objects)]
    (OUT_DIR / "report.json").write_text(json.dumps({
        "frames": S, "prompts": prompts, "sam3_seconds": round(sam3_secs, 1),
        "fragments": len(fragments), "objects_per_concept": counts,
        "objects": per_object,
    }, indent=2))
    print(f"[m3] {len(fragments)} fragments -> {len(objects)} objects: {counts}")
    print(f"[m3] done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
