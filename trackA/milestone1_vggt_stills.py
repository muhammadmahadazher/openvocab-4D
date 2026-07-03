"""Milestone 1 - VGGT-1B on a folder of still images.

Runs the CVPR 2025 best-paper model (VGGT) on a photo set, benchmarks peak
VRAM across frame counts, and exports cameras + a colored point cloud for
interactive viewing in Rerun.

Usage (from CVPR_ex01):
    python trackA/milestone1_vggt_stills.py --images vggt/examples/kitchen/images --out out/kitchen
    rerun out/kitchen/scene.rrd

Outputs:
    cameras.npz   extrinsic (S,3,4) OpenCV camera-from-world, intrinsic (S,3,3)
    points.ply    confidence-filtered colored point cloud (binary PLY)
    scene.rrd     Rerun recording: point cloud + camera frusta + source images
    report.json   per-frame-count runtime and peak VRAM
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, help="folder containing input photos")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--sweep", default="4,8,16,all",
                   help="comma list of frame counts to benchmark ('all' = every image)")
    p.add_argument("--conf-percentile", type=float, default=50.0,
                   help="drop points below this depth-confidence percentile")
    p.add_argument("--max-points", type=int, default=1_500_000,
                   help="random-subsample the exported cloud to at most this many points")
    p.add_argument("--model-dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="dtype for model weights (bf16 recommended on 8GB)")
    p.add_argument("--no-rerun", action="store_true", help="skip writing scene.rrd")
    p.add_argument("--save-depth", action="store_true",
                   help="also save per-frame depth + confidence maps (depth.npz, needed for fusion)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bootstrap order is load-bearing. On Windows, importing/initializing rerun
# after torch+CUDA corrupts the heap (STATUS_HEAP_CORRUPTION with
# torch 2.12.1+cu126 / rerun 0.23.1), so the rerun recording stream must be
# fully set up before torch is imported. The .rrd is also written to local
# disk first: rerun's native file IO fails on Google Drive virtual paths.
# ---------------------------------------------------------------------------
ARGS = parse_args()
OUT_DIR = Path(ARGS.out)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RR = None
LOCAL_RRD = None
if not ARGS.no_rerun:
    import rerun as rr

    RR = rr
    LOCAL_RRD = Path(tempfile.gettempdir()) / f"vggt_m1_{OUT_DIR.name}.rrd"
    rr.init("vggt_milestone1", spawn=False)
    rr.save(str(LOCAL_RRD))

import torch  # noqa: E402  (must come after rerun init, see above)

# Allow running against the cloned repo even without `pip install -e vggt`.
_REPO = Path(__file__).resolve().parents[1] / "vggt"
if _REPO.exists() and str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def run_inference(model, images, dtype):
    """Forward through aggregator + camera + depth heads (track head skipped to save VRAM).

    autocast is required even with half-precision weights: the DPT head adds an
    fp32 positional embedding that would otherwise feed fp32 activations into
    half-precision convolutions.
    """
    autocast = torch.amp.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32)
    with torch.no_grad(), autocast:
        imgs = images[None]  # (1,S,3,H,W)
        aggregated_tokens_list, ps_idx = model.aggregator(imgs)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, imgs, ps_idx)
    return (
        extrinsic.squeeze(0).float().cpu().numpy(),
        intrinsic.squeeze(0).float().cpu().numpy(),
        depth_map.squeeze(0).float().cpu().numpy(),
        depth_conf.squeeze(0).float().cpu().numpy(),
    )


def write_ply(path, pts, cols):
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {len(pts)}\n".encode()
        + b"property float x\nproperty float y\nproperty float z\n"
        + b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    rec = np.empty(len(pts), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    rec["xyz"] = pts.astype(np.float32)
    rec["rgb"] = cols
    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def log_rerun(pts, cols, extrinsic, intrinsic, frames_u8):
    RR.log("world/points", RR.Points3D(pts, colors=cols), static=True)
    h, w = frames_u8.shape[1:3]
    for i in range(len(extrinsic)):
        R = extrinsic[i, :, :3]
        t = extrinsic[i, :, 3]
        name = f"world/cam{i:03d}"
        RR.log(name, RR.Transform3D(translation=-R.T @ t, mat3x3=R.T), static=True)
        RR.log(f"{name}/image",
               RR.Pinhole(image_from_camera=intrinsic[i], resolution=[w, h],
                          camera_xyz=RR.ViewCoordinates.RDF),
               static=True)
        img = RR.Image(frames_u8[i])
        try:
            img = img.compress(jpeg_quality=75)
        except Exception:
            pass  # older rerun-sdk without compress(); log raw
        RR.log(f"{name}/image", img, static=True)

    RR.disconnect()  # flush before copying off the local disk
    shutil.copy2(LOCAL_RRD, OUT_DIR / "scene.rrd")
    LOCAL_RRD.unlink(missing_ok=True)


def main():
    args = ARGS
    if not torch.cuda.is_available():
        sys.exit("CUDA not available - this milestone expects the RTX 4060.")
    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.model_dtype]

    image_paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        sys.exit(f"no images found in {args.images}")
    print(f"[m1] {len(image_paths)} images | {torch.cuda.get_device_name(0)} | weights {args.model_dtype}")

    t0 = time.perf_counter()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device=device, dtype=dtype)
    model.eval()
    print(f"[m1] model loaded in {time.perf_counter() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    sweep = []
    for tok in args.sweep.split(","):
        n = len(image_paths) if tok.strip() == "all" else int(tok)
        if n <= len(image_paths) and n not in sweep:
            sweep.append(n)
    sweep.sort()

    report, results = [], None
    for n in sweep:
        images = load_and_preprocess_images([str(p) for p in image_paths[:n]]).to(device, dtype=dtype)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            out = run_inference(model, images, dtype)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() / 2**30
            report.append({"frames": n, "seconds": round(elapsed, 2), "peak_vram_gib": round(peak, 2)})
            print(f"[m1] {n:3d} frames: {elapsed:6.2f}s  peak VRAM {peak:.2f} GiB")
            results = (n, images, out)  # keep largest successful run for export
        except torch.cuda.OutOfMemoryError:
            report.append({"frames": n, "status": "OOM"})
            print(f"[m1] {n:3d} frames: OOM - stopping sweep")
            torch.cuda.empty_cache()
            break
        del images

    if results is None:
        sys.exit("all runs OOMed - lower the sweep")
    n, images, (extrinsic, intrinsic, depth_map, depth_conf) = results

    np.savez(OUT_DIR / "cameras.npz", extrinsic=extrinsic, intrinsic=intrinsic,
             image_paths=np.array([str(p) for p in image_paths[:n]]))
    if args.save_depth:
        np.savez_compressed(OUT_DIR / "depth.npz",
                            depth=np.squeeze(depth_map, axis=-1).astype(np.float16),
                            conf=depth_conf.astype(np.float16))

    # Depth-unprojected points are more accurate than the point-map branch (per repo docs).
    world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)  # (S,H,W,3)
    frames_u8 = (images.float().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

    conf = depth_conf.reshape(-1)
    pts = world_points.reshape(-1, 3)
    cols = frames_u8.reshape(-1, 3)
    keep = conf >= np.percentile(conf, args.conf_percentile)
    keep &= np.isfinite(pts).all(axis=1)
    pts, cols = pts[keep], cols[keep]
    if len(pts) > args.max_points:
        idx = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    print(f"[m1] exporting {len(pts):,} points (conf >= p{args.conf_percentile:.0f})")

    write_ply(OUT_DIR / "points.ply", pts, cols)
    if RR is not None:
        log_rerun(pts, cols, extrinsic, intrinsic, frames_u8)

    (OUT_DIR / "report.json").write_text(json.dumps({
        "images_dir": str(args.images), "model": "facebook/VGGT-1B",
        "model_dtype": args.model_dtype, "gpu": torch.cuda.get_device_name(0),
        "exported_frames": n, "exported_points": int(len(pts)), "sweep": report,
    }, indent=2))
    print(f"[m1] done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
