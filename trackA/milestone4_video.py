"""Milestone 4 - Full-video pipeline: chunked VGGT + SAM 3 video tracking + fusion.

Scales Track A past the single-chunk VRAM limit and replaces per-frame instance
matching with SAM 3's native video tracker (persistent masklet IDs):

  stage "vggt"   VGGT on overlapping chunks; each chunk is registered to the
                 global frame with a Sim(3) (Umeyama) estimated from dense
                 depth-point correspondences on the overlap frames.
  stage "track"  SAM 3 video predictor: one tracking session per text concept,
                 masklets with stable object IDs across the whole video.
  stage "fuse"   back-project every masklet through the aligned depth maps ->
                 persistent labeled 3D objects + background cloud.

Usage (from CVPR_ex01):
    python trackA/milestone4_video.py --images "C:/Users/mahad/.cache/.../images" ^
        --step 2 --prompts "chair,table,monitor,brick wall,plant" --out out\video_loft

Outputs (in --out):
    cameras.npz / depth.npz / align_report.json   (stage vggt)
    tracks/<prompt>.npz  label volumes (S,H,W) uint16  (stage track)
    objects.rrd / objects.ply / labels.npz / report.json  (stage fuse)
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
CONCEPT_COLORS = [(230, 60, 60), (60, 160, 230), (70, 200, 90), (240, 180, 40),
                  (170, 90, 220), (240, 120, 40), (90, 220, 200), (230, 100, 170)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, help="folder with the full frame sequence")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--prompts", required=True, help="comma-separated concepts")
    p.add_argument("--step", type=int, default=2, help="use every Nth frame")
    p.add_argument("--chunk-size", type=int, default=40, help="frames per VGGT chunk")
    p.add_argument("--overlap", type=int, default=8, help="shared frames between chunks")
    p.add_argument("--stages", default="vggt,track,fuse", help="which stages to run")
    p.add_argument("--conf-percentile", type=float, default=55.0)
    p.add_argument("--min-points", type=int, default=300,
                   help="min 3D points for an object to appear in a frame")
    p.add_argument("--max-obj-frame-points", type=int, default=2500,
                   help="per-frame per-object point cap (RAM bound)")
    p.add_argument("--max-obj-points", type=int, default=150_000,
                   help="total per-object point cap")
    p.add_argument("--seq-dir", default=None,
                   help="local folder for the renamed frame sequence (default: auto under C:)")
    p.add_argument("--track-res", type=int, default=1008,
                   help="max side length for frames fed to the video tracker (it works at "
                        "1008 internally; higher only costs VRAM)")
    return p.parse_args()


# rerun must be imported+initialized before torch on this machine (heap
# corruption otherwise) and .rrd files must be written to local disk, not
# Google Drive. See milestone 1.
ARGS = parse_args()
OUT_DIR = Path(ARGS.out)
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAGES = [s.strip() for s in ARGS.stages.split(",")]

RR = None
LOCAL_RRD = None
if "fuse" in STAGES:
    import rerun as rr

    RR = rr
    LOCAL_RRD = Path(tempfile.gettempdir()) / f"vggt_m4_{OUT_DIR.name}.rrd"
    rr.init("vggt_sam3_video", spawn=False)
    rr.save(str(LOCAL_RRD))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

_REPO = Path(__file__).resolve().parents[1] / "vggt"
if _REPO.exists() and str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from vggt.utils.geometry import depth_to_world_coords_points  # noqa: E402


def frame_list():
    frames = sorted(p for p in Path(ARGS.images).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    frames = frames[::ARGS.step]
    if not frames:
        sys.exit(f"no frames in {ARGS.images}")
    return frames


def umeyama_sim3(src, dst):
    """Least-squares Sim(3): dst ~ s * R @ src + t."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


# ---------------------------------------------------------------------------
# stage: vggt
# ---------------------------------------------------------------------------
def stage_vggt(frames):
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    S = len(frames)
    stride = ARGS.chunk_size - ARGS.overlap
    starts = [0] + list(range(stride, max(S - ARGS.overlap, 1), stride))
    print(f"[vggt] {S} frames -> {len(starts)} chunks (size {ARGS.chunk_size}, overlap {ARGS.overlap})")

    t0 = time.perf_counter()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    print(f"[vggt] model loaded in {time.perf_counter() - t0:.1f}s")

    g_ext = g_int = g_depth = g_conf = None
    prev = None  # (chunk_start, extrinsic, intrinsic, depth, conf) in GLOBAL frame
    align_log = []

    for ci, start in enumerate(starts):
        end = min(start + ARGS.chunk_size, S)
        paths = [str(p) for p in frames[start:end]]
        images = load_and_preprocess_images(paths).to("cuda", dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        tc = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            imgs = images[None]
            tokens, ps_idx = model.aggregator(imgs)
            pose_enc = model.camera_head(tokens)[-1]
            ext, intr = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
            dep, cf = model.depth_head(tokens, imgs, ps_idx)
        ext = ext.squeeze(0).float().cpu().numpy()
        intr = intr.squeeze(0).float().cpu().numpy()
        dep = np.squeeze(dep.squeeze(0).float().cpu().numpy(), axis=-1)
        cf = cf.squeeze(0).float().cpu().numpy()
        del images, imgs, tokens, pose_enc
        torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 2**30

        if g_depth is None:
            dh, dw = dep.shape[1:]
            g_ext = np.zeros((S, 3, 4), np.float32)
            g_int = np.zeros((S, 3, 3), np.float32)
            g_depth = np.zeros((S, dh, dw), np.float16)
            g_conf = np.zeros((S, dh, dw), np.float16)

        if ci == 0:
            sc, Rg, tg, rms = 1.0, np.eye(3), np.zeros(3), 0.0
        else:
            # correspondences on overlap frames: same pixels, two reconstructions
            src_pts, dst_pts = [], []
            rng = np.random.default_rng(ci)
            for k in range(ARGS.overlap):
                gf = start + k                       # global frame index
                pl = prev["start"] + (gf - prev["start"])  # index into prev arrays
                d_prev = prev["depth"][gf - prev["start"]].astype(np.float32)
                c_prev = prev["conf"][gf - prev["start"]].astype(np.float32)
                d_new, c_new = dep[k], cf[k]
                thr_p, thr_n = np.median(c_prev), np.median(c_new)
                valid = (c_prev >= thr_p) & (c_new >= thr_n) & (d_prev > 0) & (d_new > 0)
                ys, xs = np.nonzero(valid)
                if len(ys) < 50:
                    continue
                sel = rng.choice(len(ys), min(1500, len(ys)), replace=False)
                ys, xs = ys[sel], xs[sel]
                w_prev, _, _ = depth_to_world_coords_points(
                    d_prev, prev["ext"][gf - prev["start"]], prev["int"][gf - prev["start"]])
                w_new, _, _ = depth_to_world_coords_points(d_new, ext[k], intr[k])
                dst_pts.append(w_prev[ys, xs])
                src_pts.append(w_new[ys, xs])
            src = np.concatenate(src_pts)
            dst = np.concatenate(dst_pts)
            sc, Rg, tg = umeyama_sim3(src, dst)
            rms = float(np.sqrt(((sc * src @ Rg.T + tg - dst) ** 2).sum(axis=1).mean()))

        # bring chunk into global frame: X_g = s R X_l + t
        # extrinsic: R_g = R_l R^T ; t_g = s t_l - R_g t ; depth scales by s
        ext_g = ext.copy()
        for k in range(len(ext)):
            Rl, tl = ext[k, :, :3], ext[k, :, 3]
            ext_g[k, :, :3] = Rl @ Rg.T
            ext_g[k, :, 3] = sc * tl - ext_g[k, :, :3] @ tg
        dep_g = dep * sc
        conf_g = cf

        first_new = 0 if ci == 0 else ARGS.overlap  # overlap frames keep prev estimates
        g_ext[start + first_new:end] = ext_g[first_new:]
        g_int[start + first_new:end] = intr[first_new:]
        g_depth[start + first_new:end] = dep_g[first_new:].astype(np.float16)
        g_conf[start + first_new:end] = conf_g[first_new:].astype(np.float16)
        if ci == 0:
            g_ext[start:end], g_int[start:end] = ext_g, intr
            g_depth[start:end] = dep_g.astype(np.float16)
            g_conf[start:end] = conf_g.astype(np.float16)

        prev = {"start": start, "ext": ext_g, "int": intr, "depth": dep_g, "conf": cf}
        align_log.append({"chunk": ci, "frames": [start, end], "scale": round(sc, 4),
                          "align_rms": round(rms, 4), "seconds": round(time.perf_counter() - tc, 1),
                          "peak_vram_gib": round(peak, 2)})
        print(f"[vggt] chunk {ci + 1}/{len(starts)} f{start}-{end}: "
              f"{align_log[-1]['seconds']}s  {peak:.2f} GiB  scale {sc:.3f}  rms {rms:.4f}")

    del model
    torch.cuda.empty_cache()
    np.savez(OUT_DIR / "cameras.npz", extrinsic=g_ext, intrinsic=g_int,
             image_paths=np.array([str(p) for p in frames]))
    np.savez_compressed(OUT_DIR / "depth.npz", depth=g_depth, conf=g_conf)
    (OUT_DIR / "align_report.json").write_text(json.dumps(align_log, indent=2))
    print(f"[vggt] geometry saved -> {OUT_DIR}")


# ---------------------------------------------------------------------------
# stage: track
# ---------------------------------------------------------------------------
def resize_mask(mask, dw, dh):
    mh, mw = mask.shape
    scale = dw / mw
    new_h = max(dh, int(round(mh * scale)))
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((dw, new_h), Image.NEAREST)
    arr = np.array(img) > 127
    top = (new_h - dh) // 2
    return arr[top:top + dh, :]


def stage_track(frames):
    """3D tracking-by-detection.

    The SAM 3 / 3.1 video trackers do not fit in 8GB VRAM (the multiplex
    memory encoder allocates fixed 1152x1152 multi-channel buffers regardless
    of input resolution). Instead: per-frame SAM 3 image detection (proven at
    ~4 GiB) + our own cross-frame association using world-space 3D centroids
    from the stage-vggt geometry. World-space matching is viewpoint-invariant,
    which is exactly what per-frame 2D matching (milestone 3) lacked.
    """
    cams = np.load(OUT_DIR / "cameras.npz")
    dep = np.load(OUT_DIR / "depth.npz")
    extrinsic, intrinsic = cams["extrinsic"], cams["intrinsic"]
    depth = dep["depth"]
    conf = dep["conf"].astype(np.float32)
    S, dh, dw = depth.shape
    assert S == len(frames), "geometry frame count != frame list; rerun stage vggt"
    prompts = [s.strip() for s in ARGS.prompts.split(",") if s.strip()]
    conf_thr = np.percentile(conf, ARGS.conf_percentile)

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    t0 = time.perf_counter()
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print(f"[track] SAM 3 image model loaded in {time.perf_counter() - t0:.1f}s")

    label_vols = {pi: np.zeros((S, dh, dw), np.uint16) for pi in range(len(prompts))}
    tracks = [{} for _ in prompts]   # per prompt: id -> {"centroid", "last_frame", "frames"}
    next_id = [1] * len(prompts)
    scene_diag = None
    tp = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for f, path in enumerate(frames):
            d = depth[f].astype(np.float32)
            keep = (conf[f] >= conf_thr) & (d > 0)
            world, _, _ = depth_to_world_coords_points(d, extrinsic[f], intrinsic[f])
            if scene_diag is None:
                pts0 = world[keep]
                scene_diag = float(np.linalg.norm(pts0.max(0) - pts0.min(0))) if len(pts0) else 1.0
            eps = 0.05 * scene_diag

            state = processor.set_image(Image.open(path).convert("RGB"))
            for pi, prompt in enumerate(prompts):
                output = processor.set_text_prompt(prompt=prompt, state=state)
                masks = output["masks"]
                masks = masks.detach().float().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
                masks = np.squeeze(masks)
                if masks.ndim == 2:
                    masks = masks[None]
                scores = output["scores"].detach().float().cpu().numpy().reshape(-1)
                dets = []
                for k in range(min(len(scores), len(masks))):
                    if scores[k] < 0.55:
                        continue
                    m = resize_mask(masks[k] > 0.5, dw, dh)
                    sel = m & keep
                    if sel.sum() < ARGS.min_points:
                        continue
                    dets.append((float(scores[k]), m, world[sel].mean(axis=0)))
                # greedy association, highest score first, in world space
                for score, m, cen in sorted(dets, key=lambda x: -x[0]):
                    best, best_d = None, eps
                    for tid, tr in tracks[pi].items():
                        if f - tr["last_frame"] > 40:
                            continue
                        dist = float(np.linalg.norm(tr["centroid"] - cen))
                        if dist < best_d:
                            best, best_d = tid, dist
                    if best is None:
                        best = next_id[pi]
                        next_id[pi] += 1
                        tracks[pi][best] = {"centroid": cen, "last_frame": f, "frames": 0}
                    tr = tracks[pi][best]
                    tr["centroid"] = 0.7 * tr["centroid"] + 0.3 * cen
                    tr["last_frame"] = f
                    tr["frames"] += 1
                    label_vols[pi][f][m] = best
            if (f + 1) % 40 == 0:
                counts = {prompts[pi]: len(tracks[pi]) for pi in range(len(prompts))}
                print(f"[track] frame {f + 1}/{S}  tracks: {counts}")

    del processor, model
    torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated() / 2**30

    tracks_dir = OUT_DIR / "tracks"
    tracks_dir.mkdir(exist_ok=True)
    summary = {"method": "3D tracking-by-detection (SAM3 image + world-centroid association)",
               "seconds": round(time.perf_counter() - tp, 1), "peak_vram_gib": round(peak, 2)}
    for pi, prompt in enumerate(prompts):
        stable = {tid: tr for tid, tr in tracks[pi].items() if tr["frames"] >= 3}
        # drop flicker tracks (seen <3 frames) from the volume
        vol = label_vols[pi]
        for tid in set(tracks[pi]) - set(stable):
            vol[vol == tid] = 0
        np.savez_compressed(tracks_dir / f"{prompt.replace(' ', '_')}.npz", labels=vol)
        summary[prompt] = {"objects": len(stable)}
        print(f"[track] '{prompt}': {len(stable)} stable tracks "
              f"({len(tracks[pi]) - len(stable)} flicker tracks dropped)")
    (tracks_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[track] done in {summary['seconds']}s, peak {peak:.2f} GiB")


# ---------------------------------------------------------------------------
# stage: fuse
# ---------------------------------------------------------------------------
def stage_fuse(frames):
    cams = np.load(OUT_DIR / "cameras.npz")
    dep = np.load(OUT_DIR / "depth.npz")
    extrinsic, intrinsic = cams["extrinsic"], cams["intrinsic"]
    depth = dep["depth"]
    conf = dep["conf"].astype(np.float32)
    S, dh, dw = depth.shape
    prompts = [s.strip() for s in ARGS.prompts.split(",") if s.strip()]
    conf_thr = np.percentile(conf, ARGS.conf_percentile)

    label_vols = {}
    for pi, prompt in enumerate(prompts):
        f = OUT_DIR / "tracks" / f"{prompt.replace(' ', '_')}.npz"
        if f.exists():
            label_vols[pi] = np.load(f)["labels"]

    rng = np.random.default_rng(0)
    objects = {}   # (prompt_idx, obj_id) -> {"pts": [...], "cols": [...], "frames": int}
    bg_pts, bg_cols = [], []

    for f in range(S):
        d = depth[f].astype(np.float32)
        keep = (conf[f] >= conf_thr) & (d > 0)
        if not keep.any():
            continue
        world, _, _ = depth_to_world_coords_points(d, extrinsic[f], intrinsic[f])
        rgb = np.array(Image.open(str(cams["image_paths"][f])).convert("RGB")
                       .resize((dw, dh), Image.BILINEAR))
        ys, xs = np.nonzero(keep)
        sel = rng.choice(len(ys), min(2000, len(ys)), replace=False)
        bg_pts.append(world[ys[sel], xs[sel]])
        bg_cols.append(rgb[ys[sel], xs[sel]])

        for pi, vol in label_vols.items():
            lab = vol[f]
            for oid in np.unique(lab):
                if oid == 0:
                    continue
                m = (lab == oid) & keep
                n = int(m.sum())
                if n < ARGS.min_points:
                    continue
                key = (pi, int(oid))
                obj = objects.setdefault(key, {"pts": [], "cols": [], "frames": 0, "n": 0})
                if obj["n"] >= ARGS.max_obj_points:
                    obj["frames"] += 1
                    continue
                ys_o, xs_o = np.nonzero(m)
                take = min(ARGS.max_obj_frame_points, len(ys_o))
                sel_o = rng.choice(len(ys_o), take, replace=False)
                obj["pts"].append(world[ys_o[sel_o], xs_o[sel_o]])
                obj["cols"].append(rgb[ys_o[sel_o], xs_o[sel_o]])
                obj["frames"] += 1
                obj["n"] += take
        if (f + 1) % 40 == 0:
            print(f"[fuse] frame {f + 1}/{S}, {len(objects)} objects so far")

    bg = np.concatenate(bg_pts)
    gray = (np.concatenate(bg_cols).astype(np.float32) * 0.25 + 140).clip(0, 255).astype(np.uint8)
    RR.log("world/background", RR.Points3D(bg, colors=gray, radii=0.002), static=True)

    centers = np.stack([-extrinsic[i, :, :3].T @ extrinsic[i, :, 3] for i in range(S)])
    RR.log("world/trajectory", RR.LineStrips3D([centers], colors=[(255, 255, 255)]), static=True)
    for i in range(0, S, 10):
        R, t = extrinsic[i, :, :3], extrinsic[i, :, 3]
        RR.log(f"world/cams/cam{i:03d}", RR.Transform3D(translation=-R.T @ t, mat3x3=R.T), static=True)
        RR.log(f"world/cams/cam{i:03d}/frustum",
               RR.Pinhole(image_from_camera=intrinsic[i], resolution=[dw, dh],
                          camera_xyz=RR.ViewCoordinates.RDF), static=True)

    all_xyz, all_concept, all_object, per_object = [], [], [], []
    counts = {p: 0 for p in prompts}
    for j, ((pi, oid), obj) in enumerate(sorted(objects.items())):
        pts = np.concatenate(obj["pts"])
        base = np.array(CONCEPT_COLORS[pi % len(CONCEPT_COLORS)], dtype=np.float32)
        color = np.clip(base * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
        name = prompts[pi].replace(" ", "_")
        RR.log(f"world/objects/{name}/id{oid:03d}",
               RR.Points3D(pts, colors=np.tile(color, (len(pts), 1)), radii=0.003), static=True)
        all_xyz.append(pts)
        all_concept.append(np.full(len(pts), pi, np.int16))
        all_object.append(np.full(len(pts), j, np.int32))
        counts[prompts[pi]] += 1
        per_object.append({"concept": prompts[pi], "track_id": oid,
                           "frames_seen": obj["frames"], "points": len(pts)})
    RR.disconnect()
    shutil.copy2(LOCAL_RRD, OUT_DIR / "objects.rrd")
    LOCAL_RRD.unlink(missing_ok=True)

    if all_xyz:
        xyz = np.concatenate(all_xyz)
        concept_id = np.concatenate(all_concept)
        np.savez_compressed(OUT_DIR / "labels.npz", xyz=xyz.astype(np.float32),
                            concept_id=concept_id, object_id=np.concatenate(all_object),
                            concepts=np.array(prompts))
        header = (b"ply\nformat binary_little_endian 1.0\n"
                  + f"element vertex {len(xyz)}\n".encode()
                  + b"property float x\nproperty float y\nproperty float z\n"
                  + b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        rec = np.empty(len(xyz), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        rec["xyz"] = xyz.astype(np.float32)
        rec["rgb"] = np.array([CONCEPT_COLORS[c % len(CONCEPT_COLORS)] for c in concept_id], np.uint8)
        with open(OUT_DIR / "objects.ply", "wb") as fh:
            fh.write(header)
            fh.write(rec.tobytes())

    (OUT_DIR / "report.json").write_text(json.dumps({
        "frames": S, "prompts": prompts, "objects_per_concept": counts,
        "objects": per_object,
    }, indent=2))
    print(f"[fuse] {len(objects)} persistent objects: {counts}")
    print(f"[fuse] done -> {OUT_DIR}")


def main():
    frames = frame_list()
    print(f"[m4] {len(frames)} frames (step {ARGS.step}) | stages: {STAGES}")
    if "vggt" in STAGES:
        stage_vggt(frames)
    if "track" in STAGES:
        stage_track(frames)
    if "fuse" in STAGES:
        stage_fuse(frames)


if __name__ == "__main__":
    main()
