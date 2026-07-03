"""Milestone 5 - Camera pose accuracy: VGGT (chunk-aligned) vs COLMAP reference.

Parses a COLMAP sparse model (images.bin), matches frames by filename to a
pipeline geometry (cameras.npz), Sim(3)-aligns the VGGT trajectory to the
COLMAP one (monocular reconstructions are scale-free), and reports Absolute
Trajectory Error plus per-frame rotation error.

Usage (from CVPR_ex01):
    python trackA/eval_colmap.py --geometry out/video_loft ^
        --colmap "<dataset>/colmap/sparse/0" --out out/eval_loft
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry", required=True, help="dir with cameras.npz")
    p.add_argument("--colmap", required=True, help="COLMAP sparse model dir (with images.bin)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def read_colmap_images_bin(path):
    """name -> (R cam-from-world 3x3, t 3, center 3). Skips 2D point lists."""
    out = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            _image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            t = np.array(struct.unpack("<ddd", f.read(24)))
            _camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(n2d * 24, 1)
            # quaternion (w,x,y,z) -> rotation matrix (cam from world)
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
            ])
            out[Path(name.decode()).name] = (R, t, -R.T @ t)
    return out


def umeyama_sim3(src, dst):
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (xs ** 2).sum() * len(src))
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = read_colmap_images_bin(Path(args.colmap) / "images.bin")
    cams = np.load(Path(args.geometry) / "cameras.npz")
    extrinsic = cams["extrinsic"]
    names = [Path(str(p)).name for p in cams["image_paths"]]

    matched, v_centers, c_centers, v_R, c_R = [], [], [], [], []
    for i, name in enumerate(names):
        if name not in ref:
            continue
        Rv, tv = extrinsic[i, :, :3], extrinsic[i, :, 3]
        Rc, _tc, Cc = ref[name]
        matched.append(name)
        v_centers.append(-Rv.T @ tv)
        c_centers.append(Cc)
        v_R.append(Rv)
        c_R.append(Rc)
    v_centers, c_centers = np.array(v_centers), np.array(c_centers)
    print(f"[eval] matched {len(matched)}/{len(names)} frames against COLMAP "
          f"({len(ref)} registered)")

    s, R, t = umeyama_sim3(v_centers, c_centers)
    v_aligned = (s * v_centers @ R.T) + t
    err = np.linalg.norm(v_aligned - c_centers, axis=1)

    # rotation error: cam-from-world after alignment is Rv @ R^T
    rot_err = []
    for Rv, Rc in zip(v_R, c_R):
        dR = Rc @ (Rv @ R.T).T
        angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        rot_err.append(angle)
    rot_err = np.array(rot_err)

    extent = float(np.linalg.norm(c_centers.max(0) - c_centers.min(0)))
    results = {
        "matched_frames": len(matched),
        "colmap_registered": len(ref),
        "trajectory_extent": round(extent, 3),
        "ate_rmse": round(float(np.sqrt((err ** 2).mean())), 4),
        "ate_median": round(float(np.median(err)), 4),
        "ate_rmse_pct_extent": round(float(np.sqrt((err ** 2).mean())) / extent * 100, 2),
        "rot_err_median_deg": round(float(np.median(rot_err)), 2),
        "rot_err_p90_deg": round(float(np.percentile(rot_err, 90)), 2),
    }
    (out_dir / "eval.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        print(f"[eval] {k}: {v}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].plot(c_centers[:, 0], c_centers[:, 2], "-", color="#333", lw=2, label="COLMAP")
    axes[0].plot(v_aligned[:, 0], v_aligned[:, 2], "-", color="#e6483c", lw=1.5,
                 label="VGGT (chunked, Sim(3)-aligned)")
    axes[0].set_title("Camera trajectory (top-down)")
    axes[0].set_aspect("equal")
    axes[0].legend()
    axes[1].plot(err, color="#e6483c")
    axes[1].set_title(f"Position error per frame  (ATE RMSE {results['ate_rmse']} "
                      f"= {results['ate_rmse_pct_extent']}% of extent)")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("error (COLMAP units)")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_vs_colmap.png", dpi=130)
    print(f"[eval] done -> {out_dir}")


if __name__ == "__main__":
    main()
