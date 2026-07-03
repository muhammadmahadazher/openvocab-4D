"""Render the fused labeled scene (labels.npz + depth background) to PNGs for the README."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONCEPT_COLORS = np.array([(230, 60, 60), (60, 160, 230), (70, 200, 90), (240, 180, 40),
                           (170, 90, 220), (240, 120, 40), (90, 220, 200), (230, 100, 170)]) / 255.0

p = argparse.ArgumentParser()
p.add_argument("--geometry", required=True, help="dir with labels.npz, cameras.npz, depth.npz")
p.add_argument("--out", required=True, help="output PNG path")
args = p.parse_args()

geo = Path(args.geometry)
d = np.load(geo / "labels.npz")
xyz, cid = d["xyz"], d["concept_id"]
concepts = [str(c) for c in d["concepts"]]

# thin background from the depth maps for context
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vggt"))
from vggt.utils.geometry import depth_to_world_coords_points  # noqa: E402

cams = np.load(geo / "cameras.npz")
dep = np.load(geo / "depth.npz")
depth, conf = dep["depth"], dep["conf"].astype(np.float32)
thr = np.percentile(conf, 60)
rng = np.random.default_rng(0)
bg = []
for f in range(0, depth.shape[0], 6):
    dm = depth[f].astype(np.float32)
    keep = (conf[f] >= thr) & (dm > 0)
    w, _, _ = depth_to_world_coords_points(dm, cams["extrinsic"][f], cams["intrinsic"][f])
    pts = w[keep]
    if len(pts):
        bg.append(pts[rng.choice(len(pts), min(1200, len(pts)), replace=False)])
bg = np.concatenate(bg)

# clip outliers for a tight view box
all_pts = np.concatenate([xyz, bg])
lo, hi = np.percentile(all_pts, 2, axis=0), np.percentile(all_pts, 98, axis=0)


def inbox(p):
    return ((p >= lo) & (p <= hi)).all(axis=1)


bg = bg[inbox(bg)]
keep_obj = inbox(xyz)
xyz_v, cid_v = xyz[keep_obj], cid[keep_obj]
if len(xyz_v) > 120_000:
    sel = rng.choice(len(xyz_v), 120_000, replace=False)
    xyz_v, cid_v = xyz_v[sel], cid_v[sel]

fig = plt.figure(figsize=(16, 7), facecolor="#111")
for i, (elev, azim) in enumerate([(-65, -90), (-20, -60)]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d", facecolor="#111")
    ax.scatter(bg[:, 0], bg[:, 2], bg[:, 1], s=0.3, c="#555", alpha=0.35, linewidths=0)
    for pi in range(len(concepts)):
        m = cid_v == pi
        if m.any():
            ax.scatter(xyz_v[m, 0], xyz_v[m, 2], xyz_v[m, 1], s=0.6,
                       c=[CONCEPT_COLORS[pi % len(CONCEPT_COLORS)]],
                       label=concepts[pi] if i == 0 else None, linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((hi - lo)[[0, 2, 1]])
leg = fig.legend(loc="lower center", ncol=len(concepts), frameon=False,
                 markerscale=25, labelcolor="white", fontsize=13)
fig.tight_layout()
fig.savefig(args.out, dpi=140, facecolor="#111", bbox_inches="tight")
print(f"saved {args.out}")
