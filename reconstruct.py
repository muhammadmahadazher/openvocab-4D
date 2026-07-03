"""Open-vocabulary 4D scene reconstruction - one-command CLI.

Turns a folder of video frames plus a list of text concepts into a labeled,
navigable 3D scene (Rerun + PLY), optionally benchmarked against a COLMAP
reference. Thin orchestrator over the staged pipeline in trackA/.

Example:
    python reconstruct.py --images path/to/frames --out out/myscene ^
        --prompts "chair,table,monitor" --step 2 ^
        --eval-colmap path/to/colmap/sparse/0
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, *extra):
    cmd = [sys.executable, "-u", str(ROOT / "trackA" / script), *extra]
    print(f"\n=== {script} {' '.join(extra[:6])} ...")
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--step", default="2")
    p.add_argument("--chunk-size", default="40")
    p.add_argument("--overlap", default="8")
    p.add_argument("--stages", default="vggt,track,fuse")
    p.add_argument("--eval-colmap", default=None,
                   help="optional COLMAP sparse model dir for pose-accuracy eval")
    p.add_argument("--render", action="store_true", help="also render a README-style PNG")
    args = p.parse_args()

    run("milestone4_video.py",
        "--images", args.images, "--out", args.out, "--prompts", args.prompts,
        "--step", args.step, "--chunk-size", args.chunk_size,
        "--overlap", args.overlap, "--stages", args.stages)
    if args.eval_colmap:
        run("eval_colmap.py", "--geometry", args.out,
            "--colmap", args.eval_colmap, "--out", str(Path(args.out) / "eval"))
    if args.render:
        run("render_scene.py", "--geometry", args.out,
            "--out", str(Path(args.out) / "scene_labeled.png"))
    print("\nAll done. Open the scene with:  rerun " + str(Path(args.out) / "objects.rrd"))


if __name__ == "__main__":
    main()
