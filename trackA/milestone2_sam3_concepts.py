"""Milestone 2 - SAM 3 text-prompted concept segmentation.

Runs Meta's SAM 3 (848M) on test images with open-vocabulary text prompts,
saving instance-mask overlays for visual inspection and raw masks for the
Milestone 3 fusion step (masks -> VGGT depth -> labeled 3D objects).

Usage (from CVPR_ex01):
    python trackA/milestone2_sam3_concepts.py ^
        --images "C:/Users/mahad/datasets/loft_every10" --per-source 3 ^
        --prompts "chair,table,monitor,brick wall,plant" --out out/sam3_loft

Outputs per image:
    <stem>__overlay.png   all prompts rendered: colored masks + boxes + scores
    <stem>__masks.npz     masks (N,H,W) bool, boxes (N,4) xyxy, scores (N),
                          prompt_ids (N), prompts (list)
    report.json           timing + peak VRAM per image
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# tab10-style palette, one color per instance (cycled)
PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
           (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
           (188, 189, 34), (23, 190, 207)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", nargs="+", required=True,
                   help="image files and/or folders (folders contribute --per-source images)")
    p.add_argument("--per-source", type=int, default=3,
                   help="how many images to take from each folder source")
    p.add_argument("--prompts", required=True,
                   help="comma-separated noun phrases, e.g. 'chair,brick wall'")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--checkpoint", default=None,
                   help="local sam3.pt path (default: download from HF, needs accepted license)")
    p.add_argument("--min-score", type=float, default=0.5, help="drop detections below this score")
    return p.parse_args()


def collect_images(sources, per_source):
    paths = []
    for src in sources:
        src = Path(src)
        if src.is_dir():
            imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            step = max(1, len(imgs) // per_source)
            paths.extend(imgs[::step][:per_source])
        elif src.suffix.lower() in IMAGE_EXTS:
            paths.append(src)
    return paths


def to_numpy_masks(masks):
    m = masks.detach().float().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    m = np.squeeze(m)
    if m.ndim == 2:
        m = m[None]
    return m > 0.5


def draw_overlay(image, detections):
    """detections: list of (mask_bool_hw, box_xyxy, score, prompt)."""
    base = np.array(image.convert("RGB"), dtype=np.float32)
    for i, (mask, _box, _score, _prompt) in enumerate(detections):
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        base[mask] = base[mask] * 0.55 + color * 0.45
    out = Image.fromarray(base.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for i, (_mask, box, score, prompt) in enumerate(detections):
        color = PALETTE[i % len(PALETTE)]
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{prompt} {score:.2f}"
        tw = draw.textlength(label)
        ty = max(0, y0 - 14)
        draw.rectangle([x0, ty, x0 + tw + 6, ty + 14], fill=color)
        draw.text((x0 + 3, ty + 1), label, fill=(255, 255, 255))
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = [s.strip() for s in args.prompts.split(",") if s.strip()]
    image_paths = collect_images(args.images, args.per_source)
    if not image_paths:
        sys.exit("no images found")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available - this milestone expects the RTX 4060.")

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    print(f"[m2] {len(image_paths)} images x {len(prompts)} prompts | loading SAM 3...")
    t0 = time.perf_counter()
    model = build_sam3_image_model(checkpoint_path=args.checkpoint)
    processor = Sam3Processor(model)
    print(f"[m2] model loaded in {time.perf_counter() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    report = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        detections = []
        # SAM 3 must run under bf16 autocast: its perflib fuses ops in bf16
        # regardless of context, so fp32 inference hits dtype mismatches.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            for prompt in prompts:
                output = processor.set_text_prompt(prompt=prompt, state=state)
                masks = to_numpy_masks(output["masks"])
                boxes = output["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
                scores = output["scores"].detach().float().cpu().numpy().reshape(-1)
                for k in range(len(scores)):
                    if scores[k] >= args.min_score and k < len(masks):
                        detections.append((masks[k], boxes[k], float(scores[k]), prompt))
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**30

        stem = path.stem
        draw_overlay(image, detections).save(out_dir / f"{stem}__overlay.png")
        if detections:
            np.savez_compressed(
                out_dir / f"{stem}__masks.npz",
                masks=np.stack([d[0] for d in detections]),
                boxes=np.stack([d[1] for d in detections]),
                scores=np.array([d[2] for d in detections]),
                prompt_ids=np.array([prompts.index(d[3]) for d in detections]),
                prompts=np.array(prompts),
                source=str(path),
            )
        found = {}
        for d in detections:
            found[d[3]] = found.get(d[3], 0) + 1
        report.append({"image": str(path), "seconds": round(elapsed, 2),
                       "peak_vram_gib": round(peak, 2), "instances": found})
        print(f"[m2] {stem}: {elapsed:5.2f}s  peak {peak:.2f} GiB  {found}")

    (out_dir / "report.json").write_text(json.dumps({
        "model": "facebook/sam3 (848M)", "prompts": prompts,
        "gpu": torch.cuda.get_device_name(0), "results": report,
    }, indent=2))
    print(f"[m2] done -> {out_dir}")


if __name__ == "__main__":
    main()
