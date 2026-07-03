"""OpenVocab-4D GUI - point it at a video, type concepts, get a labeled 3D scene.

Launch:  python app.py   (or `ov4d-gui` after pip install)
Then open the printed local URL in a browser.
"""

import json
import subprocess
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "out"


def extract_frames(video_path, dest, max_frames=2000):
    import cv2

    dest.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    keep_every = max(1, total // max_frames)
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % keep_every == 0:
            cv2.imwrite(str(dest / f"frame_{saved:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        i += 1
    cap.release()
    return saved


def run_pipeline(video, frames_dir, prompts, step, chunk_size, scene_name):
    log = ""

    def emit(line):
        nonlocal log
        log += line
        return log, gr.update(), gr.update()

    if not prompts.strip():
        yield emit("ERROR: enter at least one concept, e.g. 'chair, table'\n")
        return
    out_dir = OUT_ROOT / (scene_name.strip() or "scene")

    if video:
        images = out_dir / "frames"
        yield emit(f"Extracting frames from {Path(video).name} ...\n")
        n = extract_frames(Path(video), images)
        yield emit(f"  {n} frames -> {images}\n")
    elif frames_dir and Path(frames_dir).is_dir():
        images = Path(frames_dir)
    else:
        yield emit("ERROR: provide a video file or a valid frames folder\n")
        return

    cmd = [sys.executable, "-u", str(ROOT / "reconstruct.py"),
           "--images", str(images), "--out", str(out_dir),
           "--prompts", prompts, "--step", str(int(step)),
           "--chunk-size", str(int(chunk_size)), "--render"]
    yield emit(f"\n$ {' '.join(cmd[2:])}\n\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(ROOT))
    for line in proc.stdout:
        yield emit(line)
    proc.wait()
    if proc.returncode != 0:
        yield emit(f"\nPipeline FAILED (exit {proc.returncode}) - see log above.\n")
        return

    render = out_dir / "scene_labeled.png"
    table = []
    report = out_dir / "report.json"
    if report.exists():
        r = json.loads(report.read_text())
        table = [[c, n] for c, n in r.get("objects_per_concept", {}).items()]
    log += (f"\nDone. Interactive 3D scene: {out_dir / 'objects.rrd'}\n"
            f"Open it with the button below or:  rerun \"{out_dir / 'objects.rrd'}\"\n")
    yield log, gr.update(value=str(render) if render.exists() else None), gr.update(value=table)


def open_rerun(scene_name):
    rrd = OUT_ROOT / (scene_name.strip() or "scene") / "objects.rrd"
    if not rrd.exists():
        return f"not found: {rrd}"
    subprocess.Popen([sys.executable, "-m", "rerun", str(rrd)], cwd=str(ROOT))
    return f"opening {rrd.name} in the Rerun viewer..."


def build_ui():
    with gr.Blocks(title="OpenVocab-4D") as demo:
        gr.Markdown(
            "# OpenVocab-4D\n"
            "**Video + text concepts → labeled 3D scene.** "
            "Runs VGGT (CVPR 2025 Best Paper) + SAM 3 locally; needs an NVIDIA GPU with ~8 GB "
            "VRAM and one-time [SAM 3 license acceptance](https://huggingface.co/facebook/sam3)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.File(label="Video file (mp4/mov)", file_types=["video"], type="filepath")
                frames_dir = gr.Textbox(label="...or folder of frames (path)",
                                        placeholder="C:/path/to/frames")
                prompts = gr.Textbox(label="Concepts (comma-separated)",
                                     value="chair, table, monitor")
                step = gr.Slider(1, 10, value=2, step=1,
                                 label="Frame step (use every Nth frame)")
                chunk = gr.Slider(16, 56, value=40, step=4,
                                  label="VGGT chunk size (lower if you OOM)")
                name = gr.Textbox(label="Scene name", value="my_scene")
                run_btn = gr.Button("Reconstruct", variant="primary")
                view_btn = gr.Button("Open 3D viewer (Rerun)")
                view_status = gr.Markdown()
            with gr.Column(scale=2):
                logbox = gr.Textbox(label="Pipeline log", lines=22, max_lines=22, autoscroll=True)
                render = gr.Image(label="Labeled scene render", type="filepath")
                objects = gr.Dataframe(headers=["concept", "objects"], label="Persistent 3D objects")
        run_btn.click(run_pipeline, [video, frames_dir, prompts, step, chunk, name],
                      [logbox, render, objects])
        view_btn.click(open_rerun, [name], [view_status])
    return demo


def main():
    build_ui().launch(inbrowser=True)


if __name__ == "__main__":
    main()
