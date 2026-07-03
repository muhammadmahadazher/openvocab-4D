# Frontier CV / Physical Intelligence Research — Landscape & Feasibility

**Date:** 2026-07-02 · **Target hardware:** RTX 4060 Laptop/Desktop 8GB VRAM, 16GB DDR5, i7-14th gen HX
**Goal:** Identify a state-of-the-art, portfolio-worthy CV/robotics pipeline implementable locally.

---

## 1. The landscape (mid-2026)

### CVPR signal
- **CVPR 2025 Best Paper — VGGT: Visual Geometry Grounded Transformer** (Meta AI + Oxford VGG).
  One feed-forward transformer predicts camera poses, depth maps, dense 3D point maps, and point
  tracks from 1–hundreds of images in <1s. Code + weights: `facebookresearch/vggt` (VGGT-1B;
  commercial-licensed checkpoint available; VGGT-Omega successor released May 2026).
- **CVPR 2026 Best Paper — D4RT: Efficiently Reconstructing Dynamic Scenes One D4RT at a Time**
  (Google DeepMind + UCL + Oxford). Extends the VGGT idea to *dynamic* scenes (4D): encode a video
  once, answer depth / point-tracking / camera-pose queries from one latent. Official weights not
  released; community reproduction **OpenD4RT** (`Lijiaxin0111/Open-d4rt`) has training code +
  HF checkpoints (June 2026).
- **CVPR 2026 macro-trends** (4,089 accepted papers): #1 vision-language/multimodal models (10.6%
  of highlights), #2 video generation & world models (8.8%), #3 embodied AI / robotics. The field
  is moving from static recognition → dynamic, spatial, interactive intelligence.

### Frontier labs — what's actually open and usable
| Lab | Flagship work | Open? | Notes |
|---|---|---|---|
| **Physical Intelligence (π)** | π0, π0-FAST, π0.5 (VLA: PaliGemma 3B + flow-matching action expert) | ✅ `openpi` repo, weights | Inference >8GB, LoRA fine-tune >22.5GB, full >70GB — **too big for 8GB** |
| **Google DeepMind** | Gemini Robotics / GR 1.5 (thinking VLA), Gemini Robotics On-Device, D4RT | ❌ models closed (D4RT has community repro) | Ideas/architecture papers are public |
| **Meta FAIR** | VGGT, SAM 3 (848M, promptable *concept* segmentation, text prompts, video tracking; SAM 3.1 Mar 2026), DINOv3, V-JEPA 2 (video world model + V-JEPA 2-AC for zero-shot robot planning) | ✅ all code + weights | The richest source of runnable frontier models |
| **NVIDIA** | Isaac GR00T N1/N1.5 (3B open-weight humanoid VLA), Isaac Lab, Cosmos-Reason | ✅ weights on HF | 3B fine-tune exceeds 8GB; sim stack (Isaac Lab) is heavy |
| **Hugging Face** | LeRobot library + **SmolVLA (450M VLA)** — SmolVLM-2 backbone + 100M flow-matching action expert | ✅ fully open | *Designed* for consumer GPUs; same architecture family as π0/GR00T at 1/7th the size |
| **ByteDance Seed** | Depth Anything 3 (Nov 2025) — any-view geometry foundation model | ✅ | Multiple sizes, consumer-GPU friendly |
| **OpenAI** | Multimodal GPT/o-series vision | ❌ closed | No open CV/robotics artifacts to reproduce; legacy CLIP only |
| **Anthropic** | Interpretability research | n/a | Not a CV pipeline source |

### Key architectural insight (this is the story your portfolio should tell)
Every frontier VLA — π0, π0.5, Gemini Robotics, GR00T N1.5, SmolVLA — is the **same recipe**:
> pretrained vision-language model backbone → + flow-matching / diffusion "action expert" head →
> fine-tuned on demonstration trajectories → outputs continuous action chunks.

SmolVLA is that exact recipe at 450M parameters. Implementing/fine-tuning it means you can speak
fluently about π0 and Gemini Robotics architectures in interviews, with working code to show.

Similarly, VGGT → D4RT is the perception-side story: **one transformer replacing the entire
classical SfM/MVS/tracking pipeline** (COLMAP etc.). CVPR gave both papers back-to-back best-paper
awards — this is *the* certified-hot direction in 3D vision.

---

## 2. Hardware feasibility matrix (RTX 4060, 8GB VRAM)

| Model | Params | Use | Fits 8GB? |
|---|---|---|---|
| VGGT-1B | 1.2B | 3D recon inference (bf16) | ✅ ~30–60 frames per chunk (May-2026 memory fix gives 2–3× more frames) |
| SAM 3 | 848M | concept segmentation + video tracking, inference | ✅ bf16/fp16 |
| Depth Anything 3 | 0.1–1.4B (sizes) | depth / any-view geometry | ✅ pick small/base |
| DINOv3 ViT-S/B | 21–86M | dense features, probing | ✅ trivially |
| V-JEPA 2 ViT-L | 300M | video features, probing | ✅ inference/probing |
| SmolVLA | 450M | **fine-tuning** | ✅ freeze VLM backbone, train action expert, bs 4–8 + grad-accum, bf16 |
| ACT / Diffusion Policy (LeRobot) | 50–100M | train from scratch | ✅ easily |
| π0 / π0.5 | 3.3B | inference / LoRA FT | ⚠️ inference borderline / ❌ FT (22.5GB) |
| GR00T N1.5 | 3B | fine-tune | ❌ (quantized inference only) |

**Rule of thumb that emerged:** on 8GB you do *inference* with ~1B-class perception models and
*training/fine-tuning* with ≤500M-class policy models. Both recommended tracks respect this.

---

## 3. Candidate pipelines considered (ranked)

1. **★ 4D Semantic Scene Reconstruction** (VGGT + SAM 3 [+ Depth Anything 3]) — phone video →
   camera poses + dense 3D geometry + open-vocabulary segmented, tracked objects → interactive
   3D viewer. Inference-only, extremely visual, directly cites two CVPR best papers. **CHOSEN — Track A.**
2. **★ Consumer-scale VLA: fine-tune SmolVLA in simulation** (LeRobot + LIBERO/PushT/ALOHA-sim) —
   no physical robot needed; benchmark vs ACT & Diffusion Policy baselines; the "physical
   intelligence" credential. **CHOSEN — Track B.**
3. π0/openpi fine-tune — rejected: VRAM (needs 22.5GB+ LoRA).
4. GR00T N1.5 + Isaac Lab — rejected: 3B FT exceeds 8GB; Isaac Sim itself is VRAM-hungry.
5. OpenD4RT training reproduction — rejected as primary (training 4D models needs multi-GPU), but
   its checkpoints are a good comparison/eval add-on for Track A.
6. V-JEPA 2 downstream probing — good stretch module (video representation probing), not a
   standalone flagship.

## 4. Open items to verify at implementation time
- Exact max frame-chunk for VGGT-1B bf16 on 8GB (measure; use sliding-window chunking).
- SmolVLA on 8GB: confirm batch size / grad-accum config; freeze vision-language backbone.
- SAM 3 fine-tuning (repo ships `README_TRAIN.md`) — optional, inference is enough for Track A.
- OpenD4RT checkpoint quality vs VGGT per-chunk for dynamic scenes.

## Sources
- CVPR 2025 awards: https://cvpr.thecvf.com/Conferences/2025/News/Awards_Press · https://www.cs.ox.ac.uk/news/2456-full.html
- CVPR 2026 awards & trends: https://cvpr.thecvf.com/Conferences/2026/News/Best_Papers · https://voxel51.com/blog/d4rt-cvpr-2026-best-paper-4d-reconstruction · https://www.basic.ai/blog-post/cvpr-2026-top-papers-award-winners-and-notable-works
- VGGT: https://github.com/facebookresearch/vggt · https://arxiv.org/abs/2503.11651
- D4RT: https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/ · https://arxiv.org/html/2512.08924v1 · https://github.com/Lijiaxin0111/Open-d4rt
- SAM 3: https://github.com/facebookresearch/sam3 · https://arxiv.org/pdf/2511.16719
- Depth Anything 3: https://github.com/ByteDance-Seed/depth-anything-3 · https://arxiv.org/abs/2511.10647
- DINOv3: https://github.com/facebookresearch/dinov3
- V-JEPA 2: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/
- π0/openpi: https://github.com/Physical-Intelligence/openpi · https://www.pi.website/download/pi05.pdf · https://huggingface.co/blog/pi0
- Gemini Robotics 1.5: https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/ · https://arxiv.org/pdf/2510.03342
- GR00T N1.5: https://huggingface.co/nvidia/GR00T-N1.5-3B
- SmolVLA: https://huggingface.co/blog/smolvla · https://huggingface.co/lerobot/smolvla_base
- LeRobot: https://github.com/huggingface/lerobot · https://huggingface.co/docs/lerobot
