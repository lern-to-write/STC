"""STC-Cacher latency benchmark for the StreamForest SigLIP vision tower.

Times the ViT encoding of a clip with vs without STC-Cacher (StreamForest uses
STC cacher-only, so there is no LLM-prefill stage to measure). STC is toggled by
the ``STC_PATCH_VISION`` environment variable, like the ReKV benchmark; run
baseline and +STC back to back with ``run_streamforest.sh``.

    python speed_benchmark/benchmark_streamforest.py --num-frames 16 --repeats 20

Requires a GPU and the StreamForest package on ``PYTHONPATH`` (``models/StreamForest``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def timed_encode(tower, frames, reset, repeats: int, warmup: int) -> np.ndarray:
    # Streaming setting: frames arrive one at a time. Baseline encodes each frame
    # with a full forward; +STC streams (the wrapper reuses across frames). Both
    # are per-frame, isolating the cacher's saving.
    def once():
        if reset is not None:
            reset(tower)
            tower(frames)
        else:
            for i in range(frames.shape[0]):
                tower(frames[i : i + 1])

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(); once(); end.record(); end.synchronize()
        ms.append(start.elapsed_time(end))
    return np.asarray(ms)


def main() -> int:
    ap = argparse.ArgumentParser(description="StreamForest STC-Cacher ViT benchmark")
    ap.add_argument("--ckpt", default=os.environ.get("SIGLIP_CKPT",
                                                      "google/siglip-so400m-patch14-384"))
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: a GPU is required.", file=sys.stderr)
        return 2
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)

    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
    from stc import (GlobalConfig, enable_streaming_cacher, reset_streaming_cacher,
                     stc_patch_vision_enabled)

    GlobalConfig.initialize_from_env()
    stc_on = stc_patch_vision_enabled()
    label = args.label or ("sf_stc" if stc_on else "sf")

    tower = SigLipVisionTower(args.ckpt, None).half().cuda().eval()
    if stc_on:
        enable_streaming_cacher(tower, kind="siglip")

    size = getattr(getattr(tower, "config", None), "image_size", args.image_size)
    frames = torch.randn(args.num_frames, 3, size, size).half().cuda()

    ms = timed_encode(tower, frames, reset_streaming_cacher if stc_on else None,
                      args.repeats, args.warmup)

    result = {
        "label": label, "model": "streamforest", "num_frames": args.num_frames,
        "repeats": args.repeats, "patch_vision": stc_on,
        "vit_encode_ms": {"min": float(ms.min()), "median": float(np.median(ms)),
                          "mean": float(ms.mean()), "std": float(ms.std())},
    }
    print("=" * 60)
    print(f" StreamForest ViT benchmark | label={label} | frames={args.num_frames}")
    print(f" ViT encode : min {ms.min():7.1f}  median {np.median(ms):7.1f} ms"
          f"   ({ms.min() / args.num_frames:.2f} ms/frame)")
    print("=" * 60)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
