"""STC latency benchmark for the ReKV / LLaVA-OneVision pipeline.

Instruments an already-loaded model at runtime (no changes to repo code) and
times the two stages STC accelerates:

* **ViT encoding** - wraps ``vision_tower.forward``. STC-Cacher patches the
  encoder layers, so the time it saves is captured here.
* **LLM pre-fill** - wraps ``language_model.forward`` for the visual-token
  pre-fill only (calls with ``inputs_embeds``). STC-Pruner shrinks the
  per-frame token budget, shortening this stage.

STC is configured entirely through ``STC_*`` environment variables (read by
``GlobalConfig.initialize_from_env()`` inside ``load_model``); one process runs
one configuration. The script initializes a single-process gloo group so it can
run under plain ``python``.

    python speed_benchmark/benchmark_rekv.py --num-frames 16 --repeats 20 --out out.json

Use ``run.sh`` to run baseline and +STC back to back. Requires a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def ensure_process_group() -> None:
    """ReKV's ``load_model`` calls ``dist.get_rank()``; start a 1-proc gloo group."""
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29577"))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")


class StageTimer:
    """Accumulate GPU time of wrapped calls via CUDA events.

    Each call records one event pair; sum after a single ``synchronize``.
    """

    def __init__(self) -> None:
        self.enabled = False
        self._pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def reset(self) -> None:
        self._pairs = []

    def record(self, fn, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        self._pairs.append((start, end))
        return out

    def wrap(self, fn):
        """Wrap a forward that is always timed (e.g. ``vision_tower.forward``)."""
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)
            return self.record(fn, *args, **kwargs)
        return wrapper

    def wrap_if(self, fn, predicate):
        """Wrap a forward, timing only calls where ``predicate(args, kwargs)`` holds."""
        def wrapper(*args, **kwargs):
            if self.enabled and predicate(args, kwargs):
                return self.record(fn, *args, **kwargs)
            return fn(*args, **kwargs)
        return wrapper

    def elapsed_ms(self) -> float:
        # caller must torch.cuda.synchronize() beforehand
        return float(sum(s.elapsed_time(e) for s, e in self._pairs))


def build_synthetic_video(num_frames: int, size: int, seed: int = 0) -> torch.Tensor:
    """Temporally-coherent synthetic clip ``(N, H, W, 3)`` uint8.

    A base frame drifting under small per-frame noise mimics the high temporal
    redundancy of a real stream. Latency is content-independent; this only keeps
    the cacher's token selection realistic.
    """
    rng = np.random.default_rng(seed)
    frames = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    cur = rng.integers(0, 256, size=(size, size, 3), dtype=np.int16)
    for i in range(num_frames):
        cur = np.clip(cur + rng.integers(-8, 9, size=cur.shape, dtype=np.int16), 0, 255)
        frames[i] = cur.astype(np.uint8)
    return torch.from_numpy(frames)


def load_real_video(path: str, num_frames: int, sample_fps: float) -> torch.Tensor:
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0))
    fps = round(vr.get_avg_fps())
    step = max(1, int(fps / sample_fps))
    idx = list(range(0, len(vr), step))[:num_frames]
    return torch.from_numpy(vr.get_batch(idx).asnumpy())


_HF_REPO_CACHE = {
    "llava_ov_7b": "models--llava-hf--llava-onevision-qwen2-7b-ov-hf",
    "llava_ov_0.5b": "models--llava-hf--llava-onevision-qwen2-0.5b-ov-hf",
}
_PATH_ENV = {
    "llava_ov_7b": "REKV_LLAVA_OV_7B_PATH",
    "llava_ov_0.5b": "REKV_LLAVA_OV_05B_PATH",
}


def resolve_llava_ov_path(model: str) -> str:
    """Env override, else the latest matching HuggingFace snapshot cache."""
    env_path = os.environ.get(_PATH_ENV.get(model, ""))
    if env_path:
        return env_path
    if model not in _HF_REPO_CACHE:
        raise ValueError(f"Unknown model: {model}")
    hf_home = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    snapshots = hf_home / "hub" / _HF_REPO_CACHE[model] / "snapshots"
    candidates = [p for p in snapshots.iterdir() if p.is_dir()] if snapshots.exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"No weights for {model}; set {_PATH_ENV[model]} or populate the HF cache at {snapshots}"
        )
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def run_once(model, video: torch.Tensor) -> None:
    """One full streaming encode (mirrors BaseVQA.encode_video)."""
    model.stc_pruner.reset()
    model.clear_cache()
    model.encode_init_prompt()
    model.encode_video(video)


def summarize(a: np.ndarray) -> dict:
    # On a shared GPU, clock jitter only adds time: min approximates the
    # uncontended floor, median resists spikes.
    return {
        "min": float(a.min()), "median": float(np.median(a)),
        "mean": float(a.mean()), "std": float(a.std()), "max": float(a.max()),
        "samples": [float(x) for x in a],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="STC ReKV latency benchmark")
    parser.add_argument("--model", default="llava_ov_7b")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=384,
                        help="synthetic-frame resolution (the processor resizes to the ViT input)")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--video", default=None, help="optional real video path; replaces synthetic frames")
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--label", default=None, help="result label (inferred from env if omitted)")
    parser.add_argument("--out", default=None, help="path to write the JSON result")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: a GPU is required.", file=sys.stderr)
        return 2

    ensure_process_group()
    torch.set_grad_enabled(False)
    device = 0
    torch.cuda.set_device(device)

    # Load LLaVA-OneVision directly (its load_model calls
    # GlobalConfig.initialize_from_env() and patches the tower per env). We avoid
    # model_utils so we don't import the video_llava / longva model modules.
    from model.llava_onevision_rekv import load_model as load_llava_ov
    from stc import default_config, stc_patch_vision_enabled

    model, _ = load_llava_ov(model_path=resolve_llava_ov_path(args.model), device=device,
                             n_local=15000, topk=64, chunk_size=1)
    model.eval()

    cfg = default_config()
    label = args.label or ("rekv_stc" if stc_patch_vision_enabled() else "rekv")

    # Instrument the loaded instance.
    vit_timer = StageTimer()
    llm_timer = StageTimer()
    model.vision_tower.forward = vit_timer.wrap(model.vision_tower.forward)
    model.language_model.forward = llm_timer.wrap_if(
        model.language_model.forward,
        lambda a, k: k.get("inputs_embeds") is not None,   # visual-token pre-fill only
    )

    if args.video:
        video = load_real_video(args.video, args.num_frames, args.sample_fps)
    else:
        video = build_synthetic_video(args.num_frames, args.image_size)
    num_frames = int(video.shape[0])

    for _ in range(args.warmup):
        run_once(model, video)
    torch.cuda.synchronize()

    vit_ms, llm_ms = [], []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.repeats):
        vit_timer.reset()
        llm_timer.reset()
        vit_timer.enabled = llm_timer.enabled = True
        run_once(model, video)
        vit_timer.enabled = llm_timer.enabled = False
        torch.cuda.synchronize()
        vit_ms.append(vit_timer.elapsed_ms())
        llm_ms.append(llm_timer.elapsed_ms())

    vit_s = summarize(np.asarray(vit_ms))
    llm_s = summarize(np.asarray(llm_ms))
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    result = {
        "label": label,
        "model": args.model,
        "num_frames": num_frames,
        "repeats": args.repeats,
        "config": {
            "patch_vision": stc_patch_vision_enabled(),
            "update_token_ratio": cfg.cache.update_token_ratio,
            "cache_interval": cfg.cache.cache_interval,
            "token_per_frame": cfg.model.token_per_frame,
        },
        "vit_encode_ms": vit_s,
        "llm_prefill_ms": llm_s,
        "peak_mem_gb": float(peak_gb),
    }

    print("=" * 66)
    print(f" STC latency benchmark | label={label} | model={args.model}")
    print(f" frames={num_frames}  repeats={args.repeats}  "
          f"patch_vision={result['config']['patch_vision']}  "
          f"token/frame={result['config']['token_per_frame']}  "
          f"update_ratio={result['config']['update_token_ratio']}  "
          f"N={result['config']['cache_interval']}")
    print("-" * 66)
    print(f" ViT encode   : min {vit_s['min']:8.1f}  median {vit_s['median']:8.1f} ms"
          f"   (mean {vit_s['mean']:.1f} ± {vit_s['std']:.1f})")
    print(f" LLM prefill  : min {llm_s['min']:8.1f}  median {llm_s['median']:8.1f} ms"
          f"   (mean {llm_s['mean']:.1f} ± {llm_s['std']:.1f})")
    print(f" Peak memory  : {peak_gb:8.2f} GB")
    print("=" * 66)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"written: {out}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
