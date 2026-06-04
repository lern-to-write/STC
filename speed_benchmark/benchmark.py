"""STC latency benchmark (ReKV / LLaVA-OneVision).

测量论文 Table 1 的两项延迟指标，**运行时插桩，不改动仓库任何现有代码**：

* **ViT 编码延迟** —— 用 CUDA event 包住 ``vision_tower.forward`` 累加。
  STC-Cacher 的 monkey patch 作用在 ``vision_tower...encoder.layers[*]`` 内部，
  它“跳过重计算”节省的时间天然被计入这段。
* **LLM prefill 延迟** —— 包住 ``language_model.forward``，只统计带
  ``inputs_embeds`` 的调用（视觉 token 预填，排除常量 system-prompt 预填）。
  STC-Pruner 把每帧 token 从 196 压到 ``STC_TOKEN_PER_FRAME``，预填序列变短 ⇒ 变快。

是否启用 STC 完全由 ``STC_*`` 环境变量决定（与 ``run_distributed`` 一致；
``GlobalConfig.initialize_from_env()`` 在 ``load_model`` 内部被调用）。一个进程跑一档：

    baseline ：STC_PATCH_VISION=0  STC_TOKEN_PER_FRAME=196  STC_UPDATE_TOKEN_RATIO=1.0
    ReKV+STC ：STC_PATCH_VISION=1  STC_TOKEN_PER_FRAME=64   STC_UPDATE_TOKEN_RATIO=0.25
               STC_CACHE_INTERVAL=2   (= 论文 N=2)

Cacher 的算力节省由 ``update_token_ratio`` 固定（只有该比例 token 走重计算，其余复用
reference），与画面内容无关，所以默认用时序连贯的合成帧即可复现加速比；也可 ``--video``
指定真实视频。脚本自行初始化单进程 gloo 进程组，裸 ``python`` 即可运行。

直接用 ``run.sh`` 跑两档对比更方便；单独跑一档：

    python speed_benchmark/benchmark.py --num-frames 16 --repeats 20 --out results/speed/stc.json

⚠️ 必须在带 GPU 的容器里跑（跳板机没有 GPU）。环境要求见同目录 README.md。
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


# --------------------------------------------------------------------------- #
# 进程组：ReKV 的 load_model 会调用 dist.get_rank()，单进程时用 gloo 自起一个
# --------------------------------------------------------------------------- #
def ensure_process_group() -> None:
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29577"))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")


# --------------------------------------------------------------------------- #
# CUDA event 计时器：每次调用记一对 event，测量区结束统一 synchronize 再求和
# --------------------------------------------------------------------------- #
class StageTimer:
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
        """包住一个总是要计时的前向（如 vision_tower.forward）。"""
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)
            return self.record(fn, *args, **kwargs)
        return wrapper

    def wrap_if(self, fn, predicate):
        """只在 predicate(args, kwargs) 为真时计时（如带 inputs_embeds 的预填）。"""
        def wrapper(*args, **kwargs):
            if self.enabled and predicate(args, kwargs):
                return self.record(fn, *args, **kwargs)
            return fn(*args, **kwargs)
        return wrapper

    def elapsed_ms(self) -> float:
        # 调用方负责在此之前 torch.cuda.synchronize()
        return float(sum(s.elapsed_time(e) for s, e in self._pairs))

    def num_calls(self) -> int:
        return len(self._pairs)


# --------------------------------------------------------------------------- #
# 输入
# --------------------------------------------------------------------------- #
def build_synthetic_video(num_frames: int, size: int, seed: int = 0) -> torch.Tensor:
    """时序连贯的合成视频 (Nv, H, W, 3) uint8：底图 + 逐帧小幅噪声漂移，
    模拟流式视频的高时序冗余。延迟与内容无关，这里只为让选择行为接近真实流。"""
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
    """env override -> 最新 HF snapshot 缓存（与 model_utils 同序，去掉 model_zoo）。"""
    env_path = os.environ.get(_PATH_ENV.get(model, ""))
    if env_path:
        return env_path
    if model not in _HF_REPO_CACHE:
        raise ValueError(f"Unknown model: {model}")
    hf_home = Path(os.environ.get("HF_HOME", "/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face"))
    snapshots = hf_home / "hub" / _HF_REPO_CACHE[model] / "snapshots"
    candidates = [p for p in snapshots.iterdir() if p.is_dir()] if snapshots.exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"找不到 {model} 权重；设 {_PATH_ENV[model]} 或确认 HF 缓存存在于 {snapshots}"
        )
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_once(model, video: torch.Tensor) -> None:
    """一次完整的流式编码（与 BaseVQA.encode_video 等价）。"""
    model.stc_pruner.reset()          # 清 pruner 时序 memory，保证各次独立
    model.clear_cache()
    model.encode_init_prompt()
    model.encode_video(video)


def summarize(a: np.ndarray) -> dict:
    # 共享 GPU 上时钟抖动只会“加时间”，故 min 近似无争用下界、median 抗尖刺
    return {
        "min": float(a.min()), "median": float(np.median(a)),
        "mean": float(a.mean()), "std": float(a.std()), "max": float(a.max()),
        "samples": [float(x) for x in a],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="STC ReKV latency benchmark")
    parser.add_argument("--model", default="llava_ov_7b")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="编码帧数；论文 ViT 延迟以 16 帧为基准（默认 16）")
    parser.add_argument("--image-size", type=int, default=384,
                        help="合成帧分辨率（processor 内部会归一化到 ViT 输入）")
    parser.add_argument("--repeats", type=int, default=20, help="计时重复次数")
    parser.add_argument("--warmup", type=int, default=5, help="预热次数（不计时）")
    parser.add_argument("--video", default=None, help="可选：真实视频路径，替代合成帧")
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--label", default=None, help="结果标签，缺省由 env 推断")
    parser.add_argument("--out", default=None, help="写出 JSON 的路径")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: 需要 GPU；请在容器里运行（跳板机没有 GPU）。", file=sys.stderr)
        return 2

    ensure_process_group()
    torch.set_grad_enabled(False)
    device = 0
    torch.cuda.set_device(device)

    # 直接走 LLaVA-OneVision 的 load_model（内部 GlobalConfig.initialize_from_env() +
    # 按 env 决定是否 patch）。刻意不经过 model_utils，避免连带 import video_llava /
    # longva 模型模块（可能缺依赖、且写死了别机路径）。
    from model.llava_onevision_rekv import load_model as load_llava_ov
    from stc import default_config, stc_patch_vision_enabled

    model, _ = load_llava_ov(model_path=resolve_llava_ov_path(args.model), device=device,
                             n_local=15000, topk=64, chunk_size=1)
    model.eval()

    cfg = default_config()
    label = args.label or ("rekv_stc" if stc_patch_vision_enabled() else "rekv")

    # ---- 安装计时探针（仅作用于本进程已加载的实例） ----
    vit_timer = StageTimer()
    llm_timer = StageTimer()
    model.vision_tower.forward = vit_timer.wrap(model.vision_tower.forward)
    model.language_model.forward = llm_timer.wrap_if(
        model.language_model.forward,
        lambda a, k: k.get("inputs_embeds") is not None,   # 只计视觉 token 预填
    )

    # ---- 构造输入 ----
    if args.video:
        video = load_real_video(args.video, args.num_frames, args.sample_fps)
    else:
        video = build_synthetic_video(args.num_frames, args.image_size)
    num_frames = int(video.shape[0])

    # ---- 预热 ----
    for _ in range(args.warmup):
        run_once(model, video)
    torch.cuda.synchronize()

    # ---- 计时 ----
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

    # ---- 打印 ----
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
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"written: {out}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
