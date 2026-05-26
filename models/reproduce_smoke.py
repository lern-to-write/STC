#!/usr/bin/env python3
"""Run minimal smoke inference for the three STC model repos.

This script intentionally does not patch the upstream inference source files.
For checkpoints that contain machine-specific absolute paths, it creates a
small local overlay directory with symlinks to the original weights and a
patched config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK_ROOT = ROOT.parents[1]
HF_HOME = Path(os.environ.get("HF_HOME", "/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face"))
DEFAULT_VIDEO = ROOT / "livecc/demo/sources/howto_fix_laptop_mute_1080p.mp4"

LIVECC_SNAPSHOT = (
    HF_HOME
    / "hub/models--chenjoya--LiveCC-7B-Instruct/snapshots/dbe7415c5d2024c3e2d7f898ecbc58fd9fc8da2f"
)
DISPIDER_SNAPSHOT = (
    HF_HOME
    / "hub/models--Mar2Ding--Dispider/snapshots/8bcbb3a1cde4465f82c4c447a34822b7e3806768"
)
CLIP_SNAPSHOT = (
    HF_HOME
    / "hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
)
STREAMFOREST_CKPT = ROOT / "StreamForest/ckpt/stage4-postft-qwen-siglip/StreamForest-Qwen2-7B"
SIGLIP_SNAPSHOT = (
    HF_HOME
    / "hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
)


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def make_overlay(src: Path, dst: Path, updates: dict[str, str]) -> Path:
    require_path(src / "config.json", "source config")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == "config.json":
            continue
        target = dst / item.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(item, target_is_directory=item.is_dir())
    cfg = json.loads((src / "config.json").read_text())
    cfg.update(updates)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return dst


def worker_env(repo: Path, cuda_device: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    env["HF_HOME"] = str(HF_HOME)
    env["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    env.pop("TRANSFORMERS_CACHE", None)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_worker(model: str, cuda_device: str, video: Path) -> None:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", model, "--video", str(video)]
    subprocess.run(cmd, cwd=str(ROOT), env=worker_env(ROOT / model_dir(model), cuda_device), check=True)


def model_dir(model: str) -> str:
    return {
        "livecc": "livecc",
        "dispider": "Dispider",
        "streamforest": "StreamForest",
    }[model]


def run_livecc(video: Path) -> None:
    require_path(LIVECC_SNAPSHOT, "LiveCC checkpoint")
    repo = ROOT / "livecc"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ.setdefault("VIDEO_MIN_PIXELS", str(100 * 28 * 28))
    os.environ.setdefault("FPS_MAX_FRAMES", "480")
    os.environ.setdefault("VIDEO_MAX_PIXELS", str(24576 * 28 * 28))

    from demo.infer import LiveCCDemoInfer

    infer = LiveCCDemoInfer(model_path=str(LIVECC_SNAPSHOT), device="cuda")
    state = {"video_path": str(video)}
    printed = 0
    for t in range(8):
        state["video_timestamp"] = t
        for (start, stop), response, state in infer.live_cc(
            message="Please describe the video.",
            state=state,
            max_pixels=128 * 28 * 28,
            repetition_penalty=1.05,
            streaming_eos_base_threshold=0.0,
            streaming_eos_threshold_step=0,
            do_sample=False,
        ):
            print(f"LIVECC_RESULT {start:.1f}-{stop:.1f}: {response}", flush=True)
            printed += 1
        if printed >= 2 or state.get("video_end"):
            break
    print(f"LIVECC_SMOKE_DONE {printed}", flush=True)
    os._exit(0)


def run_dispider(video: Path) -> None:
    require_path(DISPIDER_SNAPSHOT, "Dispider checkpoint")
    require_path(CLIP_SNAPSHOT / "model.safetensors", "CLIP vision tower")
    repo = ROOT / "Dispider"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    overlay = make_overlay(
        DISPIDER_SNAPSHOT,
        repo / ".repro/Dispider-local",
        {"mm_compressor": str(DISPIDER_SNAPSHOT / "stream_compressor")},
    )

    from inference import videoStream

    streamer = videoStream(str(overlay))
    output = streamer.Run(str(video), "Describe what happens in this video in one sentence.")
    print(f"DISPIDER_RESULT {output[:500]}")
    print("DISPIDER_SMOKE_DONE")


def run_streamforest(video: Path) -> None:
    require_path(STREAMFOREST_CKPT, "StreamForest checkpoint")
    require_path(SIGLIP_SNAPSHOT / "model.safetensors", "SigLIP vision tower")
    repo = ROOT / "StreamForest"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    overlay = make_overlay(
        STREAMFOREST_CKPT,
        repo / ".repro/StreamForest-Qwen2-7B-local",
        {"mm_vision_tower": str(SIGLIP_SNAPSHOT)},
    )

    from lmms_eval.api.instance import Instance
    from lmms_eval.models.streamforest import StreamForest

    model = StreamForest(
        pretrained=str(overlay),
        conv_template="qwen_2",
        max_frames_num=16,
        time_msg="short_online_v2",
        batch_size=1,
    )
    model.task_dict = {"smoke": {"test": [{"video": str(video)}]}}

    def doc_to_visual(doc):
        return [doc["video"], {"video_read_type": "decord"}]

    req = Instance(
        request_type="generate_until",
        arguments=(
            "Describe what happens in this video in one sentence.",
            {"max_new_tokens": 64, "temperature": 0, "do_sample": False},
            doc_to_visual,
            0,
            "smoke",
            "test",
        ),
        idx=0,
        metadata={"task": "smoke", "doc_id": 0, "repeats": 1},
    )
    output = model.generate_until([req])[0]
    print(f"STREAMFOREST_RESULT {output[:500]}")
    print("STREAMFOREST_SMOKE_DONE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", "livecc", "dispider", "streamforest"], default="all")
    parser.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--_worker", choices=["livecc", "dispider", "streamforest"], default=None)
    args = parser.parse_args()

    video = args.video.resolve()
    require_path(video, "smoke video")

    if args._worker == "livecc":
        run_livecc(video)
    elif args._worker == "dispider":
        run_dispider(video)
    elif args._worker == "streamforest":
        run_streamforest(video)
    elif args.model == "all":
        for model in ["livecc", "dispider", "streamforest"]:
            run_worker(model, args.cuda_device, video)
    else:
        run_worker(args.model, args.cuda_device, video)


if __name__ == "__main__":
    main()
