import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


DEFAULT_HF_HOME = "/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face"

DEFAULT_TASK_ROOTS = {
    "odvbench": ("ODVBench", "OMQA-DS-ALL", "anno/eval/ODVBench"),
    "ovbench": ("OVBench", "ovbench", "anno/eval/OVBench"),
    "ovbench_full": ("OVBench", "ovbench", "anno/eval/OVBench"),
    "ovobench": ("OVO-Bench/chunked_videos", "OVO-Bench", "OVOBench/video/chunked_videos"),
    "streamingbench": (
        "StreamingBench/Real-Time Visual Understanding",
        "StreamingBench/StreamingBench",
        "StreamingBench",
    ),
    "mlvu": ("mlvu", "MLVU/video", "MVLU/MLVU/video"),
    "mlvu_mc": ("mlvu", "MLVU/video", "MVLU/MLVU/video"),
    "mvbench": ("MVBench", "mvbench", "anno/eval/MVBench"),
    "videomme": ("Video-MME/videos", "videomme", "Video-MME/videomme"),
}


def hf_home() -> str:
    return os.path.expanduser(os.environ.get("HF_HOME", DEFAULT_HF_HOME))


def data_root() -> str:
    return os.path.expanduser(os.environ.get("STREAMFOREST_DATA_ROOT", hf_home()))


def _env_key(task_name: str, suffix: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in task_name.upper())
    return f"STREAMFOREST_{normalized}_{suffix}"


def _dedupe(paths: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for path in paths:
        if not path:
            continue
        expanded = os.path.expanduser(path)
        if expanded not in seen:
            result.append(expanded)
            seen.add(expanded)
    return result


def _under_data_root(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(data_root(), path)


def _strip_storage_prefix(path: str) -> str:
    if "s3://" in path:
        return path.split("s3://", 1)[1]
    return path


def _remote_tails(path: str) -> List[str]:
    stripped = _strip_storage_prefix(path).strip("/")
    if not stripped:
        return []
    parts = stripped.split("/")
    tails = [stripped]
    if len(parts) > 1:
        tails.append("/".join(parts[1:]))
    if len(parts) > 2:
        tails.append("/".join(parts[2:]))
    tails.append(parts[-1])
    return _dedupe(tails)


def task_roots(task_name: str, cache_name: Optional[str] = None) -> List[str]:
    roots = []
    for suffix in ("ROOT", "VIDEO_ROOT", "DATA_ROOT"):
        env_value = os.environ.get(_env_key(task_name, suffix))
        if env_value:
            roots.append(env_value)

    if cache_name:
        roots.append(cache_name if os.path.isabs(cache_name) else _under_data_root(cache_name))

    for default in DEFAULT_TASK_ROOTS.get(task_name, ()):
        roots.append(_under_data_root(default))

    roots.append(data_root())
    return _dedupe(roots)


def resolve_benchmark_video(
    task_name: str,
    dataset_folder: str,
    video_name: str,
    cache_name: Optional[str] = None,
) -> Tuple[str, List[str]]:
    if os.path.isabs(video_name) and os.path.exists(video_name):
        return video_name, [video_name]

    candidates = []
    dataset_is_remote = "s3://" in dataset_folder
    dataset_is_abs = os.path.isabs(dataset_folder)

    if dataset_folder and not dataset_is_remote:
        base = dataset_folder if dataset_is_abs else _under_data_root(dataset_folder)
        candidates.extend([os.path.join(base, video_name), os.path.join(base, os.path.basename(video_name))])

    tails = _remote_tails(dataset_folder)
    for root in task_roots(task_name, cache_name):
        candidates.append(os.path.join(root, video_name))
        candidates.append(os.path.join(root, os.path.basename(video_name)))
        for tail in tails:
            candidates.append(os.path.join(root, tail, video_name))
            candidates.append(os.path.join(root, tail, os.path.basename(video_name)))

    candidates = _dedupe(candidates)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate, candidates

    fallback = os.path.join(dataset_folder, video_name) if dataset_folder else video_name
    return fallback, candidates


def resolve_videomme_path(name: str, cache_name: Optional[str] = None) -> Tuple[str, List[str]]:
    candidates = []
    for root in task_roots("videomme", cache_name):
        candidates.append(os.path.join(root, name))
        candidates.append(os.path.join(root, "data", name))

    candidates = _dedupe(candidates)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate, candidates
    return candidates[0] if candidates else name, candidates


def existing_video_variant(path: str) -> str:
    variants = [path]
    suffix = Path(path).suffix
    if suffix:
        stem = path[: -len(suffix)]
        variants.extend([stem + suffix.upper(), stem + ".mkv", stem + ".MKV"])

    for variant in _dedupe(variants):
        if os.path.exists(variant):
            return variant
    return path
