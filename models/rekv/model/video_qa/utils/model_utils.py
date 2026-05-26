"""模型加载和管理工具"""
import os
from pathlib import Path

import torch
from logzero import logger
from model import llava_onevision_rekv, video_llava_rekv, longva_rekv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ZOO = PROJECT_ROOT / "model_zoo"
HF_HOME = Path(os.environ.get("HF_HOME", "/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face"))


def _latest_snapshot(repo_cache_name: str) -> str | None:
    snapshots = HF_HOME / "hub" / repo_cache_name / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def _model_path(env_name: str, local_name: str, repo_cache_name: str, repo_id: str) -> str:
    env_path = os.environ.get(env_name)
    if env_path:
        return env_path
    local_path = MODEL_ZOO / local_name
    if local_path.exists():
        return str(local_path)
    snapshot = _latest_snapshot(repo_cache_name)
    if snapshot:
        return snapshot
    return repo_id


# 模型配置映射
MODEL_REGISTRY = {
    'llava_ov_0.5b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_path': _model_path(
            'REKV_LLAVA_OV_05B_PATH',
            'llava-onevision-qwen2-0.5b-ov-hf',
            'models--llava-hf--llava-onevision-qwen2-0.5b-ov-hf',
            'llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
        ),
    },
    'llava_ov_7b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_path': _model_path(
            'REKV_LLAVA_OV_7B_PATH',
            'llava-onevision-qwen2-7b-ov-hf',
            'models--llava-hf--llava-onevision-qwen2-7b-ov-hf',
            'llava-hf/llava-onevision-qwen2-7b-ov-hf',
        ),
    },
    'video_llava_7b': {
        'load_func': video_llava_rekv.load_model,
        'model_path': '/mnt/data2/huggingface/hub/models--LanguageBind--Video-LLaVA-7B-hf/snapshots/4cf9d8cfc76a54f46a4cb43be5368b46b7f0d736',
    },
    'longva_7b': {
        'load_func': longva_rekv.load_model,
        'model_path': '/data/wangyiyu-20250922/LongVA-7B',
    },
}


def load_model(model_name, device, n_local=15000, topk=64, chunk_size=1):
    """加载视频问答模型"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    config = MODEL_REGISTRY[model_name]
    model, processor = config['load_func'](
        model_path=config['model_path'],
        device=device,  
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )
    
    return model, processor


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
