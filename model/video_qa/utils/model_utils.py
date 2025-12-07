"""模型加载和管理工具"""
import torch
from logzero import logger
from model.config import GlobalConfig
from model import llava_onevision_rekv, video_llava_rekv, longva_rekv


# 模型配置映射
MODEL_REGISTRY = {
    'llava_ov_0.5b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_path': 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
    },
    'llava_ov_7b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_path': '/mnt/data0/public/back/huggingface/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots/0d50680527681998e456c7b78950205bedd8a068',
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

