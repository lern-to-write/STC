"""数据加载和处理工具"""
import json
import math
import numpy as np
import torch
from decord import VideoReader, cpu
from logzero import logger


def load_and_split_anno(anno_path, world_size, rank):
    """加载并分割标注数据 - PyTorch DistributedSampler推理风格
    
    使用间隔索引方式(strided indexing)分配数据，这是torch.utils.data.distributed.DistributedSampler
    在推理模式下的标准做法。每个rank获取 [rank, rank+world_size, rank+2*world_size, ...]
    
    优点:
    - 数据自动打散，负载更均衡
    - 无数据重复，适合推理任务
    - 符合PyTorch生态标准
    
    Args:
        anno_path: 标注文件路径
        world_size: 总进程数
        rank: 当前进程编号
    
    Returns:
        分配给当前进程的标注数据列表
    """
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    
    # 使用间隔索引: 每个rank取 [rank::world_size]
    # 例如: rank0取[0,3,6...], rank1取[1,4,7...], rank2取[2,5,8...]
    return anno[rank::world_size]





def chunk_video(video, chunk_size):
    """将视频分块"""
    num_frames = video.shape[0]
    for i in range(0, num_frames, chunk_size):
        yield video[i:i + chunk_size]