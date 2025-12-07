"""使用torch.distributed.run的分布式推理 - 使用PyTorch原生gather收集结果"""
import os
import argparse
import torch
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
from logzero import logger
import pandas as pd

from model.config import GlobalConfig
from .configs import DATASETS
from .utils import (
    load_and_split_anno, 
    load_model, 
    run_evaluation,
)
from .solver_factory import create_solver


def main():
    """主函数"""
    args = parse_args()
    
    ###############################################################################
    # 初始化分布式环境
    assert torch.cuda.is_available(), "DDP推理需要至少一个GPU"
    torch.backends.cuda.matmul.allow_tf32 = getattr(args, 'tf32', False)
    torch.set_grad_enabled(False)
    
    # Setup DDP
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    
    # 设置随机种子
    seed = getattr(args, 'global_seed', 42) * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    if rank == 0:
        logger.info(f"Starting rank={rank}, seed={seed}, world_size={world_size}")
    dist.barrier()
    
    #########################################################################################
    # 加载配置和数据
    GlobalConfig.initialize_from_args(args)
    dataset_config = DATASETS[args.dataset]
    
    # 按rank分配数据
    anno = load_and_split_anno(
        dataset_config.anno_path, 
        world_size=world_size, 
        rank=rank
    )
    
    # 加载模型
    
    model, processor = load_model(
        args.model,
        n_local=args.n_local,
        device=device,
        topk=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size
    )
    
    ######################################################################
    # 同步所有进程
    dist.barrier()
    # 运行推理
    results = run_inference(model, dataset_config,processor, anno, args, rank, world_size)
    logger.info(f"[Rank {rank}] Gathering results...")
    all_results = gather_results(results, rank, world_size)
    
    # Rank 0 保存合并后的结果并运行评估
    if rank == 0:
        save_merged_results(all_results, args.save_dir)
        run_evaluation(args.save_dir, dataset_config.eval_script)
    
    dist.destroy_process_group()



def run_inference(model, dataset_config, processor, anno, args, rank, world_size):
    """运行推理并返回结果"""
    # 使用solver_factory根据配置创建正确的solver
    solver_name = dataset_config.solver

    
    vqa = create_solver(solver_name, model, processor, args)
    
    # 使用tqdm显示进度
    desc = f"Processing [Rank {rank}/{world_size}]"
    for video_sample in tqdm(anno, desc=desc, disable=rank != 0):
        vqa(video_sample)
        
    if rank == 0:
        logger.info(f"[Rank {rank}] Using solver: {solver_name}")
        logger.info(f"[Rank {rank}] Processed {len(vqa.results)} samples")
    return vqa.results  # 直接返回结果列表，不保存CSV


def gather_results(results, rank, world_size):

    # 准备接收容器（仅rank 0需要）
    gathered_results = [None] * world_size if rank == 0 else None
    # 使用PyTorch原生的gather_object收集Python对象
    dist.gather_object(
        obj=results,              # 当前rank的结果
        object_gather_list=gathered_results,  # 收集容器（仅rank 0有效）
        dst=0                     # 收集到rank 0
    )
    if rank == 0:
        # 将所有rank的结果扁平化为一个列表
        all_results = []
        for rank_results in gathered_results:
            all_results.extend(rank_results)
        logger.info(f"Gathered {len(all_results)} total results from {world_size} ranks")
        return all_results
    
    return None


def save_merged_results(results, save_dir):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为CSV
    result_file = save_dir / "results.csv"
    df = pd.DataFrame(results)
    df.to_csv(result_file, index=False)
    logger.info(f"Results saved to: {result_file}")
    logger.info(f"Total samples: {len(results)}")



def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description="分布式视频问答推理")
    
    # 必需参数
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--save_dir", required=True, help="结果保存目录")
    parser.add_argument("--sample_dir", default="./samples", help="样本保存目录")
    
    # 模型参数
    parser.add_argument("--model", default="llava_ov_7b")
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    
    # 分布式参数
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--tf32", action="store_true", help="启用TF32加速")
    
    # 数据参数
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=256)
    
    # 缓存策略参数
    parser.add_argument("--cache_strategy", default="none")
    parser.add_argument("--update_token_ratio", type=float, default=0.3)
    parser.add_argument("--token_per_frame", type=int, default=196)
    parser.add_argument("--prune_strategy", default="full_tokens")
    
    # 调试参数
    parser.add_argument("--debug", action="store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()