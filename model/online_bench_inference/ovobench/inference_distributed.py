"""
    分布式推理版本 - 使用 torch.distributed.run
    支持多GPU并行推理，提升OVBench推理速度
"""

import argparse
import os
import json
import sys
import warnings
import torch
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
from logzero import logger

warnings.filterwarnings(
    "ignore",
    message=".*do_sample.*is set to.*However.*temperature.*top_p.*",
    category=UserWarning,
    module="transformers.generation.configuration_utils"
)

warnings.filterwarnings(
    "ignore",
    message=".*copying from a non-meta parameter.*meta parameter.*no-op.*",
    category=UserWarning,
    module="torch.nn.modules.module"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "models"))


def main():
    """主函数"""
    args = parse_args()
    
    ###############################################################################
    # 初始化分布式环境
    assert torch.cuda.is_available(), "DDP推理需要至少一个GPU"
    torch.backends.cuda.matmul.allow_tf32 = getattr(args, 'tf32', False)
    torch.set_grad_enabled(False)
    
    # Setup DDP - 使用gloo后端支持多进程共享GPU
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    
    # 设置随机种子
    seed = getattr(args, 'global_seed', 42) * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    if rank == 0:
        logger.info(f"Starting distributed inference with {world_size} processes on {torch.cuda.device_count()} GPUs")
        logger.info(f"Model: {args.model}; Tasks: {args.task}")
    
    dist.barrier()
    
    #########################################################################################
    # 加载annotations
    with open(args.anno_path, "r") as f:
        annotations = json.load(f)
    
    # 处理视频路径
    for i, item in enumerate(annotations):
        annotations[i]["video"] = os.path.join(args.video_dir, item["video"])
    
    # 按任务类型分组
    backward_tasks = ["EPM", "ASI", "HLD"]
    realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
    forward_tasks = ["REC", "SSR", "CRR"]
    
    backward_anno = []
    realtime_anno = []
    forward_anno = []
    
    for anno in annotations:
        if anno["task"] in args.task:
            if anno["task"] in backward_tasks:
                backward_anno.append(anno)
            if anno["task"] in realtime_tasks:
                realtime_anno.append(anno)
            if anno["task"] in forward_tasks:
                forward_anno.append(anno)
    
    # 按rank分片数据
    backward_anno_split = split_data(backward_anno, world_size, rank)
    realtime_anno_split = split_data(realtime_anno, world_size, rank)
    forward_anno_split = split_data(forward_anno, world_size, rank)
    
    anno = {
        "backward": backward_anno_split,
        "realtime": realtime_anno_split,
        "forward": forward_anno_split
    }
    
    if rank == 0:
        logger.info(f"Total samples - Backward: {len(backward_anno)}, "
                   f"Realtime: {len(realtime_anno)}, Forward: {len(forward_anno)}")
        logger.info(f"[Rank {rank}] Processing - Backward: {len(backward_anno_split)}, "
                   f"Realtime: {len(realtime_anno_split)}, Forward: {len(forward_anno_split)}")
    
    #########################################################################################
    # 初始化模型（根据device参数传递给模型）
    model = initialize_model(args, device, rank)
    
    ######################################################################
    # 同步所有进程
    dist.barrier()
    
    # 运行推理
    results = run_inference(model, anno, args, rank, world_size)
    
    # 收集结果
    if rank == 0:
        logger.info(f"[Rank {rank}] Gathering results from all ranks...")
    
    all_results = gather_results_pipeline(results, rank, world_size)
    # Rank 0 保存合并后的结果
    if rank == 0:
        save_merged_results(all_results, args)
    
    dist.destroy_process_group()


def initialize_model(args, device, rank):
    """初始化模型"""
    # 根据模型类型初始化
    if args.model == "GPT":
        from models.GPT import EvalGPT
        assert args.gpt_api is not None, "GPT API key is required"
        model = EvalGPT(args)
        
    elif args.model == "Gemini":
        from models.Gemini import EvalGemini
        assert args.gemini_project is not None, "Gemini project is required"
        model = EvalGemini(args)
        
    elif args.model == "InternVL2":
        from models.InternVL2 import EvalInternVL2
        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
        model = EvalInternVL2(args)
        
    elif args.model == "QWen2VL_7B" or args.model == "QWen2VL_72B":
        from models.QWen2VL import EvalQWen2VL
        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
        model = EvalQWen2VL(args)
        
    elif args.model == "LongVU":
        from models.LongVU import EvalLongVU
        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
        model = EvalLongVU(args)
        
    elif args.model == "LLaVA_OneVision":
        from models.LLaVA_OneVision import EvalLLaVAOneVision
        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
        model = EvalLLaVAOneVision(args)
        
    elif args.model == "LLaVA_Video":
        from models.LLaVA_Video import EvalLLaVAVideo
        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
        model = EvalLLaVAVideo(args)
        
    elif args.model == "videollm_online":
        from models.VideoLLM_Online import EvalVideollmOnline
        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
        model = EvalVideollmOnline(args)
        
    elif args.model == "FlashVStream":
        from models.FlashVStream import EvalFlashVStream
        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
        model = EvalFlashVStream(args)
        
    elif args.model == "MiniCPM_o":
        from models.MiniCPM_o import EvalMiniCPM
        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
        model = EvalMiniCPM(args)
        
    elif args.model == "Dispider":
        from models.Dispider import EvalDispider
        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
        model = EvalDispider(args)
        
    elif args.model == "rekv":
        from models.rekv import Evalrekv
        model = Evalrekv(args)
        
    else:
        raise ValueError(f"Unsupported model: {args.model}. Please implement the model.")
    
    if rank == 0:
        logger.info(f"Model {args.model} initialized on device {device}")
    
    return model


def split_data(data_list, world_size, rank):
    """按rank分片数据"""
    if len(data_list) == 0:
        return []
    
    # 计算每个rank处理的数据范围
    total_samples = len(data_list)
    samples_per_rank = (total_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, total_samples)
    
    return data_list[start_idx:end_idx]


def run_inference(model, anno, args, rank, world_size):
    """运行推理"""
    backward_results = []
    realtime_results = []
    forward_results = []
    
    # 处理backward任务
    if len(anno["backward"]) > 0:
        desc = f"[Rank {rank}/{world_size}] Backward Tasks"
        for _anno_ in tqdm(anno["backward"], desc=desc, disable=rank != 0):
            try:
                result = process_backward_or_realtime(_anno_, model, args)
                backward_results.append(result)
            except Exception as e:
                logger.error(f"[Rank {rank}] Error processing backward sample {_anno_['id']}: {e}")
                # 添加失败的结果
                result = {
                    "id": _anno_["id"],
                    "video": _anno_["video"],
                    "task": _anno_["task"],
                    "question": _anno_["question"],
                    "response": None,
                    "ground_truth": chr(65 + _anno_["gt"]),
                    "error": str(e)
                }
                backward_results.append(result)
    
    # 处理realtime任务
    if len(anno["realtime"]) > 0:
        desc = f"[Rank {rank}/{world_size}] Realtime Tasks"
        for _anno_ in tqdm(anno["realtime"], desc=desc, disable=rank != 0):
            try:
                result = process_backward_or_realtime(_anno_, model, args)
                realtime_results.append(result)
            except Exception as e:
                logger.error(f"[Rank {rank}] Error processing realtime sample {_anno_['id']}: {e}")
                result = {
                    "id": _anno_["id"],
                    "video": _anno_["video"],
                    "task": _anno_["task"],
                    "question": _anno_["question"],
                    "response": None,
                    "ground_truth": chr(65 + _anno_["gt"]),
                    "error": str(e)
                }
                realtime_results.append(result)
    
    # 处理forward任务
    if len(anno["forward"]) > 0:
        desc = f"[Rank {rank}/{world_size}] Forward Tasks"
        for _anno_ in tqdm(anno["forward"], desc=desc, disable=rank != 0):
            try:
                result = process_forward(_anno_, model, args)
                forward_results.append(result)
            except Exception as e:
                logger.error(f"[Rank {rank}] Error processing forward sample {_anno_['id']}: {e}")
                # 为forward任务添加错误标记
                _anno_copy = _anno_.copy()
                for i in range(len(_anno_copy.get("test_info", []))):
                    _anno_copy["test_info"][i]["response"] = None
                    _anno_copy["test_info"][i]["error"] = str(e)
                forward_results.append(_anno_copy)
    
    if rank == 0:
        logger.info(f"[Rank {rank}] Processed - Backward: {len(backward_results)}, "
                   f"Realtime: {len(realtime_results)}, Forward: {len(forward_results)}")
    
    return {
        "backward": backward_results,
        "realtime": realtime_results,
        "forward": forward_results
    }


def process_backward_or_realtime(_anno_, model, args):
    """处理backward或realtime任务"""
    id = _anno_["id"]
    video = _anno_["video"]
    task = _anno_["task"]
    question = _anno_["question"]
    options = _anno_["options"]
    
    assert question is not None
    assert options is not None
    
    prompt = model.build_prompt(task=task, question=question, options=options, _anno_=None, index=None)
    chunk_video_path = os.path.join(args.chunked_dir, f"{id}.mp4")
    
    assert os.path.exists(chunk_video_path), f"Video not found: {chunk_video_path}"
    
    response = model.inference(chunk_video_path, prompt)
    
    result = {
        "id": id,
        "video": video,
        "task": task,
        "question": question,
        "response": response,
        "ground_truth": chr(65 + _anno_["gt"])
    }
    
    return result


def process_forward(_anno_, model, args):
    """处理forward任务"""
    id = _anno_["id"]
    task = _anno_["task"]
    test_info = _anno_["test_info"]
    
    _anno_copy = _anno_.copy()
    
    for i in range(len(test_info)):
        prompt = model.build_prompt(task=task, question=None, options=None, _anno_=_anno_, index=i)
        chunk_video_path = os.path.join(args.chunked_dir, f"{id}_{i}.mp4")
        
        assert os.path.exists(chunk_video_path), f"Video not found: {chunk_video_path}"
        
        response = model.inference(chunk_video_path, prompt)
        _anno_copy["test_info"][i]["response"] = response
    
    return _anno_copy

def gather_results_pipeline(results, rank, world_size):
    """
    Pipeline式all_gather - 最优雅的PyTorch原生方案
    
    策略：
    1. 每个rank统计自己的数据量
    2. 使用all_gather交换元数据
    3. 使用send/recv点对点传输大数据
    4. 避免gather_object的全局同步瓶颈
    """
    import pickle
    
    # 1. 序列化数据
    results_bytes = pickle.dumps(results, protocol=pickle.HIGHEST_PROTOCOL)
    local_size = len(results_bytes)
    
    # 2. 交换所有rank的数据大小（小数据，很快）
    size_tensor = torch.tensor([local_size], dtype=torch.long, device='cpu')
    all_sizes = [torch.zeros(1, dtype=torch.long, device='cpu') for _ in range(world_size)]
    
    dist.all_gather(all_sizes, size_tensor)
    
    all_sizes = [s.item() for s in all_sizes]
    total_size_mb = sum(all_sizes) / (1024**2)
    
    if rank == 0:
        logger.info(f"Total data: {total_size_mb:.2f} MB across {world_size} ranks")
        max_size_mb = max(all_sizes) / (1024**2)
        logger.info(f"Max rank size: {max_size_mb:.2f} MB")
    
    # 3. Pipeline传输：rank 0逐个从其他rank接收
    if rank == 0:
        all_data = []
        
        # 首先处理自己的数据
        all_data.append(results)
        
        # 接收其他rank的数据
        for src_rank in range(1, world_size):
            # 准备接收buffer
            recv_size = all_sizes[src_rank]
            recv_buffer = torch.zeros(recv_size, dtype=torch.uint8, device='cpu')
            
            # 接收数据
            logger.info(f"Receiving from rank {src_rank} ({recv_size / (1024**2):.2f} MB)...")
            dist.recv(recv_buffer, src=src_rank, tag=src_rank)
            
            # 反序列化
            recv_bytes = recv_buffer.numpy().tobytes()
            rank_results = pickle.loads(recv_bytes)
            all_data.append(rank_results)
            
            logger.info(f"✓ Received from rank {src_rank} ({src_rank}/{world_size-1})")
        
        # 合并所有结果
        merged = {"backward": [], "realtime": [], "forward": []}
        
        for rank_results in all_data:
            merged["backward"].extend(rank_results["backward"])
            merged["realtime"].extend(rank_results["realtime"])
            merged["forward"].extend(rank_results["forward"])
        
        logger.info(f"✓ Final results - Backward: {len(merged['backward'])}, "
                   f"Realtime: {len(merged['realtime'])}, Forward: {len(merged['forward'])}")
        
        return merged
        
    else:
        # 其他rank发送数据给rank 0
        send_buffer = torch.tensor(list(results_bytes), dtype=torch.uint8, device='cpu')
        
        logger.info(f"[Rank {rank}] Sending {local_size / (1024**2):.2f} MB to rank 0...")
        dist.send(send_buffer, dst=0, tag=rank)
        logger.info(f"[Rank {rank}] Send completed")
        
        return None


def save_merged_results(results, args):
    """保存合并后的结果"""
    if args.save_results:
        result_dir = Path(args.result_dir) / args.model
        result_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = result_dir / f"{args.model}_{'_'.join(args.task)}_{args.mode}_distributed.json"
        
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to: {result_file}")
        logger.info(f"Total samples saved - Backward: {len(results['backward'])}, "
                   f"Realtime: {len(results['realtime'])}, Forward: {len(results['forward'])}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run OVBench with Distributed Inference')
    
    # 数据路径参数
    parser.add_argument("--anno_path", type=str, default="data/ovo_bench_new.json", 
                       help="Path to the annotations")
    parser.add_argument("--video_dir", type=str, default="data/src_videos", 
                       help="Root directory of source videos")
    parser.add_argument("--chunked_dir", type=str, default="data/chunked_videos", 
                       help="Root directory of chunked videos")
    parser.add_argument("--result_dir", type=str, default="results", 
                       help="Root directory of results")
    
    # 任务参数
    parser.add_argument("--mode", type=str, required=True, choices=["online", "offline"], 
                       help="Online or Offline model for testing")
    parser.add_argument("--task", type=str, required=False, nargs="+",
                       choices=["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"],
                       default=["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"],
                       help="Tasks to evaluate")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
    parser.add_argument("--model_path", type=str, required=False, default=None,
                       help="Path to the model checkpoint")
    parser.add_argument("--save_results", type=bool, default=True, 
                       help="Save results to a file")
    
    # API参数（用于GPT和Gemini）
    parser.add_argument("--gpt_api", type=str, required=False, default=None,
                       help="GPT API key")
    parser.add_argument("--gemini_project", type=str, required=False, default=None,
                       help="Gemini project name")
    
    # ReKV相关参数
    parser.add_argument("--retrieve_size", type=int, required=False, default=64, 
                       help="Retrieval window size for ReKV and related models")
    
    # 分布式参数
    parser.add_argument("--global_seed", type=int, default=42,
                       help="Global random seed")
    parser.add_argument("--tf32", action="store_true", 
                       help="Enable TF32 acceleration")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()

