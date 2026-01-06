from utils.data_execution import load_data
from src.model.modelclass import Model
from benchmark.Benchmark import Benchmark

import argparse
import multiprocessing as mp
import torch
import os
import json
from typing import List, Dict, Any
import math

def split_data(data, num_chunks: int) -> List:
    """将数据切分成多个块"""
    chunk_size = math.ceil(len(data) / num_chunks)
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def merge_results(result_files: List[str], output_file: str):
    """合并多个 JSON 数组格式的临时文件"""
    all_results = []
    for result_file in result_files:
        print(f"Merging file: {result_file}")
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)  # 整个文件是一个 JSON 数组
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    print(f"[Warning] {result_file} is not a JSON array, skipping.")
        except Exception as e:
            print(f"[Error] Failed to read {result_file}: {e}")
            continue

    # 写入最终输出文件（JSON 数组格式）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 清理临时文件
    for temp_file in result_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def worker_process(worker_id: int, gpu_id: int, data_chunk: List, args_dict: Dict, temp_output_file: str):
    """工作进程函数"""
    try:
        # 设置GPU设备
        torch.cuda.set_device(gpu_id)
        
        # 重新解析参数
        class Args:
            def __init__(self, args_dict):
                for key, value in args_dict.items():
                    setattr(self, key, value)
        
        args = Args(args_dict)
        
        print(f"Worker {worker_id} started on GPU {gpu_id}, processing {len(data_chunk)} samples")
        
        ####### MODEL ############
        model = Model()
        
        if args.model_name == "MiniCPM-V":
            from src.model.MiniCPMV import MiniCPMV
            model = MiniCPMV()
        if args.model_name == "MiniCPMo":
            print(f"Worker {worker_id}: MiniCPMo loaded")
            from src.model.MiniCPMo import MiniCPMo
            model = MiniCPMo()
        if args.model_name == "rekv":
            from src.model.rekv import rekv
            model = rekv()
        
        # 将模型移动到指定GPU
        if hasattr(model, 'model'):
            model.model = model.model.cuda(gpu_id)
        elif hasattr(model, 'cuda'):
            model.cuda(gpu_id)
        
        ####### BENCHMARK #######
        benchmark = Benchmark(data_chunk)
        
        if args.benchmark_name == "Streaming":
            from benchmark.StreamingBench import StreamingBench
            benchmark = StreamingBench(data_chunk)
        if args.benchmark_name == "StreamingProactive":
            from benchmark.StreamingBenchProactive import StreamingBenchProactive
            benchmark = StreamingBenchProactive(data_chunk)
        if args.benchmark_name == "StreamingSQA":
            from benchmark.StreamingBenchSQA import StreamingBenchSQA
            benchmark = StreamingBenchSQA(data_chunk)
        if args.benchmark_name == "StreamingOpenStreamText":
            from benchmark.StreamingOpenStreamText import StreamingOpenStreamText
            benchmark = StreamingOpenStreamText(data_chunk)
        
        # 执行评估，输出到临时文件
        benchmark.eval(data_chunk, model, temp_output_file, args.context_time)
        print(f"Worker {worker_id} completed, results saved to {temp_output_file}")
        
    except Exception as e:
        print(f"Worker {worker_id} error: {str(e)}")
        raise

def main(args):
    data = load_data(args.data_file)
    print(f"Loaded {len(data)} samples")
    
    # 确定使用的GPU数量
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs")
    else:
        num_gpus = 1
        print("No GPU found, using CPU only")
    
    # 如果只有1个GPU或者数据量很小，直接运行单进程版本
    if num_gpus <= 1 or len(data) < 8:
        print("Using single process mode")
        run_single_process(args, data)
        return
    
    # 多进程并行处理
    print(f"Using multi-process mode with {num_gpus} workers")
    
    # 切分数据
    data_chunks = split_data(data, num_gpus)
    
    # 准备参数
    args_dict = {
        'model_name': args.model_name,
        'benchmark_name': args.benchmark_name,
        'context_time': args.context_time
    }
    
    # 创建进程
    processes = []
    temp_output_files = []
    
    for i in range(num_gpus):
        gpu_id = i % num_gpus
        temp_output_file = f"{args.output_file}.temp_{i}"
        temp_output_files.append(temp_output_file)
        
        # 如果数据块为空，跳过
        if i >= len(data_chunks) or len(data_chunks[i]) == 0:
            continue
            
        p = mp.Process(
            target=worker_process,
            args=(i, gpu_id, data_chunks[i], args_dict, temp_output_file)
        )
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 检查是否有进程失败
    for i, p in enumerate(processes):
        if p.exitcode != 0:
            print(f"Warning: Worker {i} exited with code {p.exitcode}")
    
    # 合并结果
    print("Merging results...")
    merge_results(temp_output_files, args.output_file)
    print(f"Final results saved to {args.output_file}")

def run_single_process(args, data):
    """单进程版本，保持原有逻辑"""
    ####### BENCHMARK #######
    benchmark = Benchmark(data)

    if args.benchmark_name == "Streaming":
        from benchmark.StreamingBench import StreamingBench
        benchmark = StreamingBench(data)
    if args.benchmark_name == "StreamingProactive":
        from benchmark.StreamingBenchProactive import StreamingBenchProactive
        benchmark = StreamingBenchProactive(data)
    if args.benchmark_name == "StreamingSQA":
        from benchmark.StreamingBenchSQA import StreamingBenchSQA
        benchmark = StreamingBenchSQA(data)
    if args.benchmark_name == "StreamingOpenStreamText":
        from benchmark.StreamingOpenStreamText import StreamingOpenStreamText
        benchmark = StreamingOpenStreamText(data)

    ####### MODEL ############
    model = Model()
 
    if args.model_name == "MiniCPM-V":
        from src.model.MiniCPMV import MiniCPMV
        model = MiniCPMV()
    if args.model_name == "MiniCPMo":
        print("MiniCPMo loaded")
        from src.model.MiniCPMo import MiniCPMo
        model = MiniCPMo()
    if args.model_name == "rekv":
        from src.model.rekv import rekv
        model = rekv()

    benchmark.eval(data, model, args.output_file, args.context_time)

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--benchmark_name", type=str, required=True, help="Name of the benchmark")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--context_time", type=int, required=True, help="Time before the query")
    args = parser.parse_args()
    main(args)