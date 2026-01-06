import torch
import time
import argparse
from PIL import Image
import numpy as np
from models.dispider.model.builder import load_pretrained_model
from models.dispider.mm_utils import process_images
from models.dispider.constants import IMAGE_TOKEN_INDEX

def create_dummy_frames(num_frames=16, image_size=336):
    """
    创建测试用的假图像帧
    
    参数:
        num_frames: 帧数量，默认16帧
        image_size: 图像尺寸
    """
    frames = []
    for i in range(num_frames):
        # 创建随机RGB图像
        img = Image.fromarray(
            np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        )
        frames.append(img)
    return frames

def benchmark_vit_encoding(model, image_processor, frames, device, num_warmup=3, num_runs=10):
    """
    测试VIT编码时间
    
    参数:
        model: 加载的模型
        image_processor: 图像处理器
        frames: 输入帧列表
        device: 设备
        num_warmup: 预热次数
        num_runs: 测试运行次数
    """
    print(f"\n{'='*60}")
    print(f"测试VIT编码性能 - 处理 {len(frames)} 帧")
    print(f"{'='*60}")
    
    # 获取vision tower和compressor
    compressor = model.get_compressor()
    vision_tower = compressor.compressor.get_vision_tower()
    
    # 预处理图像
    image_tensor = process_images(frames, image_processor[0], model.config)
    if type(image_tensor) is list:
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float16)
    
    print(f"图像张量形状: {image_tensor.shape}")
    
    # 预热
    print(f"\n预热运行 {num_warmup} 次...")
    with torch.no_grad():
        for i in range(num_warmup):
            _ = vision_tower(image_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # 正式测试
    print(f"正式测试运行 {num_runs} 次...")
    vit_times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            clip_features = vision_tower(image_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = (end_time - start_time) * 1000  # 转换为毫秒
            vit_times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f} ms")
    
    # 统计结果
    avg_time = np.mean(vit_times)
    std_time = np.std(vit_times)
    min_time = np.min(vit_times)
    max_time = np.max(vit_times)
    
    print(f"\n{'='*60}")
    print(f"VIT编码时间统计 ({len(frames)} 帧):")
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  标准差:   {std_time:.2f} ms")
    print(f"  最小时间: {min_time:.2f} ms")
    print(f"  最大时间: {max_time:.2f} ms")
    print(f"  每帧平均: {avg_time/len(frames):.2f} ms")
    print(f"{'='*60}")
    
    return clip_features, image_tensor, avg_time

def benchmark_llm_prefilling(model, tokenizer, clip_features, image_tensor, device, num_warmup=3, num_runs=10):
    """
    测试LLM prefilling时间（通过Qwen2模型的embed和forward）
    
    参数:
        model: 加载的模型
        tokenizer: 分词器（实际上用compressor的tokenizer）
        clip_features: VIT编码后的特征
        image_tensor: 原始图像张量
        device: 设备
        num_warmup: 预热次数
        num_runs: 测试运行次数
    """
    print(f"\n{'='*60}")
    print(f"测试LLM Prefilling性能")
    print(f"{'='*60}")
    
    # 获取compressor和相关组件
    compressor = model.get_compressor()
    time_tokenizer = compressor.tokenizer
    
    # 准备文本输入 - 使用compressor的tokenizer
    prompt = "Describe what is happening in this video."
    text_tokens = time_tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    
    # 构造带图像token的输入
    image_token_tensor = torch.tensor([IMAGE_TOKEN_INDEX], dtype=torch.long).to(device)
    input_ids = torch.cat([image_token_tensor.unsqueeze(0), text_tokens], dim=1)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"输入tokens长度: {input_ids.shape[1]}")
    print(f"CLIP特征形状: {clip_features.shape}")
    
    # 准备模型输入 - 获取embeddings
    # 我们需要手动准备所有需要的参数
    try:
        # 创建一个简单的输入来测试LLM prefilling
        # 这里我们只测试LLM主干网络的prefilling，不包含完整的多模态处理
        
        # 使用纯文本来测试LLM prefilling
        test_prompt = "This is a test prompt for benchmarking the LLM prefilling performance with multiple tokens."
        test_tokens = tokenizer(test_prompt, return_tensors='pt').input_ids.to(device)
        test_attention_mask = torch.ones_like(test_tokens)
        
        # 添加更多tokens来模拟实际的输入长度（包括图像特征的tokens）
        # 假设16帧经过处理后大约有 16 * 576 / 4 = 2304 个tokens（基于mix_spatial_tokens）
        num_image_tokens = clip_features.shape[0] * clip_features.shape[1] // 4
        print(f"估计图像tokens数量: {num_image_tokens}")
        
        # 创建更长的输入来模拟完整的prefilling
        additional_tokens = torch.ones((1, num_image_tokens), dtype=torch.long, device=device) * tokenizer.pad_token_id
        full_input_ids = torch.cat([test_tokens, additional_tokens], dim=1)
        full_attention_mask = torch.ones_like(full_input_ids)
        
        print(f"完整输入长度（模拟图像+文本）: {full_input_ids.shape[1]} tokens")
        
        # 预热
        print(f"\n预热运行 {num_warmup} 次...")
        with torch.no_grad():
            for i in range(num_warmup):
                # 获取embeddings
                inputs_embeds = model.get_model().embed_tokens(full_input_ids)
                
                # 调用Qwen2的forward
                _ = model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_attention_mask,
                    use_cache=True,
                    return_dict=True
                )
                if device == 'cuda':
                    torch.cuda.synchronize()
        
        # 正式测试
        print(f"正式测试运行 {num_runs} 次...")
        prefill_times = []
        
        with torch.no_grad():
            for i in range(num_runs):
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                # 获取embeddings
                inputs_embeds = model.get_model().embed_tokens(full_input_ids)
                
                # LLM prefilling - 第一次forward pass生成KV cache
                outputs = model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_attention_mask,
                    use_cache=True,
                    return_dict=True
                )
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000  # 转换为毫秒
                prefill_times.append(elapsed)
                print(f"  Run {i+1}: {elapsed:.2f} ms")
        
        # 统计结果
        avg_time = np.mean(prefill_times)
        std_time = np.std(prefill_times)
        min_time = np.min(prefill_times)
        max_time = np.max(prefill_times)
        
        print(f"\n{'='*60}")
        print(f"LLM Prefilling时间统计:")
        print(f"  序列长度: {full_input_ids.shape[1]} tokens")
        print(f"  平均时间: {avg_time:.2f} ms")
        print(f"  标准差:   {std_time:.2f} ms")
        print(f"  最小时间: {min_time:.2f} ms")
        print(f"  最大时间: {max_time:.2f} ms")
        print(f"  平均每token: {avg_time/full_input_ids.shape[1]:.3f} ms")
        print(f"{'='*60}")
        
        return avg_time
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n尝试简化版本的测试...")
        
        # 简化版：只测试一个较短的序列
        simple_prompt = "Test prompt."
        simple_tokens = tokenizer(simple_prompt, return_tensors='pt').input_ids.to(device)
        simple_attention_mask = torch.ones_like(simple_tokens)
        
        print(f"使用简化输入，长度: {simple_tokens.shape[1]} tokens")
        
        # 预热
        print(f"\n预热运行 {num_warmup} 次...")
        with torch.no_grad():
            for i in range(num_warmup):
                inputs_embeds = model.get_model().embed_tokens(simple_tokens)
                _ = model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=simple_attention_mask,
                    use_cache=True
                )
                if device == 'cuda':
                    torch.cuda.synchronize()
        
        # 正式测试
        print(f"正式测试运行 {num_runs} 次...")
        prefill_times = []
        
        with torch.no_grad():
            for i in range(num_runs):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                
                inputs_embeds = model.get_model().embed_tokens(simple_tokens)
                outputs = model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=simple_attention_mask,
                    use_cache=True
                )
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000
                prefill_times.append(elapsed)
                print(f"  Run {i+1}: {elapsed:.2f} ms")
        
        avg_time = np.mean(prefill_times)
        print(f"\n简化版LLM Prefilling平均时间: {avg_time:.2f} ms")
        print(f"注意: 这只是{simple_tokens.shape[1]} tokens的测试")
        
        return avg_time

def main():
    parser = argparse.ArgumentParser(description='测量模型处理16帧时VIT和LLM prefilling的性能')
    parser.add_argument('--model-path', type=str, default='/mnt/data0/public/back/huggingface/hub/Dispider',
                        help='模型路径')
    parser.add_argument('--model-name', type=str, default='long-qwen',
                        help='模型名称')
    parser.add_argument('--num-frames', type=int, default=16,
                        help='测试的帧数量（默认16）')
    parser.add_argument('--image-size', type=int, default=336,
                        help='图像尺寸（默认336）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备（cuda或cpu）')
    parser.add_argument('--num-warmup', type=int, default=3,
                        help='预热次数（默认3）')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='测试运行次数（默认10）')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"性能测试配置")
    print(f"{'='*60}")
    print(f"模型路径: {args.model_path}")
    print(f"模型名称: {args.model_name}")
    print(f"测试帧数: {args.num_frames}")
    print(f"图像尺寸: {args.image_size}x{args.image_size}")
    print(f"设备: {args.device}")
    print(f"预热次数: {args.num_warmup}")
    print(f"测试次数: {args.num_runs}")
    print(f"{'='*60}")
    
    # 加载模型
    print("\n加载模型中...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None,
        args.model_name,
        device=args.device
    )
    print("模型加载完成！")
    
    model.eval()
    
    # 创建测试帧
    print(f"\n创建 {args.num_frames} 个测试帧...")
    frames = create_dummy_frames(args.num_frames, args.image_size)
    
    # 测试VIT编码
    clip_features, image_tensor, vit_time = benchmark_vit_encoding(
        model, image_processor, frames, args.device,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs
    )
    
    # 测试LLM prefilling
    llm_time = benchmark_llm_prefilling(
        model, tokenizer, clip_features, image_tensor, args.device,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs
    )
    
    # 总结
    print(f"\n{'='*60}")
    print(f"性能测试总结 ({args.num_frames} 帧)")
    print(f"{'='*60}")
    print(f"VIT编码平均时间:        {vit_time:.2f} ms")
    print(f"LLM Prefilling平均时间:  {llm_time:.2f} ms")
    print(f"总计:                   {vit_time + llm_time:.2f} ms")
    print(f"{'='*60}\n")
    
    # 输出到文件
    output_file = f"benchmark_results_{args.num_frames}frames.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"性能测试结果\n")
        f.write(f"{'='*60}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"测试帧数: {args.num_frames}\n")
        f.write(f"图像尺寸: {args.image_size}x{args.image_size}\n")
        f.write(f"设备: {args.device}\n")
        f.write(f"测试次数: {args.num_runs}\n\n")
        f.write(f"VIT编码平均时间:        {vit_time:.2f} ms\n")
        f.write(f"LLM Prefilling平均时间:  {llm_time:.2f} ms\n")
        f.write(f"总计:                   {vit_time + llm_time:.2f} ms\n")
    
    print(f"结果已保存到: {output_file}")

if __name__ == '__main__':
    main()