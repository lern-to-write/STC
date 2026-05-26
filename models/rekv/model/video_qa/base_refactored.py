"""重构的BaseVQA - 简洁优雅的视频问答基类"""
import re
import torch
from logzero import logger
from .utils.data_utils import chunk_video

import torch.distributed as dist
from decord import VideoReader, cpu

class BaseVQA:
    """视频问答基类 - 所有函数<15行"""
    
    choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    def __init__(self, model, processor, args):
        self.model = model
        self.processor = processor
        self.args = args
        self.results = []
    
    def __call__(self, video_sample):
        """前向传播 - 处理一个视频样本"""
        video = self.load_video(video_sample['video_path'], self.args.sample_fps)
        video_tensor = self._to_tensor(video)
        self.encode_video(video_tensor)
        return self.answer_questions(video_sample)
    def load_video(self,video_path, sample_fps=1):
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), int(fps / sample_fps))]
        video = vr.get_batch(frame_idx).asnumpy()
        logger.debug(f'Loaded video: {video.shape}')
        return video
    def _to_tensor(self, video):
        """转换为tensor"""
        if isinstance(video, torch.Tensor):
            return video
        return torch.from_numpy(video)
    
    def encode_video(self, video):
        """编码视频为KV缓存"""
        self.model.clear_cache()
        self.model.encode_init_prompt()
        self.model.encode_video(video)
        
        ########################################
        rank = dist.get_rank()
        if rank == 0:
            logger.debug(f'Video encoded, cache size: {self._get_cache_size():.1f} GB')
        ########################################
    
    def _get_cache_size(self):
        """获取缓存大小（GB）"""
        return self.model.calc_memory_usage() / (1024**3)
    
    def answer_questions(self, video_sample):
        """批量回答问题"""
        results = []
        for qa in video_sample['conversations']:
            result = self.answer_single(qa, video_sample['video_id'])
            results.append(result)
            self.results.append(result)
        return results
    
    def answer_single(self, qa_pair, video_id):
        """回答单个问题 - 子类实现"""
        raise NotImplementedError
    
    def format_mcqa_prompt(self, question, choices):
        """格式化多选题提示"""
        formatted_choices = "\n".join([
            f"({self.choice_letters[i]}) {choice}" 
            for i, choice in enumerate(choices)
        ])
        formatted_q = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."
        return self.model.get_prompt(formatted_q, mc=True)
    
    def format_openqa_prompt(self, question):
        """格式化开放式问题提示"""
        return self.model.get_prompt(question)
    
    def extract_choice(self, pred_text):
        """从预测文本提取选项"""
        pred_text = pred_text.strip()
        if ")" in pred_text:
            idx = pred_text.index(")")
            return pred_text[idx - 1:idx]
        return pred_text[0] if pred_text else 'A'
    
    def save_results(self, save_path):
        """保存结果到CSV"""
        import pandas as pd
        
        from pathlib import Path

        # In your save_results method
        save_dir = Path('results/eval')
        df = pd.DataFrame(self.results)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved {len(self.results)} results to {save_path}")