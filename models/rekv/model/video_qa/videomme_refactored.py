"""VideoMME专用solver - 支持时间和内存统计"""
import torch
from logzero import logger
from .rekv_offline_refactored import ReKVOfflineVQA


class VideoMMEReKVOfflineVQA(ReKVOfflineVQA):
    """VideoMME数据集专用 - 增加时间/内存统计"""
    
    def __init__(self, model, processor, args):
        super().__init__(model, processor, args)
        # 初始化统计变量
        self.acc_time = 0.0
        self.max_mem = 0.0
    
    def encode_video(self, video):
        """编码视频 - 增加性能统计"""
        self.model.clear_cache()
        self.model.encode_init_prompt()
        
        # 开始性能监控
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        gpu_start_event = torch.cuda.Event(enable_timing=True)
        gpu_end_event = torch.cuda.Event(enable_timing=True)
        gpu_start_event.record()
        
        # 执行编码
        self.model.encode_video(video)
        
        # 结束性能监控
        gpu_end_event.record()
        torch.cuda.synchronize()
        
        # 记录统计信息
        gpu_time = gpu_start_event.elapsed_time(gpu_end_event) / 1000.0
        gen_max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        self.acc_time += gpu_time
        self.max_mem = max(gen_max_mem, self.max_mem)
        
        logger.debug(f"Video encoding: {gpu_time:.2f}s, mem: {gen_max_mem:.1f}MB")
        logger.debug(f"Accumulated time: {self.acc_time:.2f}s, max mem: {self.max_mem:.1f}MB")
    
    def answer_questions(self, video_sample):
        """批量回答问题 - 重置统计"""
        # 每个视频重置统计
        self.acc_time = 0.0
        self.max_mem = 0.0
        
        # 保存视频级别的信息（如duration）
        self.current_video_info = {
            'duration': video_sample.get('duration'),
        }
        
        return super().answer_questions(video_sample)
    
    def _format_mc_result(self, pred, qa_pair, video_id):
        """格式化多选题结果 - 添加duration字段和正确选项处理"""
        result = super()._format_mc_result(pred, qa_pair, video_id)
        
        # VideoMME特殊处理：answer直接是选项字母（如'A','B'），不需要转换
        answer = qa_pair.get('answer')
        if answer and answer in self.choice_letters:
            result['correct_choice'] = answer
            result['qa_acc'] = float(result['pred_choice'] == answer) * 100
        
        # 添加视频duration字段
        if hasattr(self, 'current_video_info') and self.current_video_info.get('duration'):
            result['duration'] = self.current_video_info['duration']
        
        return result

