"""流式视频问答 - 重构版本"""
import torch
import numpy as np
from logzero import logger
from decord import VideoReader, cpu
from .base_refactored import BaseVQA


class ReKVStreamVQA(BaseVQA):
    """流式视频问答 - 支持时间窗口的增量编码"""
    
    def __call__(self, video_sample):
        """处理流式视频样本 - 增量编码"""
        video = self._load_stream_video(video_sample['video_path'])
        video_tensor = self._to_tensor(video)
        
        # 初始化
        self.model.clear_cache()
        self.model.encode_init_prompt()
        
        video_start_idx = 0
        video_end_idx = 0
        
        # 逐问题增量编码
        for qa in video_sample['conversations']:
            # 计算需要编码的时间窗口
            temporal_window = self._get_temporal_window(qa)
            
            # 如果需要编码新帧
            if temporal_window[-1] > video_end_idx:
                video_end_idx = temporal_window[-1]
                new_frames = video_tensor[int(video_start_idx):int(video_end_idx)]
                self.model.encode_video(new_frames)
                video_start_idx = video_end_idx
            
            # 回答问题
            result = self.answer_single(qa, video_sample['video_id'])
            self.results.append(result)
        
        return self.results
    
    def _load_stream_video(self, video_path):
        """加载流式视频"""
        if video_path.endswith('.npy'):
            video = np.load(video_path)
            num_frames = len(video)
            fps_ratio = self.args.sample_fps
            assert fps_ratio <= 1, "sample_fps should <= 1 for .npy files"
            frame_idx = np.linspace(0, num_frames-1, int(num_frames*fps_ratio), dtype=int)
            return video[frame_idx]
        else:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), int(fps / self.args.sample_fps))]
            return vr.get_batch(frame_idx).asnumpy()
    
    def _get_temporal_window(self, qa_pair):
        """获取时间窗口（转换为帧索引）"""
        start_time = qa_pair.get('start_time', 0)
        end_time = qa_pair.get('end_time', float('inf'))
        start_idx = start_time * self.args.sample_fps
        end_idx = end_time * self.args.sample_fps
        return [start_idx, end_idx]
    
    def answer_single(self, qa_pair, video_id):
        """回答单个问题 - 流式仅支持开放式问答"""
        question = qa_pair['question']
        prompt = self.format_openqa_prompt(question)
        
        pred = self.model.question_answering(
            {"question": question, "prompt": prompt},
            max_new_tokens=256
        )
        
        return self._format_open_result(pred, qa_pair, video_id)

