import torch
import numpy as np
from logzero import logger
from decord import VideoReader, cpu

from video_qa.base import BaseVQA, work


class ReKVStreamVQA(BaseVQA):
    def load_video(self, video_path):
        if video_path.endswith('.npy'):  # FPS=1
            video = np.load(video_path)
            assert self.sample_fps <= 1
            num_frames = len(video)
            frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
            video = video[frame_idx]
        else:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
            video = vr.get_batch(frame_idx).asnumpy()
        return video

    def video_open_qa(self, question, max_new_tokens=1024):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        video_path = video_sample['video_path']
        video_start_idx = video_end_idx = 0
        video = self.load_video(video_path)
        video_tensor = torch.from_numpy(video)

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']

            temporal_windows = torch.tensor([sample['start_time'], sample['end_time']]) * self.sample_fps
            temporal_windows = temporal_windows.tolist()

            # encode video until receiving QA
            if temporal_windows[-1] > video_end_idx:
                video_end_idx = temporal_windows[-1]
                self.qa_model.encode_video(video_tensor[int(video_start_idx):int(video_end_idx)])
                video_start_idx = video_end_idx
        
            # OpenQA
            qa_results = self.video_open_qa(question, max_new_tokens=256)
            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'question': question,
                'answer': answer,
                'pred_answer': qa_results['pred_answer'],
            })
 

if __name__ == "__main__":
    work(ReKVStreamVQA)
