import warnings
import random
import json
import os
import math
import argparse

import pandas as pd
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import (
    logging,
    LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor,
    VideoLlavaForConditionalGeneration, VideoLlavaProcessor
)
import logzero
from logzero import logger

from model import llava_onevision_rekv, video_llava_rekv, longva_rekv


MODELS = {
    'llava_ov_0.5b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-0.5b-ov-hf',
    },
    'llava_ov_7b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-7b-ov-hf',
    },
    'llava_ov_72b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-72b-ov-hf',
    },
    'video_llava_7b': {
        'load_func': video_llava_rekv.load_model,
        'model_class': VideoLlavaForConditionalGeneration,
        'processor_class': VideoLlavaProcessor,
        'model_path': 'model_zoo/Video-LLaVA-7B-hf',
    },
    'longva_7b': {
        'load_func': longva_rekv.load_model,
        'model_path': 'model_zoo/LongVA-7B',
    },
}


class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None,
                 retrieve_size=64, chunk_size=1) -> None:
        
        self.sample_fps = sample_fps

        self.qa_model = qa_model
        self.qa_processor = qa_processor

        # Retrieval Hyperparams
        assert chunk_size <= retrieve_size, f'chunk_size: {chunk_size}, retrieve_size: {retrieve_size}'
        self.retrieve_size = retrieve_size
        self.chunk_size = chunk_size

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        self.anno = anno
        self.eval_grounding = 'temporal_windows' in anno[0]['conversations'][0]

        self.save_dir = save_dir
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.record = {(self.retrieve_size, self.chunk_size): []}

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def load_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
        video = vr.get_batch(frame_idx).asnumpy()
        logger.debug(f'video shape: {video.shape}')
        return video
    
    def calc_recall_precision(self, gt_temporal_windows, retrieved_mask):
        total_intersection_length = 0.0
    
        for (start_sec, end_sec) in gt_temporal_windows:
            start = math.floor(start_sec)
            end = math.ceil(end_sec)
            for i in range(start, end):
                if i < len(retrieved_mask) and retrieved_mask[i]:
                    intersection_start = max(start_sec, i)
                    intersection_end = min(end_sec, i + 1)
                    total_intersection_length += intersection_end - intersection_start

        gt_len = sum([end_sec - start_sec for start_sec, end_sec in gt_temporal_windows])
        retrieved_len = sum(retrieved_mask).item()

        recall = total_intersection_length / gt_len if gt_len > 0 else 0
        precision = total_intersection_length / retrieved_len if retrieved_len > 0 else 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return recall, precision, f1
    
    def format_mcqa_prompt(self, question, candidates):
        assert len(question) > 0, f"Q: {question}"

        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."

        return {
            "question": f"{question}",
            "formatted_question": formatted_question,
            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
        }

    def extract_characters_regex(self, s):
        s = s.strip()
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            return s[0]

    def video_open_qa(self, question, max_new_tokens=1024):
        pass

    def video_close_qa(self, question, candidates, correct_choice):
        pass

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        pass

    def analyze(self, debug=False):
        video_annos = self.anno[:1] if debug else self.anno
        for video_sample in tqdm(video_annos):
            logger.debug(f'video_id: {video_sample["video_id"]}')
            self.analyze_a_video(video_sample)

        dfs = []
        for (retrieve_size, chunk_size), dict_list in self.record.items():
            df = pd.DataFrame(dict_list)
            df['retrieve_size'] = retrieve_size
            df['chunk_size'] = chunk_size
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv', index=False)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def work(QA_CLASS):
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="llava_ov_7b")
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
    args = parser.parse_args()

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)

    # fix random seed
    random.seed(2024)
    logger.info('seed: 2024')

    # VideoQA model
    model_path = MODELS[args.model]['model_path']
    load_func = MODELS[args.model]['load_func']
    logger.info(f"Loading VideoQA model: {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        n_local=args.n_local,
        topk=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
    )

    # Load ground truth file
    anno = json.load(open(args.anno_path))

    retrieve_analyzer = QA_CLASS(
        anno=anno,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        retrieve_size=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        save_dir=args.save_dir,
    )

    retrieve_analyzer.analyze(debug=args.debug)
