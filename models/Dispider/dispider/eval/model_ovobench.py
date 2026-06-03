import argparse
import copy
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from dispider.constants import (
    DEFAULT_ANS_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_TODO_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from dispider.conversation import conv_templates
from dispider.mm_utils import get_model_name_from_path, tokenizer_image_token
from dispider.model.builder import load_pretrained_model


BACKWARD_TASKS = {"EPM", "ASI", "HLD"}
REALTIME_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
FORWARD_TASKS = {"REC", "SSR", "CRR"}
ALL_TASKS = sorted(BACKWARD_TASKS | REALTIME_TASKS | FORWARD_TASKS)

BR_PROMPT_TEMPLATE = """Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C).
Do not include any additional text or explanation in your response.
"""

REC_PROMPT_TEMPLATE = """You're watching a video in which people may perform a certain type of action repetitively.
The person performing this kind of action are referred to as 'they' in the following statement.
Your task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one.
Now, answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3...) indicating the total count.
Do not include any additional text or explanation in your response.
"""

SSR_PROMPT_TEMPLATE = """You're watching a tutorial video which contain a sequential of steps.
The following is one step from the whole procedures:
{}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""

CRR_PROMPT_TEMPLATE = """You're responsible of answering questions based on the video content.
The following question are relevant to the latest frames, i.e. the end of the video.
{}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""


warnings.filterwarnings(
    "ignore",
    message=".*copying from a non-meta parameter.*meta parameter.*no-op.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*do_sample.*is set to.*",
    category=UserWarning,
)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=None):
        super().__init__()
        self.stops = stops or []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq


def get_seq_time(vr, frame_idx, num_clip):
    frm_per_clip = len(frame_idx) // num_clip
    key_frame = [
        [frame_idx[i * frm_per_clip], frame_idx[i * frm_per_clip + frm_per_clip - 1]]
        for i in range(num_clip)
    ]
    time = vr.get_frame_timestamp(key_frame)
    return np.hstack([time[:, 0, 0], time[:, 1, 1]])


def calculate_diff(scene_sep, start_frame):
    diff = [scene_sep[0] - start_frame]
    for i in range(len(scene_sep) - 1):
        diff.append(scene_sep[i + 1] - scene_sep[i])
    return diff


def load_video(vis_path, scene_sep, num_frm=16, max_clip=10000, sample_frame=None):
    block_size = 1
    vr = VideoReader(vis_path, num_threads=1)
    total_frame_num = len(vr) if sample_frame is None else (sample_frame[0][1] - sample_frame[0][0])
    fps = vr.get_avg_fps()
    total_time = total_frame_num / fps
    frame_idx = []

    if len(scene_sep) == 0:
        num_clip = total_time / num_frm
        num_clip = int(block_size * np.round(num_clip / block_size)) if num_clip > block_size else int(np.round(num_clip))
        num_clip = max(num_clip, 1)
        num_clip = min(num_clip, max_clip)
        total_num_frm = num_frm * num_clip
        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    else:
        end_frame = total_frame_num if sample_frame is None else sample_frame[0][1]
        new_scene_sep = []
        for ele in scene_sep:
            sep = int(fps * (ele + 1))
            sep = min(sep, end_frame - 1)
            new_scene_sep.append(sep)
        scene_sep = new_scene_sep + [end_frame - 1]
        if len(scene_sep) > max_clip:
            diff = calculate_diff(scene_sep, start_frame=0)
            min_idx = np.argsort(diff[:-1])[: len(scene_sep) - max_clip]
            for i in np.sort(min_idx)[::-1]:
                del scene_sep[i]
        start = 0
        for end_frame in scene_sep:
            idx_list = np.linspace(start, end_frame, num=num_frm, endpoint=False)
            frame_idx.extend([int(idx) for idx in idx_list])
            start = end_frame
        num_clip = len(scene_sep)
        total_num_frm = num_frm * num_clip

    time_idx = get_seq_time(vr, frame_idx, num_clip)
    img_array = vr.get_batch(frame_idx).asnumpy()

    _, h, w, _ = img_array.shape
    if h != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(min(h, w), min(h, w)))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    return [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)], time_idx, num_clip


def preprocess_time(time, num_clip, tokenizer):
    time = time.reshape(2, num_clip)
    seq = []
    for i in range(num_clip):
        start, end = time[:, i]
        sentence = "This contains a clip sampled in %d to %d seconds" % (
            int(np.round(start)),
            int(np.round(end)),
        ) + DEFAULT_IMAGE_TOKEN
        seq.append(tokenizer_image_token(sentence, tokenizer, return_tensors="pt"))
    return seq


def preprocess_question(questions, tokenizer):
    return [tokenizer_image_token(q + DEFAULT_TODO_TOKEN, tokenizer, return_tensors="pt") for q in questions]


def process_data(video_id, question, model_config, tokenizer, processor, processor_large, time_tokenizer):
    num_frames = int(os.getenv("DISPIDER_NUM_FRAMES", os.getenv("DISPIDER_NUMFRAMES", "16")))
    max_clips = int(os.getenv("DISPIDER_MAX_CLIPS", "10000"))

    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates["qwen"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    frames, time_idx, num_clips = load_video(video_id, [], num_frames, max_clips)
    video = processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    video = video.view(num_clips, num_frames, *video.shape[1:])
    video_large = processor_large.preprocess(frames, return_tensors="pt")["pixel_values"]
    video_large = video_large.view(num_clips, num_frames, *video_large.shape[1:])[:, :1].contiguous()

    seqs = preprocess_time(time_idx, num_clips, time_tokenizer)
    seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=time_tokenizer.pad_token_id)
    compress_mask = seqs.ne(time_tokenizer.pad_token_id)

    question_ids = preprocess_question([question], time_tokenizer)
    question_ids = torch.nn.utils.rnn.pad_sequence(
        question_ids,
        batch_first=True,
        padding_value=time_tokenizer.pad_token_id,
    )
    qs_mask = question_ids.ne(time_tokenizer.pad_token_id)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    return input_ids, video, video_large, seqs, compress_mask, question_ids, qs_mask


class EvalDispider:
    def __init__(self, model_path):
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, image_processor, self.context_len = load_pretrained_model(model_path, None, model_name)
        self.image_processor, self.time_tokenizer = image_processor
        self.image_processor_large = self.image_processor
        if self.time_tokenizer.pad_token is None:
            self.time_tokenizer.pad_token = "<pad>"
        stop_words_ids = [torch.tensor(self.tokenizer("<|im_end|>").input_ids).cuda()]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def inference(self, video_file_name, prompt):
        input_ids, image_tensor, image_tensor_large, seqs, compress_mask, qs, qs_mask = process_data(
            video_file_name,
            prompt,
            self.model.config,
            self.tokenizer,
            self.image_processor,
            self.image_processor_large,
            self.time_tokenizer,
        )
        input_ids = input_ids.unsqueeze(0).to(device="cuda", non_blocking=True)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                images_large=image_tensor_large.to(dtype=torch.float16, device="cuda", non_blocking=True),
                seqs=seqs.to(device="cuda", non_blocking=True),
                compress_mask=compress_mask.to(device="cuda", non_blocking=True),
                qs=qs.to(device="cuda", non_blocking=True),
                qs_mask=qs_mask.to(device="cuda", non_blocking=True),
                ans_token=self.time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors="pt").input_ids.to(
                    device="cuda", non_blocking=True
                ),
                todo_token=self.time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors="pt").input_ids.to(
                    device="cuda", non_blocking=True
                ),
                q_id=None,
                insert_position=0,
                ans_position=[],
                do_sample=False,
                max_new_tokens=1024,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self.stopping_criteria,
                use_cache=True,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def build_prompt(task, question, options, anno, index):
    if task in BACKWARD_TASKS or task in REALTIME_TASKS:
        formatted_options = "; ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options)) + ";"
        return BR_PROMPT_TEMPLATE.format(question, formatted_options)
    if task == "REC":
        return REC_PROMPT_TEMPLATE.format("How many times did they " + anno["activity"] + "?")
    if task == "SSR":
        return SSR_PROMPT_TEMPLATE.format(anno["test_info"][index]["step"])
    if task == "CRR":
        return CRR_PROMPT_TEMPLATE.format(anno["question"])
    raise ValueError(f"Unsupported OVO-Bench task: {task}")


def load_annotations(path, tasks, max_samples, num_chunks, chunk_idx):
    with open(path, "r") as f:
        annotations = json.load(f)
    annotations = [item for item in annotations if item["task"] in tasks]
    if max_samples is not None:
        annotations = annotations[:max_samples]
    return annotations[chunk_idx::num_chunks]


def evaluate(model, annotations, chunked_dir):
    results = {"backward": [], "realtime": [], "forward": []}
    for item in tqdm(annotations, desc="OVO-Bench"):
        task = item["task"]
        if task in BACKWARD_TASKS or task in REALTIME_TASKS:
            prompt = build_prompt(task, item["question"], item["options"], None, None)
            video_path = Path(chunked_dir) / f"{item['id']}.mp4"
            result = {
                "id": item["id"],
                "video": item["video"],
                "task": task,
                "question": item["question"],
                "response": None,
                "ground_truth": chr(65 + item["gt"]),
            }
            try:
                result["response"] = model.inference(str(video_path), prompt)
            except Exception as exc:
                result["error"] = str(exc)
            key = "backward" if task in BACKWARD_TASKS else "realtime"
            results[key].append(result)
        else:
            output_item = copy.deepcopy(item)
            for idx, test_item in enumerate(output_item["test_info"]):
                prompt = build_prompt(task, None, None, output_item, idx)
                video_path = Path(chunked_dir) / f"{item['id']}_{idx}.mp4"
                try:
                    test_item["response"] = model.inference(str(video_path), prompt)
                except Exception as exc:
                    test_item["response"] = None
                    test_item["error"] = str(exc)
            results["forward"].append(output_item)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run Dispider on OVO-Bench.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--anno-path", required=True)
    parser.add_argument("--chunked-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--clip-ckpt-path", default=None)
    parser.add_argument("--model-name", default="Dispider")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.clip_ckpt_path:
        os.environ["DISPIDER_CLIP_CKPT_PATH"] = args.clip_ckpt_path
    if args.chunk_idx < 0 or args.chunk_idx >= args.num_chunks:
        raise ValueError("--chunk-idx must be in [0, --num-chunks)")

    annotations = load_annotations(args.anno_path, set(args.tasks), args.max_samples, args.num_chunks, args.chunk_idx)
    model = EvalDispider(args.model_path)
    results = evaluate(model, annotations, args.chunked_dir)

    output_dir = Path(args.result_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model_name}_{'_'.join(args.tasks)}_offline_chunk{args.chunk_idx}-of-{args.num_chunks}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved OVO-Bench results to {output_path}")


if __name__ == "__main__":
    main()
