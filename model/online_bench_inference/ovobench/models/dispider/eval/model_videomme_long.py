import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
# TODO
# sys.path.append($$YOUR_MODEL_PATH)
from dispider.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_ANS_TOKEN, DEFAULT_TODO_TOKEN
from dispider.conversation import conv_templates, SeparatorStyle
from dispider.model.builder import load_pretrained_model
from dispider.utils import disable_torch_init
from dispider.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import pickle
from decord import VideoReader
import numpy as np

from transformers import StoppingCriteria, StoppingCriteriaList
from petrel_client.client import Client
client = Client('~/petreloss.conf')

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


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
    key_frame = [[frame_idx[i*frm_per_clip], frame_idx[i*frm_per_clip+frm_per_clip-1]] for i in range(num_clip)]
    time = vr.get_frame_timestamp(key_frame)
    return np.hstack([time[:, 0, 0], time[:, 1, 1]])


def calculate_diff(scene_sep, start_frame):
    diff = [scene_sep[0]-start_frame]
    for i in range(len(scene_sep)-1):
        diff.append(scene_sep[i+1]-scene_sep[i])
    return diff


def load_video(vis_path, scene_sep, num_frm=16, max_clip=4, sample_frame=None):
    block_size = 1
    vr = VideoReader(vis_path)
    total_frame_num = len(vr) if sample_frame is None else (sample_frame[0][1]-sample_frame[0][0])
    fps = vr.get_avg_fps()
    total_time = total_frame_num / fps

    if len(scene_sep) == 0:
        num_clip = total_time / num_frm
        num_clip = int(block_size*np.round(num_clip/block_size)) if num_clip > block_size else int(np.round(num_clip))
        num_clip = max(num_clip, 5)
        num_clip = min(num_clip, max_clip)
        total_num_frm = num_frm * num_clip
        start_frame = 0 if sample_frame is None else sample_frame[0][0]
        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    else:
        ref_clip = total_time / num_frm
        ref_clip = int(block_size*np.round(ref_clip/block_size)) if ref_clip > block_size else int(np.round(num_clip))
        ref_clip = max(ref_clip, 5)
        num_clip = max(len(scene_sep), ref_clip)
        num_clip = min(num_clip, max_clip)
        total_num_frm = num_frm * num_clip
        start_frame = 0 if sample_frame is None else sample_frame[0][0]
        frame_idx = []
        if len(scene_sep) < num_clip:
            diff = calculate_diff(scene_sep, start_frame)
            ratio = np.array(diff) / (total_frame_num / num_clip)
            ratio = np.maximum(np.round(ratio), 1)
            intervals = np.array(diff) / ratio
            num_clip = int(np.sum(ratio))
            total_num_frm = num_frm * num_clip
            new_sep = []
            start_ = start_frame
            for i in range(len(diff)):
                for k in range(int(ratio[i])):
                    new_sep.append(int(start_+intervals[i]*(k+1)))
                start_ = scene_sep[i]
            scene_sep = new_sep
            assert len(scene_sep) == num_clip
        elif len(scene_sep) > max_clip:
            diff = calculate_diff(scene_sep, start_frame)
            min_idx = np.argsort(diff[:-1])[:len(scene_sep)-max_clip] ##minimum diff to remove
            for i in np.sort(min_idx)[::-1]:
                del scene_sep[i]

        start_ = start_frame
        for end_frame in scene_sep:
            idx_list = np.linspace(start_, end_frame, num=num_frm, endpoint=False)
            frame_idx.extend([int(id) for id in idx_list])
            start_ = end_frame

    time_idx = get_seq_time(vr, frame_idx, num_clip)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    a, H, W, _ = img_array.shape
    if H != W:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(min(H, W), min(H, W)))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs, time_idx, num_clip


def preprocess_time(time, num_clip, tokenizer):
    time = time.reshape(2, num_clip)
    seq = []

    block_size = 1
    for i in range(num_clip):
        start, end = time[:, i]
        start = int(np.round(start))
        end = int(np.round(end))
        if (i+1) % block_size == 0:
            history_end = end
        sentence = 'This contains a clip sampled in %d to %d seconds' % (start, end) + DEFAULT_IMAGE_TOKEN
        sentence = tokenizer_image_token(sentence, tokenizer, return_tensors='pt')
        seq.append(sentence)
    return seq


def preprocess_question(questions, tokenizer):
    seq = []
    for q in questions:
        sentence = tokenizer_image_token(q+DEFAULT_TODO_TOKEN, tokenizer, return_tensors='pt')
        seq.append(sentence)
    
    return seq


def process_data(video_id, scene_sep, question, candidates, model_config, image_folder, tokenizer, processor, processor_large, time_tokenizer):
    num_frames = 16
    num_clips = 32
    system = 'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n'
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + system + question + '\n' + '\n'.join(candidates)

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], 'The best answer is:')
    prompt = conv.get_prompt()

    # image = video_id
    # presigned_url = client.generate_presigned_url(image, client_method ='get_object', expires_in=3600) if 's3://' in image else image
    frames, time_idx, num_clips = load_video(video_id, scene_sep, num_frames, num_clips)
    video = processor.preprocess(frames, return_tensors='pt')['pixel_values']
    video = video.view(num_clips, num_frames, *video.shape[1:])
    video_large = processor_large.preprocess(frames, return_tensors='pt')['pixel_values']
    video_large = video_large.view(num_clips, num_frames, *video_large.shape[1:])[:, :1].contiguous()
    seqs = preprocess_time(time_idx, num_clips, time_tokenizer)
    seqs = torch.nn.utils.rnn.pad_sequence(
        seqs, 
        batch_first=True,
        padding_value=time_tokenizer.pad_token_id)
    compress_mask = seqs.ne(time_tokenizer.pad_token_id)
    question = preprocess_question([question], time_tokenizer)
    question = torch.nn.utils.rnn.pad_sequence(
        question, 
        batch_first=True,
        padding_value=time_tokenizer.pad_token_id)
    qs_mask = question.ne(time_tokenizer.pad_token_id)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')[:-1]

    return input_ids, video, video_large, seqs, compress_mask, question, qs_mask


def eval_dataset(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    print(model)
    image_processor, time_tokenizer = image_processor
    image_processor_large = image_processor
    if time_tokenizer.pad_token is None:
        time_tokenizer.pad_token = '<pad>'

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    stop_words_ids = [
        torch.tensor(tokenizer('<|im_end|>').input_ids).cuda(),
    ]
    stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])

    qa_json = args.Eval_QA_root
    image_folder = args.image_folder

    with open(qa_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = get_chunk(data, args.num_chunks, args.chunk_idx)

    eval_dict = []
    for video_item in tqdm(data):
        video_path = video_item['video_path']
        questions = video_item['questions']
        scene_sep = video_item['scene_sep'] if 'scene_sep' in video_item else []
        try:
            for item in questions:
                question = item['question']
                candidates = item['options']
                #=================================You need to change this code =========================
                input_ids, image_tensor, image_tensor_large, seqs, compress_mask, qs, qs_mask = process_data(video_path, scene_sep, question, candidates, model.config, image_folder, tokenizer, image_processor, image_processor_large, time_tokenizer)
                input_ids = input_ids.unsqueeze(0).to(device='cuda', non_blocking=True)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                        images_large=image_tensor_large.to(dtype=torch.float16, device='cuda', non_blocking=True),
                        seqs=seqs.to(device='cuda', non_blocking=True),
                        compress_mask=compress_mask.to(device='cuda', non_blocking=True),
                        qs=qs.to(device='cuda', non_blocking=True),
                        qs_mask=qs_mask.to(device='cuda', non_blocking=True),
                        ans_token=time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors="pt").input_ids.to(device='cuda', non_blocking=True),
                        todo_token=time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors="pt").input_ids.to(device='cuda', non_blocking=True),
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria=stopping_criteria,
                        use_cache=True)

                input_token_len = input_ids.shape[1]
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                outputs = outputs.strip()
                item['response'] = outputs
                #=======================================================================================
                # print(f'q_id:{q_id}, output:{outputs}!\n')
        except:
            print('Fail processing video %s' % video_path)
            pass
        eval_dict.append(video_item)

    # eval results
    eval_dataset_json = args.chat_conversation_output_folder
    os.makedirs(os.path.dirname(eval_dataset_json), exist_ok=True)
    with open(eval_dataset_json, 'w', encoding='utf-8') as f:
        json.dump(eval_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--dataset_name", type=str, default=None, help="The type of LLM")
    parser.add_argument("--Eval_QA_root", type=str, default='./', help="folder containing QA JSON files")
    parser.add_argument("--Eval_Video_root", type=str, default='./', help="folder containing video data")
    parser.add_argument("--chat_conversation_output_folder", type=str, default='./Chat_results', help="")
    args = parser.parse_args()

    eval_dataset(args)
