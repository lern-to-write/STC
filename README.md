# ReKV

Official PyTorch code of "Streaming Video Question-Answering with In-context Video KV-Cache Retrieval", *ICLR* 2025.

## Abstract

We propose **ReKV**, a novel, training-free approach that integrates seamlessly with existing Video Large Language Models (Video-LLMs) to enable efficient streaming video question-answering (**StreamingVQA**).

Traditional VideoQA systems struggle with long videos, as they must process the entire video before responding to queries, and repeat this process for each new question. In contrast, our approach analyzes long videos in a streaming fashion, allowing for prompt responses as soon as user queries are received. 
- Building on a common Video-LLM, we first incorporate a sliding-window attention mechanism, ensuring that input frames attend to a limited number of preceding frames, thereby reducing computational overhead.
- To prevent information loss, we store processed video key-value caches (KV-Caches) in RAM and disk, reloading them into GPU memory as needed. 
- Additionally, we introduce a retrieval method that leverages an external retriever or the parameters within Video-LLMs to retrieve only query-relevant KV-Caches, ensuring both efficiency and accuracy in question answering.

ReKV enables the separation of video analyzing and question-answering across different processes and GPUs, significantly enhancing the efficiency of StreamingVQA. 
Through comprehensive experimentation, we validate the efficacy and practicality of our approach, which significantly boosts efficiency and enhances applicability over existing VideoQA models.

## Directory Structure

```
.
├── data        processed benchmarks
├── model       code for integrating ReKV with various Video-LLMs
├── model_zoo   pretrained Video-LLM checkpoints
├── results     evaluation results
└── video_qa    code for StreamingVQA & OfflineVQA
```

## Preparation

Our setup: Ubuntu 22.04, CUDA 12.6, 8x Nvidia H800 (80GB)

- Clone this repo: `git clone https://github.com/Becomebright/ReKV.git`
- Prepare the conda environment: `bash prepare.sh`
- Download pretrained Video-LLMs under `model_zoo/`
  - [llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
  - [llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)
  - [llava-onevision-qwen2-72b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf)
  - [LongVA-7B](https://huggingface.co/lmms-lab/LongVA-7B)
  - [Video-LLaVA-7B-hf](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)
- Download benchmarks under `data/`
  - [MLVU-dev-mc](https://huggingface.co/datasets/MLVU/MVLU)
  - [QAEgo4D-test-mc](https://huggingface.co/datasets/Becomebright/QAEgo4D-MC-test/tree/main)
  - [EgoSchema-full](https://huggingface.co/datasets/lmms-lab/egoschema)
  - [ActivityNet-QA](https://huggingface.co/datasets/lmms-lab/ActivityNetQA)
  - [RVS](https://huggingface.co/datasets/Becomebright/RVS)
  - [CGBench](https://huggingface.co/datasets/CG-Bench/CG-Bench)
  - The `data/` folder should be arranged as:
    ```
    data
    ├── activitynet_qa
    │   ├── test.json
    │   └── videos
    ├── cgbench
    │   ├── full_mc.json
    │   └── videos
    ├── egoschema
    │   ├── full.json
    │   └── videos
    ├── mlvu
    │   ├── dev_debug_mc.json
    │   └── videos
    ├── qaego4d
    │   ├── test_mc.json
    │   └── videos
    └── rvs
        ├── ego
        │   ├── ego4d_oe.json
        │   └── videos
        └── movie
            ├── movienet_oe.json
            └── videos
    ```
- Increases the memory map limit for processes (needed for offloading KV-Caches): `sudo sysctl -w vm.max_map_count=262144`

## Evaluation

```bash
# The number of processes utilized for parallel evaluation.
# Normally, set it to the number of GPUs on your machine.
# Yet, llava_ov_72b needs 4x 80GB GPUs. So set num_chunks to num_gpus//4.
num_chunks=8

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
model=llava_ov_0.5b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie
# MLVU has an extremely long video (~9hr). Remove it in the annotation file if your system doesn't have enough RAM.
dataset=qaego4d

python -m video_qa.run_eval \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 0.5 \
    --n_local 15000 \
    --retrieve_size 64
```

## Citation

```latex
@inproceedings{di2025rekv,
  title={Streaming Video Question-Answering with In-context Video KV-Cache Retrieval},
  author={Di, Shangzhe and Yu, Zhelun and Zhang, Guanghao and Li, Haoyuan and Cheng, Hao and Li, Bolin and He, Wanggui and Shu, Fangxun and Jiang, Hao and others},
  booktitle={ICLR},
  year={2025}
}
```

## Acknowledgements

Our code is based on [InfLLM](https://github.com/thunlp/InfLLM), [StreamingLLM](https://github.com/mit-han-lab/streaming-llm), and [Flash-VStream](https://github.com/IVGSZ/Flash-VStream).