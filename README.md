<div align="center">

<h1> 🌊  STC: Accelerating Streaming Video Large Language Models <br> via Hierarchical Token Compression 🚀 </h1>

<p align="center">
  <a href="https://arxiv.org/abs/your-paper-id"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/your-org"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue" alt="HuggingFace"></a>
  <a href="https://github.com/your-username/STC/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
</p>

<h4 align="center">
  [Author Name]<sup>1</sup>, [Author Name]<sup>1</sup>
  <br>
  <sup>1</sup> [Affiliation]
</h4>

</div>

---

## 🔥 News

* **`2025.11.30`** 🤗 We release **STC**, a plug-and-play streaming acceleration framework featuring **Streaming Token Compression**!

## 📌 Highlights

STC is designed to tackle the unique challenges of streaming video understanding:

* **⚡ Streaming-First Design:** Optimized for latency-sensitive applications (e.g., live sports, AR glasses) where frames arrive continuously.
* **🧩 STC-Cacher (Core):** Exploits temporal redundancy by caching visual features for similar frames (Cosine Similarity $> 0.85$), significantly reducing ViT encoding overhead.
* **✂️ STC-Pruner:** Compresses visual tokens *after* encoding to shorten the LLM prefill sequence while preserving spatiotemporal saliency.
* **🔌 Plug-and-Play:** Seamlessly integrates with SOTA VideoLLMs like **ReKV**, **Dispider**, **StreamForest**, and **Livecc**.
* **🚀 Proven Efficiency:**
    * **99%** Accuracy retention on ReKV.
    * **24.5%** Reduction in ViT encoding latency.
    * **45.3%** Reduction in LLM pre-filling latency.


## ✨ Overview

<p align="center"> <img src="images/overview.jpg" width="1000" align="center"> </p>

> **TL;DR:** STC introduces a **Hierarchical Token Compression** framework. It uses **STC-Cacher** to skip redundant ViT computations and **STC-Pruner** to reduce memory footprint for the LLM, operating in a strict **causal** manner.

### Why STC?

1.  **ViT is the Bottleneck:** In streaming, ViT encoding consumes 2-3x more time than image understanding.
2.  **Temporal Redundancy:** Adjacent frames in streams are highly similar. Re-computing them is wasteful.
3.  **Causal Processing:** Unlike offline methods, STC adapts to incrementally arriving frames without needing global context.

## 🦁 Model Zoo & Core Codes

We support the following models enhanced with STC. Checkpoints coming soon.

| Model Base | Status | Code Path |
| :--- | :--- | :--- |
| **ReKV (LLaVA-OneVision)** | ✅ Supported | [`model/llava_onevision_rekv.py`](model/llava_onevision_rekv.py) |
| **StreamForest** | 🚧 Coming Soon | - |
| **Dispider** | 🚧 Coming Soon | - |
| **Livecc** | 🚧 Coming Soon | - |

**Core Implementation:**
* **Cache Logic:** [`model/cache.py`](model/cache.py) (Class: `STC_CACHE`)
* **Prune Logic:** [`model/prune.py`](model/prune.py) (Class: `STC_Pruner`)

## 🛠 Preparation

```bash
# Clone the repository
git clone [https://github.com/your-username/STC.git](https://github.com/your-username/STC.git)
cd STC

# Setup Environment
conda create -n stc python=3.11 -y
conda activate stc

# Install Dependencies

pip install -e .
pip install requirements.txt

# Install LongVA dependencies
cd model/longva && pip install -e . && cd ../..
```
## 🚀 Performance Evaluation

We evaluate STC on both **Online (Streaming)** benchmarks to demonstrate real-time capabilities and **Offline** benchmarks to ensure robust general video understanding.
- Download pretrained Video-LLMs under `model_zoo/`
  - [llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)
> **Note:** Full results and comparisons can be found in our [Paper](https://www.google.com/search?q=link-to-paper).

| Category | Dataset | 
| :--- | :--- | 
| **🌊 Online** | **StreamingBench** | 
| | **OVO-Bench** | 
| **💾 Offline** | **MLVU** | 
| | **VideoMME** | 
| | **EgoSchema** | 
-----

### 🌊 Online Benchmarks (Streaming)

These benchmarks evaluate the model's ability to understand videos in a streaming fashion, where frames are received sequentially.

#### 1\. StreamingBench

Download the dataset from [mjuicem/StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench).

  * **Required files:** `Real_Time_Visual_Understanding.csv` and `Real-Time Visual Understanding_*.zip`.
  * **Configuration:** Update `eval/scripts/eval_streamingbench.sh`:
      * Set `TASK_CSV` to the path of the CSV file.
      * Set `VIDEO_DIR` to the unzipped video directory.

<!-- end list -->

```bash
bash eval/scripts/eval_streamingbench.sh
```

#### 2\. OVO-Bench

  * **Videos:** Download `src_videos.tar.parta[a-e]` from [JoeLeelyf/OVO-Bench (HF)](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench).
  * **Metadata:** Download `ovo_bench_new.json` from [JoeLeelyf/OVO-Bench (Github)](https://github.com/JoeLeelyf/OVO-Bench).
  * **Configuration:** Update `eval/scripts/eval_ovobench.sh`:
      * Set `TASK_JSON` to the path of `ovo_bench_new.json`.
      * Set `VIDEO_DIR` to the unzipped video directory.

<!-- end list -->

```bash
bash eval/scripts/eval_ovobench.sh
```

-----

### 💾 Offline Benchmarks (Standard)
**Supported Datasets:** `MLVU`, `EgoSchema`, `Videomme`

We use standard benchmarks to verify that STC maintains high performance on general video understanding tasks.

- Download benchmarks under `data/`
  - [MLVU-dev](https://huggingface.co/datasets/MLVU/MVLU)
  - [EgoSchema](https://huggingface.co/datasets/lmms-lab/egoschema)
  - [Videomme](https://huggingface.co/datasets/lmms-lab/Video-MME)


Run the evaluation using the unified script:

```bash
# Example: Evaluating on MLVU
python video_qa/run_eval.py \
    --model llava_ov_7b \
    --dataset mlvu \
    --num_chunks 8 \
    --sample_fps 0.5 \
    --retrieve_size 64
```

To evaluate `egoschema` or `videomme`, simply change the `--dataset` argument to the respective dataset name.

## 👍 Acknowledgment

We thank the open-source efforts of [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT) and [ReKV](https://github.com/Becomebright/ReKV).

## ✏️ Citation

If you find STC useful, please cite our paper:

```bibtex
@article{stc2025,
  title={STC: Accelerating Streaming Video Large Language Models via Hierarchical Token Compression},
  author={Name, Author and Name, Author},
  journal={arXiv preprint arXiv:2511.xxxxx},
  year={2025}
}
```

## 📩 Contact

For any questions, please open an issue or contact [email@address.com].