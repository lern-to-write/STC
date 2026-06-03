<div align="center">

<img src="assert/logo.png" alt="STC logo" width="760">

<h1> 🌊 Accelerating Streaming Video Large Language Models via Hierarchical Token Compression 🚀 </h1>

<h4 align="center">
  Yiyu Wang<sup>1*</sup>, Xuyang Liu<sup>1,2*†</sup>, Xiyan Gui<sup>1,3</sup>, Xinying Lin<sup>4</sup>, Boxue Yang<sup>1</sup>,
  <br>
  Chenfei Liao<sup>1,5</sup>, Tailai Chen<sup>1</sup>, Linfeng Zhang<sup>1✉</sup>
  <br><br>
  <sup>1</sup> EPIC Lab, SJTU &emsp; <sup>2</sup> Sichuan University &emsp;<sup>3</sup> HUST &emsp; <sup>4</sup> SYSU &emsp; <sup>5</sup> HKUST (Guangzhou)
</h4>

<p align="center"><i> ⚡ The <strong>first</strong> plug-and-play token compression framework for streaming video understanding. </i></p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.00891"><img src="https://img.shields.io/badge/arXiv-2512.00891-AD1C18?logo=arXiv&logoColor=white" alt="arXiv"></a>
  <a href="https://arxiv.org/abs/2512.00891"><img src="https://img.shields.io/badge/CVPR-2026-pink" alt="CVPR 2026"></a>
  <a href="https://mp.weixin.qq.com/s/PsNkR28yIFXqAQmAb62Yrg"><img src="https://img.shields.io/badge/PR-@PaperWeekly-blue" alt="PaperWeekly PR"></a>
  <a href="https://github.com/lern-to-write/STC/stargazers"><img src="https://img.shields.io/github/stars/lern-to-write/STC?style=social" alt="GitHub stars"></a>
</p>

</div>

## 🔥 News

* **`2026.06.03`** We refactored the codebase into a standalone `stc` Python package. STC-Cacher, STC-Pruner, HuggingFace ViT integrations, and ReKV eval drivers now live in a cleaner layout with updated quick-start and reproduction docs.
* **`2026.02.21`** 🎊🎊 Our [STC](https://arxiv.org/pdf/2512.00891) has been accepted by **CVPR 2026**! The codebase is under comprehensive cleanup. Stay tuned!
* **`2025.12.02`** 🤗🤗 We release our latest work [STC](https://arxiv.org/pdf/2512.00891), **the first** plug-and-play inference acceleration framework for streaming video understanding! [Code](https://github.com/lern-to-write/STC) is available!
* **`2025.08.21`** 🎉🎉 Our [VidCom<sup>2</sup>](https://arxiv.org/abs/2505.14454) has been accepted by **EMNLP 2025** main conference!
* **`2025.05.21`** 🤗🤗 We release [VidCom<sup>2</sup>](https://arxiv.org/abs/2505.14454), a plug-and-play inference acceleration method of **VideoLLMs**. [Code](https://github.com/xuyang-liu16/VidCom2) is available!

## 📌 Highlights

STC is a streaming-first token compression framework for plug-and-play acceleration of video large language models:

* **⚡ Streaming-First Design:** Optimized for latency-sensitive settings such as live sports, AR glasses, and long-running video streams where frames arrive continuously.
* **🧩 STC-Cacher:** Exploits temporal redundancy by selectively recomputing visual encoder tokens instead of fully re-encoding every frame.
* **✂️ STC-Pruner:** Compresses visual tokens after encoding to shorten the LLM prefill sequence while preserving spatiotemporal saliency.
* **🔌 Plug-and-Play Integration:** The core package is model-agnostic, with current ReKV integration and vendored research code for StreamForest, Dispider, and LiveCC.

## 🧱 Code Architecture

The current codebase is organized around a standalone `stc` Python package. ReKV consumes this package through its model wrappers and evaluation drivers.

```
STC_new/
├── stc/                Python package
│   ├── config.py           Dataclasses, env-driven defaults, legacy singleton
│   ├── core/               Shared algorithms
│   │   ├── selectors.py        Token similarity / dynamic-token selection
│   │   └── layer_ratio.py      Per-layer skip-ratio allocator
│   ├── cacher/             STC-Cacher
│   │   ├── state.py            STCCache per-stream state
│   │   └── reference_forward.py  Selective ViT-layer forwards
│   ├── pruner/             STC-Pruner
│   │   ├── pruner.py           STCPruner orchestrator
│   │   ├── scoring.py          Gaussian / dual-anchor score functions
│   │   ├── anchors.py          Temporal/spatial anchors
│   │   ├── index_mapper.py     Per-frame -> flat token index mapping
│   │   └── specs.py            MODEL_SPECS for supported layouts
│   ├── integrations/
│   │   └── hf_vit.py           register_stc_cacher() for HF CLIP / SigLIP
│   └── utils/distributed.py
├── models/             Vendored research code + model entrypoints
│   ├── rekv/               ReKV streaming inference consumer of stc
│   ├── livecc/             LiveCC vendored code
│   ├── Dispider/           Dispider vendored code
│   ├── StreamForest/       StreamForest vendored code
│   └── reproduce_smoke.py
├── benchmarks/         Offline / OVO-Bench / StreamingBench data
├── scripts/eval_rekv/  ReKV evaluation shell drivers
└── results/            Run outputs, ignored by git
```

### Core APIs

* **Configuration:** [`stc/config.py`](stc/config.py) defines `CacheConfig`, `ModelConfig`, and `STCConfig`. ReKV loads STC runtime settings from environment variables through `GlobalConfig.initialize_from_env()`.
* **Cacher:** [`stc/cacher/`](stc/cacher/) provides `STCCache` and selective vision-transformer forward functions.
* **Pruner:** [`stc/pruner/`](stc/pruner/) provides `STCPruner`, token scoring, temporal/spatial anchors, and model-specific index mapping.
* **HF ViT integration:** [`stc/integrations/hf_vit.py`](stc/integrations/hf_vit.py) monkey-patches HuggingFace CLIP/SigLIP-style vision encoder layers.
* **ReKV integration:** [`models/rekv/model/llava_onevision_rekv.py`](models/rekv/model/llava_onevision_rekv.py) wires STC into the LLaVA-OneVision ReKV path.

STC-Cacher and STC-Pruner are applied differently:

* **STC-Cacher** is a monkey patch. `register_stc_cacher()` replaces `vision_tower.vision_model.encoder.layers[*].forward` with selective-recompute forwards.
* **STC-Pruner** is an explicit call. ReKV creates `self.stc_pruner = STCPruner()` and calls `self.stc_pruner.compress(...)` after vision encoding / projection / pooling and before LLM prefill.

## 📚 Reproduction Guides

We added model-specific reproduction notes so a fresh GPU machine can go from
environment setup to a smoke run without relying on our internal container or
absolute paths. Start with the quick reproduction guide for the model you want
to run, then use the matching changes document to inspect exactly what was
modified from the upstream repository and why.

| Model | Quick reproduce | What changed from upstream |
| :--- | :--- | :--- |
| **ReKV** | [`docs/rekv/REPRODUCE.md`](docs/rekv/REPRODUCE.md) | [`docs/rekv/CHANGES.md`](docs/rekv/CHANGES.md) |
| **StreamForest** | [`docs/streamforest/REPRODUCE.md`](docs/streamforest/REPRODUCE.md) | [`docs/streamforest/CHANGES.md`](docs/streamforest/CHANGES.md) |
| **Dispider** | [`docs/dispider/reproduce.md`](docs/dispider/reproduce.md) | [`docs/dispider/changes.md`](docs/dispider/changes.md) |
| **LiveCC** | Code vendored under [`models/livecc/`](models/livecc/); reproduction guide TBD | — |

The vendored model code keeps the original research implementations, but we
slightly adjusted path handling, model discovery, launch scripts, and benchmark
entrypoints where needed to make one-command reproduction practical. Those
changes are documented with the corresponding upstream `git diff` in each
`CHANGES.md`.

## 🛠 Installation

```bash
pip install -e .            # core package, requires torch
pip install -e .[hf]        # adds transformers for HF CLIP / SigLIP integrations
```

`requirements.txt` is a reference environment snapshot. It is not the minimal package dependency declaration; prefer `pyproject.toml` for the maintained install surface.

## 🚀 Quick Start

For end-to-end benchmark reproduction, follow the model-specific guides above.
This section shows the minimal package API and the environment variables used
by the current ReKV integration.

### Use the standalone package

```python
import stc

cache = stc.STCCache()
config = stc.STCConfig()

# vision_tower is the HF vision encoder of an already-loaded model,
# e.g. model.vision_tower (CLIP / SigLIP style).
stc.register_stc_cacher(
    vision_tower,
    kind="siglip",          # or "clip"
    cache=cache,
    config=config.cache,
)

pruner = stc.STCPruner(config.model)
compressed = pruner.compress(features, model="llava_ov")

# Advance per-chunk state during streaming inference.
cache.reset_for_chunk(chunk_idx=1, update_token_ratio=0.25)
```

### Control ReKV integration with environment variables

ReKV reads STC settings from environment variables. STC-specific CLI arguments have been removed from the ReKV drivers.

```bash
export STC_PATCH_VISION=1          # 1 enables STC-Cacher monkey patch; 0 disables it
export STC_TOKEN_PER_FRAME=64      # visual token budget per frame
export STC_UPDATE_TOKEN_RATIO=0.25 # selective recompute ratio
export STC_CACHE_INTERVAL=2        # reference refresh interval
```

Fixed defaults in the current ReKV integration:

* Cacher strategy is `selective` when `STC_PATCH_VISION=1`, otherwise `none`.
* Selector metric is `cosine`.
* Pruner strategy is `gaussian`.
* `encode_chunk_size=1`, `channel_keep_ratio=0.5`, and `spatial_temporal_alpha=0.5`.

Use `STC_TOKEN_PER_FRAME=196` for LLaVA-OneVision full-token retention.

## 📊 Performance Evaluation

We evaluate STC on **4 streaming VideoLLM baselines** (Dispider, LiveCC, StreamForest, ReKV) across **5 benchmarks** (OVO-Bench, StreamingBench, EgoSchema, MLVU-dev, VideoMME), under a 0.5 fps streaming protocol.

> **TL;DR** — On the ReKV framework, STC retains up to **99%** of accuracy while cutting **ViT encoding latency by 24.5%** and **LLM pre-filling latency by 45.3%**. It surpasses VidCom² by **1.6** on both OVO-Bench and StreamingBench, and ToMe by **5.6 / 5.8** respectively. Latencies below are in seconds (ViT: encode 16 frames; LLM: pre-fill time).

### Streaming Benchmarks (ReKV, LLaVA-OV-7B backbone)

| Method | OVO Real-Time | OVO Backward | OVO Forward | StreamingBench | ViT Enc. Lat. | LLM Pref. Lat. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ReKV | 64.4 | 64.6 | 52.6 | 69.1 | 103.7 | 482.4 |
| + ToMe | 53.1 | 60.7 | 46.4 | 59.4 | 70.5 (↓32.0%) | 257.8 (↓46.6%) |
| + VisionZip | 53.8 | 58.4 | 47.5 | 60.4 | 103.7 | 258.3 (↓46.5%) |
| + VidCom² | 60.4 | 59.0 | 50.4 | 63.6 | 103.7 | 259.1 (↓46.3%) |
| **+ STC (Cacher & Pruner)** | **62.5** | **63.3** | **52.0** | **65.2** | **78.3 (↓24.5%)** | **263.7 (↓45.3%)** |

### STC-Cacher Generality (OVO-Bench, across 3 baselines)

Reported as `baseline → + STC-Cacher`. STC-Cacher targets ViT encoding latency and plugs into diverse backbones.

| Model | Real-Time | Backward | Forward | ViT Enc. Lat. |
| :--- | :---: | :---: | :---: | :---: |
| Dispider | 51.0 → 49.1 | 40.1 → 36.6 | 40.4 → 39.2 | 26.4 → 18.9 (↓28.4%) |
| LiveCC | 57.0 → 53.8 | 56.4 → 54.2 | 59.7 → 57.3 | 181.2 → 126.8 (↓30.0%) |
| StreamForest | 61.6 → 59.1 | 70.8 → 68.2 | 54.3 → 52.3 | 103.7 → 67.7 (↓34.7%) |

### Offline Long-Video Understanding (ReKV)

| Method | EgoSchema | MLVU-dev | VideoMME | Avg |
| :--- | :---: | :---: | :---: | :---: |
| ReKV | 57.7 | 68.6 | 57.7 | 61.3 |
| + ToMe | 55.2 | 63.1 | 51.7 | 56.7 |
| + VisionZip | 55.8 | 63.2 | 51.6 | 56.9 |
| + VidCom² | 60.6 | 67.1 | 56.8 | 61.5 |
| **+ STC-Pruner** | **60.8** | **67.6** | **57.1** | **61.8** |

See the [paper](https://arxiv.org/abs/2512.00891) for the full results, including per-subset VideoMME breakdowns and ablations.

### Smoke Test

```bash
# Baseline ReKV
bash scripts/eval_rekv/eval_rekv_smoke.sh rekv

# ReKV + STC
bash scripts/eval_rekv/eval_rekv_smoke.sh rekv_stc
```

### Offline Benchmarks

Supported datasets include `mlvu`, `egoschema`, and `videomme` variants registered in [`models/rekv/model/video_qa/configs.py`](models/rekv/model/video_qa/configs.py).

```bash
bash scripts/eval_rekv/eval_offline_benchs.sh \
  --dataset mlvu \
  --model llava_ov_7b \
  --save_dir results/mlvu_rekv_stc
```

### OVO-Bench

Download videos and metadata from the OVO-Bench release, then set paths through environment variables if needed:

```bash
export ANNO_PATH=/path/to/ovo_bench_new.json
export VIDEO_DIR=/path/to/src_videos
export CHUNKED_DIR=/path/to/chunked_videos
export RESULT_DIR=results/ovobench_rekv_stc

bash scripts/eval_rekv/ovobench_scripts/eval_rekv.sh
bash scripts/eval_rekv/ovobench_scripts/score_rekv.sh
```

### StreamingBench

Download StreamingBench from [mjuicem/StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench). The ReKV driver lives under:

```bash
bash scripts/eval_rekv/streamingbench_scripts/eval_rekv.sh
bash scripts/eval_rekv/streamingbench_scripts/score_rekv.sh
```

### Dataset References

* **StreamingBench:** [mjuicem/StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench)
* **OVO-Bench:** [JoeLeelyf/OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench)
* **MLVU:** [MLVU/MVLU](https://huggingface.co/datasets/MLVU/MVLU)
* **EgoSchema:** [lmms-lab/egoschema](https://huggingface.co/datasets/lmms-lab/egoschema)
* **VideoMME:** [lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME)

## 👍 Acknowledgment

* Thanks to [ReKV](https://github.com/Becomebright/ReKV) for their great work and codebase.
* Thanks to [StreamForest](https://github.com/MCG-NJU/StreamForest) for their great work and codebase.
* Thanks to [Dispider](https://github.com/Mark12Ding/Dispider) for their great work and codebase.
* Thanks to [LiveCC](https://github.com/showlab/livecc) for their great work and codebase.

## ✏️ Citation

Please consider citing our paper in your publications if our findings help your research.

```bibtex
@inproceedings{wang2026stc,
  title={Accelerating Streaming Video Large Language Models via Hierarchical Token Compression},
  author={Wang, Yiyu and Liu, Xuyang and Gui, Xiyan and Lin, Xinying and Yang, Boxue and Liao, Chenfei and Chen, Tailai and Zhang, Linfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## 📩 Contact

For any question about our paper or code, please email `liuxuyang@stu.scu.edu.cn` or `ustywan8@ljmu.ac.uk`.
