<div align="center">

<img src="assert/logo.png" alt="STC logo" width="720">

<h1>🌊 STC: Accelerating Streaming Video LLMs<br>via Hierarchical Token Compression 🚀</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2512.00891"><img src="https://img.shields.io/badge/arXiv-2512.00891-B31B1B?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://arxiv.org/abs/2512.00891"><img src="https://img.shields.io/badge/CVPR-2026-1E90FF" alt="CVPR 2026"></a>
  <a href="https://mp.weixin.qq.com/s/PsNkR28yIFXqAQmAb62Yrg"><img src="https://img.shields.io/badge/PR-PaperWeekly-blue" alt="PaperWeekly"></a>
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-≥2.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <a href="https://github.com/lern-to-write/STC/stargazers"><img src="https://img.shields.io/github/stars/lern-to-write/STC?style=social" alt="GitHub stars"></a>
</p>

<h4 align="center">
  Yiyu Wang<sup>1*</sup>, Xuyang Liu<sup>1,2*†</sup>, Xiyan Gui<sup>1,3</sup>, Xinying Lin<sup>4</sup>, Boxue Yang<sup>1</sup>,
  <br>
  Chenfei Liao<sup>1,5</sup>, Tailai Chen<sup>1</sup>, Linfeng Zhang<sup>1✉</sup>
  <br><br>
  <sup>1</sup> EPIC Lab, SJTU &emsp; <sup>2</sup> Sichuan University &emsp; <sup>3</sup> HUST &emsp; <sup>4</sup> SYSU &emsp; <sup>5</sup> HKUST (Guangzhou)
</h4>

<p align="center"><i>⚡ The <strong>first</strong> plug-and-play token-compression framework for streaming video understanding.</i></p>

<p align="center">
  <b>↓24.5%</b> ViT encoding latency &nbsp;·&nbsp; <b>↓45.3%</b> LLM pre-filling latency &nbsp;·&nbsp; <b>up to 99%</b> accuracy retained
  <br>
  <sub>on the ReKV framework — see <a href="#-results">Results</a></sub>
</p>

</div>

<p align="center">
  <a href="#-highlights">Highlights</a> ·
  <a href="#-supported-frameworks">Supported Frameworks</a> ·
  <a href="#-results">Results</a> ·
  <a href="#-installation">Installation &amp; Reproduction</a> ·
  <a href="#-evaluation">Evaluation</a> ·
  <a href="#-latency-benchmark">Latency Benchmark</a> ·
  <a href="#-citation">Citation</a>
</p>

---

## 🔥 News

- **`2026.06.04`** &nbsp;🚀 Added a runtime **latency benchmark** (`speed_benchmark/`), add support for **StreamForest & Dispider**.
- **`2026.06.03`** &nbsp;🧱 Refactored the codebase into a standalone **`stc`** Python package — STC-Cacher, STC-Pruner, HF ViT integrations in a clean layout.
- **`2026.02.21`** &nbsp;🎊 STC is accepted by **CVPR 2026**!
- **`2025.12.02`** &nbsp;🤗 We release [STC](https://arxiv.org/pdf/2512.00891), **the first** plug-and-play inference-acceleration framework for streaming video understanding.
- **`2025.08.21`** &nbsp;🎉 Our [VidCom²](https://arxiv.org/abs/2505.14454) is accepted by **EMNLP 2025** (main).
- **`2025.05.21`** &nbsp;🤗 We release [VidCom²](https://arxiv.org/abs/2505.14454), a plug-and-play acceleration method for VideoLLMs.

---

## ✨ Highlights

<table>
<tr>
<td width="50%" valign="top">

#### ⚡ Streaming-First
Built for latency-sensitive, continuously-arriving frames — live sports, AR glasses, long-running streams.

#### 🧩 STC-Cacher
Exploits **temporal redundancy**: selectively recomputes only the *dynamic* visual tokens of each frame and reuses the rest, instead of fully re-encoding every frame.

</td>
<td width="50%" valign="top">

#### ✂️ STC-Pruner
Compresses visual tokens **after encoding** to shorten the LLM pre-fill sequence, while preserving spatiotemporal saliency.

#### 🔌 Plug-and-Play & Hardware-Aware
Model-agnostic core; drops into **ReKV, StreamForest, and Dispider** with one call.

</td>
</tr>
</table>

---

## 🧩 Supported Frameworks

STC's core package is framework-agnostic. STC-Cacher attaches to any HuggingFace pre-LN **CLIP / SigLIP** vision tower via a one-line monkey-patch; STC-Pruner is an explicit call before LLM pre-fill.

| Framework | Vision Tower | STC-Cacher | STC-Pruner | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **ReKV** | SigLIP (LLaVA-OneVision) | ✅ | ✅ | Reference integration |
| **StreamForest** | SigLIP | ✅ | — | Per-frame streaming cacher |
| **Dispider** | CLIP | ✅ | — | Per-frame streaming cacher |
| **LiveCC** | — | 🔜 | 🔜 | Vendored; integration WIP |

---

## 📊 Results

> **TL;DR** — On **ReKV**, STC retains up to **99%** of accuracy while cutting **ViT encoding latency by 24.5%** and **LLM pre-filling latency by 45.3%**, surpassing VidCom² by **1.6** on both OVO-Bench and StreamingBench (and ToMe by **5.6 / 5.8**). Evaluated on **4 streaming VideoLLM baselines** × **5 benchmarks** at a 0.5 fps streaming protocol. Latencies are in seconds (ViT: encode 16 frames; LLM: pre-fill time).

#### Streaming benchmarks — ReKV (LLaVA-OV-7B)

| Method | OVO Real-Time | OVO Backward | OVO Forward | StreamingBench | ViT Enc. Lat. | LLM Pref. Lat. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ReKV | 64.4 | 64.6 | 52.6 | 69.1 | 103.7 | 482.4 |
| &nbsp;&nbsp;+ ToMe | 53.1 | 60.7 | 46.4 | 59.4 | 70.5 <sub>↓32.0%</sub> | 257.8 <sub>↓46.6%</sub> |
| &nbsp;&nbsp;+ VisionZip | 53.8 | 58.4 | 47.5 | 60.4 | 103.7 | 258.3 <sub>↓46.5%</sub> |
| &nbsp;&nbsp;+ VidCom² | 60.4 | 59.0 | 50.4 | 63.6 | 103.7 | 259.1 <sub>↓46.3%</sub> |
| **+ STC** <sub>(Cacher & Pruner)</sub> | **62.5** | **63.3** | **52.0** | **65.2** | **78.3** <sub>↓24.5%</sub> | **263.7** <sub>↓45.3%</sub> |

#### STC-Cacher generality — OVO-Bench, `baseline → + STC-Cacher`

| Framework | Real-Time | Backward | Forward | ViT Enc. Lat. |
| :--- | :---: | :---: | :---: | :---: |
| Dispider | 51.0 → 49.1 | 40.1 → 36.6 | 40.4 → 39.2 | 26.4 → 18.9 <sub>↓28.4%</sub> |
| LiveCC | 57.0 → 53.8 | 56.4 → 54.2 | 59.7 → 57.3 | 181.2 → 126.8 <sub>↓30.0%</sub> |
| StreamForest | 61.6 → 59.1 | 70.8 → 68.2 | 54.3 → 52.3 | 103.7 → 67.7 <sub>↓34.7%</sub> |

#### Offline long-video understanding — ReKV

| Method | EgoSchema | MLVU-dev | VideoMME | Avg |
| :--- | :---: | :---: | :---: | :---: |
| ReKV | 57.7 | 68.6 | 57.7 | 61.3 |
| &nbsp;&nbsp;+ ToMe | 55.2 | 63.1 | 51.7 | 56.7 |
| &nbsp;&nbsp;+ VisionZip | 55.8 | 63.2 | 51.6 | 56.9 |
| &nbsp;&nbsp;+ VidCom² | 60.6 | 67.1 | 56.8 | 61.5 |
| **+ STC-Pruner** | **60.8** | **67.6** | **57.1** | **61.8** |

<sub>See the <a href="https://arxiv.org/abs/2512.00891">paper</a> for full results, per-subset VideoMME breakdowns, and ablations.</sub>

---

## 🛠 Installation

```bash
pip install -e .            # core package (requires torch)
pip install -e .[hf]        # + transformers, for HF CLIP / SigLIP integrations
```



#### Reproducing the baseline frameworks

To make the vendored frameworks easy to reproduce, we made **small, documented
modifications** to each upstream repo (path handling, model discovery, launch
scripts, benchmark entrypoints). For every framework we ship a doc pair — a
**REPRODUCE** guide (fresh GPU machine → environment → smoke run) and a
**CHANGES** doc (exactly what we adapted, with the upstream `git diff`). Just
follow the guide for the framework you want to run.

| Framework | Quick reproduce | What changed from upstream |
| :--- | :--- | :--- |
| **ReKV** | [`docs/rekv/REPRODUCE.md`](docs/rekv/REPRODUCE.md) | [`docs/rekv/CHANGES.md`](docs/rekv/CHANGES.md) |
| **StreamForest** | [`docs/streamforest/REPRODUCE.md`](docs/streamforest/REPRODUCE.md) | [`docs/streamforest/CHANGES.md`](docs/streamforest/CHANGES.md) |
| **Dispider** | [`docs/dispider/reproduce.md`](docs/dispider/reproduce.md) | [`docs/dispider/changes.md`](docs/dispider/changes.md) |
| **LiveCC** | Vendored under [`models/livecc/`](models/livecc/) · guide TBD | — |

---

## 🧪 Evaluation

Copy a block, replace the `/path/to/...` placeholders with your own, and run.
Outputs land under `results/`. Each block runs **+ STC**; for the baseline,
change the mode arg (`rekv_stc`→`rekv`, `sf_stc`→`sf`, `dispider_stc`→`dispider`).

#### ReKV (Baseline)

```bash

# Offline benchmark (dataset: mlvu / egoschema / videomme / ...)
export STC_PATCH_VISION=1 STC_TOKEN_PER_FRAME=64 STC_UPDATE_TOKEN_RATIO=0.25
bash scripts/eval_rekv/eval_offline_benchs.sh \
  --dataset mlvu --model llava_ov_7b --save_dir results/mlvu_stc

# OVO-Bench
export ANNO_PATH=/path/to/ovo_bench_new.json
export VIDEO_DIR=/path/to/src_videos
export CHUNKED_DIR=/path/to/chunked_videos
export STC_PATCH_VISION=1 STC_TOKEN_PER_FRAME=64 STC_UPDATE_TOKEN_RATIO=0.25
bash scripts/eval_rekv/ovobench_scripts/eval_rekv.sh
bash scripts/eval_rekv/ovobench_scripts/score_rekv.sh
```

#### StreamForest (Baseline)

```bash
export STREAMFOREST_CKPT_PATH=/path/to/StreamForest-Qwen2-7B
TASKS=ovobench bash scripts/eval_streamforest/eval_streamforest.sh sf_stc
```

#### Dispider (Baseline)

```bash
export MODEL_PATH=/path/to/Dispider
export CLIP_CKPT_PATH=/path/to/clip-vit-large-patch14
export ANNO_PATH=/path/to/ovo_bench_new.json
export CHUNKED_DIR=/path/to/chunked_videos
bash scripts/eval_dispider/eval_dispider_ovobench.sh dispider_stc
```

<details>
<summary><b>STC knobs </b></summary>

```bash
export STC_PATCH_VISION=1          # enable STC-Cacher (0 = baseline)
export STC_TOKEN_PER_FRAME=64      # STC-Pruner token budget per frame (196 = full; ReKV only)
export STC_UPDATE_TOKEN_RATIO=0.25 # STC-Cacher selective-recompute ratio
export STC_CACHE_INTERVAL=4        # full reference frame every N frames
```

- StreamForest & Dispider are **cacher-only** (no pruner). CUDA-graph replay and
  per-frame shared token selection are always on (not user-configurable) and fall
  back safely if unsupported.
- More options (GPUs, frames, tasks, sharding) are in each
  `scripts/eval_<framework>/README.md`.

</details>

---

## ⏱️ Latency Benchmark

Measure the latency reduction (baseline vs +STC) on **your** GPU —
runtime-instrumented, no code changes, 16-frame default. One script per framework
under [`speed_benchmark/`](speed_benchmark/):

```bash
GPU=0 bash speed_benchmark/run_rekv.sh          # ReKV: ViT encoding + LLM pre-fill
GPU=0 bash speed_benchmark/run_streamforest.sh  # StreamForest (SigLIP): ViT encoding
GPU=0 bash speed_benchmark/run_dispider.sh      # Dispider (CLIP): ViT encoding
```

Each runs the baseline and +STC configurations and prints the reduction. Pin a
dedicated GPU and read **min / median** over repeats — absolute latency is
GPU-dependent, the **reduction ratio** is what reproduces. See
[`speed_benchmark/README.md`](speed_benchmark/README.md) for options and methodology.

---

## 🏗️ Architecture

The codebase is organized around the standalone **`stc`** package; each framework consumes it through its own model wrappers / eval drivers.

<details>
<summary><b>Repository layout</b></summary>

```
STC/
├── stc/                      Standalone Python package
│   ├── config.py                 Dataclasses + env-driven config
│   ├── core/                     Shared algorithms (token similarity, layer ratios)
│   ├── cacher/                   STC-Cacher
│   │   ├── state.py                  Per-stream cache state
│   │   ├── reference_forward.py      Selective ViT-layer forwards
│   │   └── graph.py                  CUDA-graph runner for selective frames
│   ├── pruner/                   STC-Pruner (scoring, anchors, index mapping, specs)
│   └── integrations/
│       ├── hf_vit.py                 register_stc_cacher() for HF CLIP / SigLIP
│       └── streaming.py              enable_streaming_cacher() — per-frame wrapper
├── models/                   Framework code + entrypoints (rekv, StreamForest, Dispider, livecc)
├── speed_benchmark/          Latency harness (benchmark.py + run.sh)
├── scripts/eval_rekv/        ReKV evaluation drivers
└── results/                  Run outputs (git-ignored)
```

</details>

**How the two components attach:**

- **STC-Cacher** is a *monkey-patch*: `register_stc_cacher()` swaps each `vision_model.encoder.layers[*].forward` for a selective-recompute forward. For batched towers, `enable_streaming_cacher()` additionally splits a clip into per-frame calls so the cache advances frame-by-frame (reset once per video with `reset_streaming_cacher()`).
- **STC-Pruner** is an *explicit call*: `STCPruner().compress(...)` runs after vision encoding / projection / pooling and before LLM pre-fill.

---

## 🙏 Acknowledgements

Built on the excellent work of
[ReKV](https://github.com/Becomebright/ReKV),
[StreamForest](https://github.com/MCG-NJU/StreamForest),
[Dispider](https://github.com/Mark12Ding/Dispider), and
[LiveCC](https://github.com/showlab/livecc) — thanks to all authors for releasing their code.

---

## ✏️ Citation

If STC helps your research, please consider citing:

```bibtex
@inproceedings{wang2026stc,
  title     = {Accelerating Streaming Video Large Language Models via Hierarchical Token Compression},
  author    = {Wang, Yiyu and Liu, Xuyang and Gui, Xiyan and Lin, Xinying and
               Yang, Boxue and Liao, Chenfei and Chen, Tailai and Zhang, Linfeng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## 📮 Contact

Questions about the paper or code? Email `liuxuyang@stu.scu.edu.cn` or `ustywan8@ljmu.ac.uk`.
