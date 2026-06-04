# Speed Benchmark

测量 STC 在 ReKV / LLaVA-OneVision-7B 上的 **ViT 编码** 与 **LLM prefill** 延迟降幅
（对应论文 Table 1）。运行时给已加载的模型插桩计时，**不改动仓库任何现有代码**。

```
speed_benchmark/
├── benchmark.py   计时 harness（一个进程跑一档，由 STC_* 环境变量控制）
└── run.sh         驱动：跑 baseline 与 ReKV+STC 两档，打印降幅对比表
```

## 用法

```bash
# 在带 GPU 的容器里、仓库根目录下
GPU=4 bash speed_benchmark/run.sh          # 跑两档并对比（默认 16 帧）
bash speed_benchmark/run.sh rekv           # 只跑 baseline
bash speed_benchmark/run.sh rekv_stc       # 只跑 ReKV+STC
```

可调环境变量：`GPU`（pin 哪张卡）、`NUM_FRAMES`（默认 16）、`REPEATS`（默认 20）、
`WARMUP`（默认 5）、`VIDEO`（用真实视频替代合成帧）、`REKV_MODEL`、`STC_CACHE_INTERVAL`
（N，默认 2）、`STC_CUDA_GRAPH`（rekv_stc 默认 1）。结果 JSON 落在 `results/speed/`。

**CUDA graph（`STC_CUDA_GRAPH=1`，rekv_stc 默认开）**：selective 帧的整塔前向用 CUDA graph
捕获一次后只 replay，消掉分配 + 上千次 kernel launch 开销。这是让 STC-Cacher 在 H20 上
真正加速 ViT 的关键——纯 ViT 隔离下 selective 从 12.9ms 降到 **6.97ms（反超 full 8.8ms）**，
输出与 eager 比特级一致；端到端 ReKV(16帧/N=4) ViT 编码 353→126ms。捕获失败会自动回退
eager，不会让 run 失败。`STC_CUDA_GRAPH=0` 可关掉做对照。

输出示例：

```
  stage         baseline       +STC        speedup  reduction
  ViT encode      250.9  ->    185.3 ms    x1.35   (↓26.1%)
  LLM prefill     854.4  ->    965.1 ms    x0.89   (↓-12.9%)
 paper (ReKV, Table 1): ViT ↓24.5%, LLM prefill ↓45.3%
```

## 测什么

- **ViT 编码延迟**：包住 `vision_tower.forward`。STC-Cacher 的 patch 在其内部，节省天然计入。
- **LLM prefill 延迟**：包住 `language_model.forward`，只统计带 `inputs_embeds` 的调用
  （视觉 token 预填，排除常量 system-prompt）。STC-Pruner 压低每帧 token 数 ⇒ 预填变短。

开关全由 `STC_*` 环境变量控制（与 `run_distributed` 一致）：
`baseline = PATCH_VISION=0 / TOKEN_PER_FRAME=196 / UPDATE_RATIO=1.0`，
`+STC = PATCH=1 / 64 / 0.25 / CACHE_INTERVAL=2`（= 论文 N=2）。

## 环境要求（重要）

- **transformers 4.46**：ReKV 的 llava_onevision 代码按旧结构写（顶层 `model.language_model`）。
  transformers 5.x 把这些属性挪到 `model.model.*`，会直接报错。本机用
  `envs/lmms-streamforest-py312-tf446`（tf 4.46 + flash_attn）。
- **logzero**：装在独立目录 `envs/_stc_extra_site`，经 PYTHONPATH 注入，不污染现有环境。
- `run.sh` 已把上述 python 与 site 目录设为默认；换机器时用 `STC_PY` / `STC_EXTRA_SITE`
  / `HF_HOME` 覆盖。

## 读数注意

共享集群的 GPU 时钟（DVFS）和容器 CPU 的 kernel-launch 抖动会带来 30–50% 方差，
且通常**无权锁频**。因此：

- **pin 一张空闲卡**（`GPU=...`），别和别的任务抢；
- 看 **min / median**，不要看单次或 mean（时钟抖动只会加时间，min 近似无争用下界）；
- **绝对延迟随 GPU 型号变化，可复现的是降幅比例**。

降幅与序列长度强相关：**ViT** 在短序列（16 帧）才有明显加速，长序列因 selective-recompute
的固定 bookkeeping 开销在快卡上摊不薄而退化为打平；**LLM prefill** 反之，需要足够长的序列
（如 64 帧）盖过 ReKV 的 KV-cache 固定开销才显著（16→32→64 帧：↓−13% → ↓+7.5% → ↓~40%）。
完整实测见 [`../results/speed/SUMMARY.md`](../results/speed/SUMMARY.md)。
```
