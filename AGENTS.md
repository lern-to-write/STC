# AGENTS.md — STC_new

Streaming Token Compression（STC-Cacher + STC-Pruner），从 ReKV 抽出来的库。
`stc/` 是 Python 包；`models/`、`scripts/`、`benchmarks/`、`results/` 是
vendored 的研究代码 + 驱动脚本。

总览见 [README.md](file:///apdcephfs_tj5/share_303570626/yiyuwang/work_space/STC_new/README.md)；
本文件只补充 README 没说、但 agent 容易踩坑的点。

仓库远程：`https://github.com/lern-to-write/STC.git`，工作分支
`refactor/stc-package-cleanup`。

## 机器与开发环境（前置条件）

> ⚠️ **当前登录节点是跳板机，没有 GPU。所有 GPU 算力都在容器里。
> 严禁在跳板机上跑任何需要 GPU 的命令**（包括下面所有 `eval_*.sh`、
> `run_distributed.py`、`reproduce_smoke.py`）。

进容器：

```bash
taiji_client exec G085_aigc_VLM_streaming_agent_v2 8b1d81b89e45274d019e493c682e02cc bash
```

约定：

- **不要改公共容器环境**。新虚拟环境一律放在
  `/apdcephfs_tj5/share_303570626/yiyuwang/envs/`（已有
  `lmms-streamforest-py312-tf446`、`vllm_qwen36*`、`vst-qwen35` 等）。
- 下依赖/模型前必须开代理：
  ```bash
  export http_proxy=http://star-proxy.oa.com:3128
  export https_proxy=http://star-proxy.oa.com:3128
  export ftp_proxy=http://star-proxy.oa.com:3128
  ```
  内网白名单 `no_proxy` 见
  [`/apdcephfs_tj5/share_303570626/yiyuwang/work_space/doc_space/base/machine_base_do_v2.md`](file:////apdcephfs_tj5/share_303570626/yiyuwang/work_space/doc_space/base/machine_base_do_v2.md)。
- HuggingFace 缓存：`export HF_HOME=/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face`
  （仓库内所有 shell 脚本都默认这个路径）。
- 共享盘：`/apdcephfs_tj5`（多 TB cephfs）。
- 工作空间根：`/apdcephfs_tj5/share_303570626/yiyuwang/work_space`。

## 安装与冒烟

```bash
pip install -e .          # 只装核心（仅 torch）
pip install -e .[hf]      # 加上 transformers，hf_vit 集成需要
```

[`requirements.txt`](file:///apdcephfs_tj5/share_303570626/yiyuwang/work_space/STC_new/requirements.txt)
是参考环境的 `pip freeze` 快照（torch 2.8.0+cu128、flash_attn 2.8.3、
transformers 装的是 GitHub 上某个 commit）。**它与 `pyproject.toml`
的 `torch>=2.1` 不是真实依赖声明**，别盲目用 `pip install -r requirements.txt`
覆盖现有环境。

**仓库里没有单元测试、没有 lint/format/typecheck 配置、没有 CI、没有 pre-commit。**
唯一的端到端验证是
[`scripts/eval_rekv/`](file:///apdcephfs_tj5/share_303570626/yiyuwang/work_space/STC_new/scripts/eval_rekv/)
（README 写的是 `scripts/eval/`，**真实路径是 `scripts/eval_rekv/`**）。
最便宜的冒烟：

```bash
bash scripts/eval_rekv/eval_rekv_smoke.sh rekv        # 基线 ReKV
bash scripts/eval_rekv/eval_rekv_smoke.sh rekv_stc    # ReKV + STC
```

## 跑 ReKV 评估器（主入口）

入口是 `model.video_qa.run_distributed`，**不是 `stc.*` 下的模块**，位于
[`models/rekv/`](file:///apdcephfs_tj5/share_303570626/yiyuwang/work_space/STC_new/models/rekv/)，
要求 `PYTHONPATH` 同时包含**项目根**和 **`models/rekv`**：

```bash
export PYTHONPATH="$PWD/models/rekv:$PWD:${PYTHONPATH:-}"
export STC_PATCH_VISION=1
export STC_TOKEN_PER_FRAME=64
export STC_UPDATE_TOKEN_RATIO=0.25
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
  -m model.video_qa.run_distributed --dataset smoke --model llava_ov_7b \
  --save_dir results/foo
```

DDP 后端是 **gloo**（见
[run_distributed.py:32](file:///apdcephfs_tj5/share_303570626/yiyuwang/work_space/STC_new/models/rekv/model/video_qa/run_distributed.py#L32)），
不是 nccl。数据集名称在
[`configs.py`](file:///apdcephfs_tj5/share_303570626/yiyuwang/work_space/STC_new/models/rekv/model/video_qa/configs.py)
注册：`smoke / videomme / videomme_subset / mlvu / egoschema / egoschema_subset
/ qaego4d / cgbench / activitynet_qa / rvs_ego / rvs_movie`。

## STC 环境变量控制

[`stc/config.py`](file:///apdcephfs_tj5/share_303570626/yiyuwang/work_space/STC_new/stc/config.py)
里 `GlobalConfig.initialize_from_env()` 从 `STC_*` 环境变量加载 ReKV 的 STC 配置。
STC 相关 CLI 参数已经移除，后续不要再给 `run_distributed.py` 传
`--cache_strategy` / `--prune_strategy` / `--token_per_frame` /
`--update_token_ratio`。

常用变量：

- `STC_PATCH_VISION=0|1` 控制是否真正 patch HF vision tower。
- `STC_TOKEN_PER_FRAME=<int>` 控制每帧保留 token 数；`196` 对 LLaVA-OV 近似全量保留。
- `STC_UPDATE_TOKEN_RATIO=<float>` 控制 selective recompute 比例。
- `STC_CACHE_INTERVAL=<int>` 控制 reference refresh 间隔，默认 `2`。

其他 STC 算法参数固定：

- `STC_PATCH_VISION=1` 时 cacher 策略固定为 `selective`，否则为 `none`。
- selector metric 固定为 `cosine`。
- pruner 策略固定为 `gaussian`；要"关闭剪枝"，请配合 `STC_TOKEN_PER_FRAME=196`。
- `encode_chunk_size=1`、`channel_keep_ratio=0.5`、`spatial_temporal_alpha=0.5`。
- `STC_UPDATE_TOKEN_RATIO` 必须在 `(0, 1]`，传 `0.0` 直接抛异常。

`STCConfig` 是**进程级单例**。`run_distributed.py` 每个 rank 调一次
`GlobalConfig.initialize_from_env()`；而绕过单例的代码路径
（`register_stc_cacher(...)`、`STCPruner(...)`）需要显式传
`CacheConfig` / `ModelConfig`，否则才落回默认。

`STCPruner.compress(features, model=...)` 只认
[`MODEL_SPECS`](file:///apdcephfs_tj5/share_303570626/yiyuwang/work_space/STC_new/stc/pruner/specs.py)
里的三个键：`llava_ov`、`llava_vid`、`clip`。

## 模型权重路径解析

[`utils/model_utils.py`](file:///apdcephfs_tj5/share_303570626/yiyuwang/work_space/STC_new/models/rekv/model/video_qa/utils/model_utils.py)
按这个顺序找权重：环境变量 → `model_zoo/<本地名>` → `$HF_HOME/hub` 缓存。

可用的覆盖环境变量：

- `REKV_LLAVA_OV_7B_PATH` 对应 `llava_ov_7b`

`HF_HOME` 在所有 shell 脚本里默认是
`/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face`。

**注意**：`models/rekv/model_zoo` 在仓库里不存在（`.gitignore` 显式排除），
代码里 `MODEL_ZOO = PROJECT_ROOT / "model_zoo"` 也只在你手动建该目录时有效；
正常依赖环境变量或 HF cache。

`MODEL_REGISTRY` 里 `video_llava_7b` 和 `longva_7b` 写死了**别的机器上**的
绝对路径（`/mnt/data2/...`、`/data/wangyiyu-20250922/...`），在本机上原样运行
会失败，需要先改路径或加环境变量分支。


## 其他需要知道的约定

- `models/{Dispider,StreamForest,livecc}/` 是**故意保持上游原样**的 vendored
  代码，不要随意 patch（除非用户明确要求）。
- 日志/注释中英文混杂是有意的，不要"统一翻译"，下游解析可能 match 这些字符串。
- `STCCache` 是**每流一个**实例，不是单例；`stc.default_cache()` 是兼容旧代码的
  进程级默认值，新代码请显式 `STCCache()`。
- `.gitignore` 把 `results/`、`models/StreamForest/ckpt/`、
  `models/livecc/{webpage,demo/sources}/`、`models/rekv/model_zoo` 都排掉了 ——
  跑评估产生的输出和模型权重都不会被 commit。
