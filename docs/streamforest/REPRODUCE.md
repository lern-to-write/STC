# Reproducing StreamForest On A GPU Machine

This guide is for a normal Linux machine with NVIDIA GPU and CUDA. It covers
the patched `models/StreamForest` code in this repo, including the current
StreamForest + `STC-Cacher` path.

The current StreamForest integration only monkey patches the vision tower for
`STC-Cacher`. It does not apply `STC-Pruner`.

## 1. Check GPU Visibility First

Before installing anything else, make sure the exact shell you will use for
evaluation can already see CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected output is `True` and at least `1`. If this prints `False 0`, fix the
machine shell / container / driver setup first. Installing more Python
packages will not solve that problem.

## 2. Create Environment

```bash
cd STC_new/models/StreamForest

conda create -n streamforest python=3.12 -y
conda activate streamforest

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `requirements.txt` stops on `flash-attn`, do not block on that for the smoke
test. Install the missing runtime packages explicitly:

```bash
pip install \
  accelerate==0.29.3 \
  evaluate==0.4.1 \
  pytz==2024.2 \
  sqlitedict==2.1.0 \
  av==13.1.0
```

Notes:

- We verified the one-sample smoke path without `flash-attn`.
- `torch` must still be a CUDA-enabled build that matches your machine.
- To avoid shell ambiguity, it is better to pin the Python entry point:

```bash
export PYTHON_EXECUTABLE="$(which python)"
```

## 3. Download Model And Data

Put all Hugging Face assets under one directory:

```bash
export HF_HOME=/path/to/hugging_face
mkdir -p "$HF_HOME"
```

Download the StreamForest checkpoint:

```bash
huggingface-cli download MCG-NJU/StreamForest-Qwen2-7B \
  --local-dir "$HF_HOME/StreamForest-Qwen2-7B"
```

Download the official StreamForest annotation package:

```bash
huggingface-cli download MCG-NJU/StreamForest-Annodata \
  --repo-type dataset \
  --local-dir "$HF_HOME/StreamForest-Annodata"
```

For OVO-Bench, prepare videos under:

```bash
$HF_HOME/OVO-Bench/chunked_videos/*.mp4
```

For the smoke test, at least this file should exist:

```bash
$HF_HOME/OVO-Bench/chunked_videos/0.mp4
```

## 4. Set Paths

```bash
export HF_HOME=/path/to/hugging_face
export STREAMFOREST_DATA_ROOT="$HF_HOME"
export STREAMFOREST_CKPT_PATH="$HF_HOME/StreamForest-Qwen2-7B"
export STREAMFOREST_ANNO_ROOT="$HF_HOME/StreamForest-Annodata/eval"
export STREAMFOREST_OVOBENCH_ROOT="$HF_HOME/OVO-Bench/chunked_videos"
export STREAMFOREST_OUTPUT_DIR="$PWD/results"

source scripts/env/streamforest_env.sh
```

`streamforest_env.sh` also adds both `models/StreamForest` and the repo root to
`PYTHONPATH`, so the local `stc` package is importable.

If your local annotation or video layout differs from the commands above,
override the corresponding environment variables directly.

## 5. Enable STC-Cacher

Set `STC_PATCH_VISION=1` to enable the StreamForest-side monkey patch:

```bash
export STC_PATCH_VISION=1
```

Optional runtime knobs:

```bash
export STC_UPDATE_TOKEN_RATIO=0.25
export STC_CACHE_INTERVAL=2
```

## 6. Run A Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 \
LIMIT=1 \
MAX_FRAMES=8 \
bash scripts/eval/run_smoke.sh
```

This runs one `ovobench_backward_tracking` sample. Results are written under:

```bash
$STREAMFOREST_OUTPUT_DIR/eval/StreamForest-Qwen2-7B
```

This is the exact path we used to verify that the current StreamForest +
`STC-Cacher` integration can load the model, load OVO-Bench annotations, run
generation, and save outputs.

## 7. Run More Tasks

OVO-Bench group smoke test. This verifies that all three OVO-Bench subtasks
load and run:

```bash
CUDA_VISIBLE_DEVICES=0 \
TASKS=ovobench \
LIMIT=1 \
MAX_FRAMES=8 \
bash scripts/eval/run_eval.sh
```

Single-task example:

```bash
CUDA_VISIBLE_DEVICES=0 \
TASKS=ovobench_backward_tracking \
LIMIT=10 \
MAX_FRAMES=128 \
bash scripts/eval/run_eval.sh
```

Multi-task example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TASKS="ovobench,streamingbench,videomme,mlvu_mc" \
NUM_GPUS=4 \
MAX_FRAMES=2048 \
bash scripts/eval/run_eval.sh
```

Slurm is optional. Only enable it on clusters that really use Slurm:

```bash
STREAMFOREST_USE_SLURM=1 \
PARTITION=<your_partition> \
TASKS=ovobench \
NUM_GPUS=8 \
bash scripts/eval/run_eval.sh
```

## Minimal Checklist

- `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`
  returns `True` and at least `1`.
- `PYTHON_EXECUTABLE` points to the environment you installed.
- `STREAMFOREST_CKPT_PATH` points to `StreamForest-Qwen2-7B`.
- `STREAMFOREST_ANNO_ROOT` contains `OVOBench/json/backward_tracking.json`.
- `STREAMFOREST_OVOBENCH_ROOT` contains `0.mp4`.
- `STC_PATCH_VISION=1 CUDA_VISIBLE_DEVICES=0 bash scripts/eval/run_smoke.sh`
  finishes one sample.

## Troubleshooting

- `torch.cuda.device_count() == 0`: your current shell cannot see GPU. Fix the
  shell / container / driver first.
- `video path does not exist`: set `STREAMFOREST_OVOBENCH_ROOT`.
- `Loading local JSON dataset` does not appear: check `STREAMFOREST_ANNO_ROOT`.
- `accelerate: command not found`: activate the target environment and export
  `PYTHON_EXECUTABLE="$(which python)"`.
- `ModuleNotFoundError: sqlitedict`, `evaluate`, `pytz`, or `av`: install the
  exact versions listed above.
- `petrel_client` warnings can be ignored when you are loading model/data from
  local disk or Hugging Face cache.
- `flash-attn` build failure: not required for the verified one-sample smoke
  path in this repo.
