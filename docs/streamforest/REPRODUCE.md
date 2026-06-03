# Reproducing StreamForest On A GPU Machine

This guide is for a normal Linux machine with NVIDIA GPU, CUDA, and internet
access. It runs the patched StreamForest code under `models/StreamForest`.

## 1. Create Environment

```bash
cd STC_new/models/StreamForest

conda create -n streamforest python=3.10 -y
conda activate streamforest

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `flash-attn` fails to build, install a wheel matching your CUDA/PyTorch
version, then rerun the last command.

## 2. Download Model And Data

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

For the OVO-Bench smoke test, prepare videos as:

```bash
$HF_HOME/OVO-Bench/chunked_videos/0.mp4
```

The complete expected OVO-Bench video layout is:

```bash
$HF_HOME/OVO-Bench/chunked_videos/*.mp4
```

## 3. Set Paths

```bash
export HF_HOME=/path/to/hugging_face
export STREAMFOREST_DATA_ROOT="$HF_HOME"
export STREAMFOREST_CKPT_PATH="$HF_HOME/StreamForest-Qwen2-7B"
export STREAMFOREST_ANNO_ROOT="$HF_HOME/StreamForest-Annodata/eval"
export STREAMFOREST_OUTPUT_DIR="$PWD/results"

source scripts/env/streamforest_env.sh
```

If your videos are somewhere else, set the task-specific root:

```bash
export STREAMFOREST_OVOBENCH_ROOT=/path/to/OVO-Bench/chunked_videos
```

## 4. Run A Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/run_smoke.sh
```

This runs one `ovobench_backward_tracking` sample with `MAX_FRAMES=8`. Results
are written to:

```bash
$STREAMFOREST_OUTPUT_DIR/eval/StreamForest-Qwen2-7B
```

## 5. Run More Tasks

```bash
CUDA_VISIBLE_DEVICES=0 \
TASKS=ovobench_backward_tracking \
LIMIT=10 \
MAX_FRAMES=128 \
bash scripts/eval/run_eval.sh
```

For full benchmarks, remove `LIMIT` and use the task groups you need:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TASKS="ovobench,streamingbench,videomme,mlvu_mc" \
NUM_GPUS=4 \
MAX_FRAMES=2048 \
bash scripts/eval/run_eval.sh
```

Slurm is optional. Enable it only on clusters that support Slurm:

```bash
STREAMFOREST_USE_SLURM=1 \
PARTITION=<your_partition> \
TASKS=ovobench \
NUM_GPUS=8 \
bash scripts/eval/run_eval.sh
```

## Minimal Checklist

- `STREAMFOREST_CKPT_PATH` points to `StreamForest-Qwen2-7B`.
- `STREAMFOREST_ANNO_ROOT` contains `OVOBench/json/backward_tracking.json`.
- `STREAMFOREST_OVOBENCH_ROOT` or `$HF_HOME/OVO-Bench/chunked_videos` contains
  `0.mp4`.
- `CUDA_VISIBLE_DEVICES=0 bash scripts/eval/run_smoke.sh` finishes one sample.

## Troubleshooting

- `video path does not exist`: set `STREAMFOREST_OVOBENCH_ROOT`.
- `Loading local JSON dataset` does not appear: check `STREAMFOREST_ANNO_ROOT`.
- `accelerate: command not found`: activate the conda environment and reinstall
  `requirements.txt`.
- CUDA or `flash_attn` import errors: install PyTorch and `flash-attn` versions
  matching the machine CUDA driver.
