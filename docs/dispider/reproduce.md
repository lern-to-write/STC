# Reproduce Dispider On A Clean GPU Machine

This document is for a new Linux GPU machine. It does not assume the Taiji container or any `/apdcephfs_tj5/...` paths.

## Hardware And Driver

Minimum practical setup:

- Linux x86_64 machine with NVIDIA GPU.
- NVIDIA driver new enough to run CUDA 11.8 user-space libraries.
- Recommended GPU memory: 80GB for full OVO-Bench / long-video inference. Smaller GPUs may work for smoke tests but can OOM on full evaluation.
- Python 3.10.

Dispider's upstream README was reproduced with CUDA 11.8, PyTorch 2.2.0, transformers 4.41.2, and flash-attn 2.5.9.post1. Do not build flash-attn against CUDA 12.x for this setup.

## Directory Layout

Use any writable workspace. The commands below assume:

```bash
export WORKDIR=/data/dispider_repro
mkdir -p "$WORKDIR"
cd "$WORKDIR"
```

Expected layout after setup:

```text
$WORKDIR/
  STC_new/
  checkpoints/
    Dispider/
    clip-vit-large-patch14/
  OVO-Bench/
    ovo_bench_new.json
    chunked_videos/
```

## Get Code

Clone or copy this repository, then enter Dispider:

```bash
cd "$WORKDIR"
git clone <YOUR_STC_NEW_REPO_URL> STC_new
cd "$WORKDIR/STC_new/models/Dispider"
```

If the code was already copied instead of cloned, only the final `cd` matters.

## Create Environment

Recommended: use conda or micromamba so CUDA 11.8 toolkit is installed inside the environment.

```bash
conda create -n dispider python=3.10 -y
conda activate dispider

conda install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install "numpy<2" packaging ninja wheel setuptools
MAX_JOBS=8 pip install flash-attn==2.5.9.post1 --no-build-isolation
pip install transformers==4.41.2 deepspeed==0.9.5 accelerate==0.27.2 \
  pydantic==1.10.13 timm==0.6.13 decord shortuuid sentencepiece protobuf
```

Validate the environment:

```bash
nvcc --version
python - <<'PY'
import torch, transformers, flash_attn
print("torch:", torch.__version__, "torch cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("flash_attn:", flash_attn.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

Expected:

- `nvcc` reports CUDA 11.8.
- `torch` reports `2.2.0+cu118`.
- `torch.version.cuda` reports `11.8`.
- `flash_attn` imports successfully.

## Download Checkpoints

Download Dispider and CLIP to ordinary local paths:

```bash
cd "$WORKDIR"
pip install -U huggingface_hub

huggingface-cli download Mar2Ding/Dispider \
  --local-dir "$WORKDIR/checkpoints/Dispider"

huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir "$WORKDIR/checkpoints/clip-vit-large-patch14"

export MODEL_PATH="$WORKDIR/checkpoints/Dispider"
export CLIP_CKPT_PATH="$WORKDIR/checkpoints/clip-vit-large-patch14"
export DISPIDER_CLIP_CKPT_PATH="$CLIP_CKPT_PATH"
```

If your machine cannot access Hugging Face directly, download the same two repositories elsewhere and copy them into the two local directories above.

## Prepare OVO-Bench Data

Prepare OVO-Bench according to the OVO-Bench data release. The Dispider script expects:

```text
$WORKDIR/OVO-Bench/ovo_bench_new.json
$WORKDIR/OVO-Bench/chunked_videos/0.mp4
$WORKDIR/OVO-Bench/chunked_videos/1558_0.mp4
...
```

Set:

```bash
export ANNO_PATH="$WORKDIR/OVO-Bench/ovo_bench_new.json"
export CHUNKED_DIR="$WORKDIR/OVO-Bench/chunked_videos"
test -f "$ANNO_PATH"
test -d "$CHUNKED_DIR"
```

The `chunked_videos` filenames must match the OVO-Bench annotation ids:

- Backward / realtime tasks use `{id}.mp4`.
- Forward tasks use `{id}_{index}.mp4`.

## Single Video Smoke Test

```bash
cd "$WORKDIR/STC_new/models/Dispider"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export DISPIDER_CLIP_CKPT_PATH="$CLIP_CKPT_PATH"

python inference.py \
  --model_path "$MODEL_PATH" \
  --video_path "$CHUNKED_DIR/1637_0.mp4" \
  --prompt "What is happening in the video?"
```

Use any existing mp4 from `chunked_videos` if `1637_0.mp4` is not present.

## OVO-Bench Smoke Test

Run one sample:

```bash
cd "$WORKDIR/STC_new/models/Dispider"
MAX_SAMPLES=1 TASKS=EPM NUM_GPUS=1 NUM_CHUNKS=1 \
MODEL_PATH="$MODEL_PATH" \
CLIP_CKPT_PATH="$CLIP_CKPT_PATH" \
ANNO_PATH="$ANNO_PATH" \
CHUNKED_DIR="$CHUNKED_DIR" \
bash scripts/eval/ovobench.sh
```

## Full OVO-Bench

Single GPU:

```bash
NUM_GPUS=1 NUM_CHUNKS=1 \
MODEL_PATH="$MODEL_PATH" \
CLIP_CKPT_PATH="$CLIP_CKPT_PATH" \
ANNO_PATH="$ANNO_PATH" \
CHUNKED_DIR="$CHUNKED_DIR" \
bash scripts/eval/ovobench.sh
```

Eight GPUs, sharded by sample:

```bash
NUM_GPUS=8 NUM_CHUNKS=8 \
MODEL_PATH="$MODEL_PATH" \
CLIP_CKPT_PATH="$CLIP_CKPT_PATH" \
ANNO_PATH="$ANNO_PATH" \
CHUNKED_DIR="$CHUNKED_DIR" \
bash scripts/eval/ovobench.sh
```

Outputs are written to:

```text
$WORKDIR/STC_new/models/Dispider/results/ovobench/Dispider/
```

## Optional Scoring

If this repository also contains the OVO-Bench scorer under `models/rekv/model/online_bench_inference/ovobench`, run:

```bash
RUN_SCORE=1 NUM_GPUS=8 NUM_CHUNKS=8 \
MODEL_PATH="$MODEL_PATH" \
CLIP_CKPT_PATH="$CLIP_CKPT_PATH" \
ANNO_PATH="$ANNO_PATH" \
CHUNKED_DIR="$CHUNKED_DIR" \
bash scripts/eval/ovobench.sh
```

If the scorer is not present, the inference JSON files are still generated and can be scored with the official OVO-Bench scorer.

## Troubleshooting

- `flash_attn` build fails or imports with undefined CUDA symbols: check `which nvcc` and `nvcc --version`. It must point to CUDA 11.8 for this reproduction.
- PyTorch warns about NumPy ABI: install `numpy<2`.
- CLIP path error: set `DISPIDER_CLIP_CKPT_PATH` or pass `CLIP_CKPT_PATH` to `ovobench.sh`.
- OOM on smaller GPUs: first verify with `MAX_SAMPLES=1`. For non-official smoke runs, reduce `DISPIDER_NUM_FRAMES` or `DISPIDER_MAX_CLIPS`; do not use reduced settings for reported full benchmark numbers.
