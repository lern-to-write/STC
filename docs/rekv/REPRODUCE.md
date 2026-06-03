# Reproducing ReKV On A GPU Machine

This guide runs the vendored ReKV code in `models/rekv` on a normal Linux GPU
machine. It does not require the original ReKV repository after this repository
has been cloned.

## 1. Create Environment

```bash
cd STC_new

conda create -n stc-rekv python=3.10 -y
conda activate stc-rekv

# Install a PyTorch build matching your CUDA driver first.
# Example for CUDA 12.1; adjust for your machine if needed.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -e .[hf]
pip install -r requirements.txt
```

`requirements.txt` is a reference environment snapshot. If one pinned package is
not available for your CUDA/Python version, install the matching wheel manually
and rerun the rest of the setup.

## 2. Prepare Model And One Video

Choose a Hugging Face cache/data root:

```bash
export HF_HOME=/path/to/hugging_face
mkdir -p "$HF_HOME"
```

Download a LLaVA-OneVision checkpoint. The 0.5B model is faster for a smoke
check; the 7B model matches the default research setting.

```bash
huggingface-cli download llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  --local-dir "$HF_HOME/llava-onevision-qwen2-0.5b-ov-hf"

export REKV_LLAVA_OV_05B_PATH="$HF_HOME/llava-onevision-qwen2-0.5b-ov-hf"
```

Prepare any short local mp4 and point the smoke annotation to it by editing:

```text
benchmarks/offline/smoke/smoke_rekv.json
```

Set the first sample's `video_path` to your mp4 path. The smoke question can be
changed to match your video; its purpose is only to verify that model loading,
video encoding, retrieval, and generation run end to end.

## 3. Run Baseline ReKV Smoke

```bash
export HF_HOME=/path/to/hugging_face
export REKV_MODEL=llava_ov_0.5b
export CUDA_VISIBLE_DEVICES=0

bash scripts/eval_rekv/eval_rekv_smoke.sh rekv
```

Expected output directory:

```text
results/smoke_rekv
```

## 4. Run ReKV + STC Smoke

```bash
export HF_HOME=/path/to/hugging_face
export REKV_MODEL=llava_ov_0.5b
export CUDA_VISIBLE_DEVICES=0

bash scripts/eval_rekv/eval_rekv_smoke.sh rekv_stc
```

The smoke script sets these STC values for `rekv_stc`:

```bash
STC_PATCH_VISION=1
STC_TOKEN_PER_FRAME=64
STC_UPDATE_TOKEN_RATIO=0.25
```

For baseline `rekv`, it keeps full LLaVA-OneVision visual tokens:

```bash
STC_PATCH_VISION=0
STC_TOKEN_PER_FRAME=196
STC_UPDATE_TOKEN_RATIO=1.0
```

## 5. Run Offline Benchmarks

After downloading benchmark annotations and videos under `benchmarks/offline`,
run the unified distributed entry point through the wrapper script:

```bash
CUDA_VISIBLE_DEVICES=0 \
STC_PATCH_VISION=1 \
STC_TOKEN_PER_FRAME=64 \
STC_UPDATE_TOKEN_RATIO=0.25 \
bash scripts/eval_rekv/eval_offline_benchs.sh \
  --dataset mlvu \
  --model llava_ov_0.5b \
  --num_gpus 1 \
  --num_processes 1 \
  --save_dir results/mlvu_rekv_stc
```

Supported dataset names are defined in:

```text
models/rekv/model/video_qa/configs.py
```

## 6. Run OVO-Bench

Prepare OVO-Bench metadata and videos, then set paths explicitly:

```bash
export HF_HOME=/path/to/hugging_face
export ANNO_PATH=/path/to/ovo_bench_new.json
export VIDEO_DIR=/path/to/OVO-Bench/src_videos
export CHUNKED_DIR=/path/to/OVO-Bench/chunked_videos
export RESULT_DIR=results/ovobench_rekv_stc

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
TOTAL_PROCESSES=4 \
STC_PATCH_VISION=1 \
STC_TOKEN_PER_FRAME=64 \
STC_UPDATE_TOKEN_RATIO=0.25 \
bash scripts/eval_rekv/ovobench_scripts/eval_rekv.sh
```

## Minimal Checklist

- `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
- `REKV_LLAVA_OV_05B_PATH` or `REKV_LLAVA_OV_7B_PATH` points to a local model.
- `benchmarks/offline/smoke/smoke_rekv.json` points to an existing mp4.
- `bash scripts/eval_rekv/eval_rekv_smoke.sh rekv` completes before running full benchmarks.
- `bash scripts/eval_rekv/eval_rekv_smoke.sh rekv_stc` completes before reporting STC results.

## Troubleshooting

- `video path does not exist`: update `video_path` in the smoke or benchmark annotation JSON.
- `Unknown model`: use `llava_ov_0.5b` or `llava_ov_7b`, or add a model entry in `models/rekv/model/video_qa/utils/model_utils.py`.
- `flash_attn` build/import errors: install a wheel matching the machine CUDA and PyTorch version.
- Out of memory on 7B: use `REKV_MODEL=llava_ov_0.5b` for smoke or reduce processes per GPU.
