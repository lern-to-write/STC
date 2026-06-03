# Dispider Change Log

This document records the final kept changes after reverting the earlier broad compatibility edits. The code diff against the upstream Dispider checkout is saved in:

```text
docs/dispider/upstream_diff.patch
```

## Summary

The final code changes are intentionally small:

- Keep upstream Dispider model code unchanged except for one configurable CLIP checkpoint path.
- Add a standalone OVO-Bench inference script for Dispider.
- Add a shell entrypoint that works on a clean GPU machine with user-provided paths.
- Add reproduction documentation for clean-machine setup.

No transformers-version compatibility hacks are kept.

## Changed Files

### `models/Dispider/dispider/model/language_model/builder.py`

What changed:

```diff
-        self.vision_encoder = CLIPVisionTower('YOUR_CLIP_CKPT_PATH')
+        clip_ckpt_path = os.environ.get('DISPIDER_CLIP_CKPT_PATH', 'YOUR_CLIP_CKPT_PATH')
+        self.vision_encoder = CLIPVisionTower(clip_ckpt_path)
```

Why:

- Upstream hard-codes the placeholder `YOUR_CLIP_CKPT_PATH`, so the released inference code cannot run without editing source.
- The change keeps upstream behavior as the fallback but allows users to set `DISPIDER_CLIP_CKPT_PATH` at runtime.
- This is the only modification to an upstream model file.

### `models/Dispider/dispider/eval/model_ovobench.py`

What changed:

- Added a new OVO-Bench inference script for Dispider.
- Reuses Dispider's existing model loading and generation path.
- Implements OVO-Bench prompt formatting for backward, realtime, and forward tasks.
- Supports task filtering with `--tasks`.
- Supports sharding with `--num-chunks` and `--chunk-idx`.
- Supports quick validation with `--max-samples`.
- Writes results in the same grouped format expected by the existing OVO scorer:

```json
{
  "backward": [],
  "realtime": [],
  "forward": []
}
```

Why:

- The upstream Dispider repo provides VideoMME evaluation but does not provide OVO-Bench evaluation.
- The new script adds OVO-Bench support without changing upstream `inference.py` or model internals.
- Sharding is implemented at the annotation level so multi-GPU runs can be launched with simple shell parallelism.

### `models/Dispider/scripts/eval/ovobench.sh`

What changed:

- Added a shell entrypoint for OVO-Bench.
- Does not assume internal cluster paths.
- Uses normal environment variables:

```bash
MODEL_PATH=/path/to/Dispider
CLIP_CKPT_PATH=/path/to/clip-vit-large-patch14
ANNO_PATH=/path/to/ovo_bench_new.json
CHUNKED_DIR=/path/to/chunked_videos
RESULT_DIR=/path/to/results
NUM_GPUS=8
NUM_CHUNKS=8
TASKS="EPM ASI HLD OCR ACR ATR STU FPD OJR REC SSR CRR"
bash scripts/eval/ovobench.sh
```

Why:

- External users should be able to run the evaluation on a clean GPU machine.
- Data/model paths vary across machines and must be runtime configuration, not source edits.
- The script keeps `ANNO_PATH` and `CHUNKED_DIR` required because there is no universal local location for OVO-Bench data.

### `docs/dispider/reproduce.md`

What changed:

- Added clean-machine reproduction instructions.
- Documents hardware assumptions, CUDA 11.8 setup, Python environment creation, dependency versions, checkpoint download, OVO-Bench data layout, smoke tests, and full evaluation commands.
- Does not require Taiji, `/apdcephfs_tj5/...`, or our container base environment.

Why:

- The original instructions were too specific to our internal container.
- The target reader is someone reproducing Dispider on a separate clean GPU machine.

### `docs/dispider/changes.md`

What changed:

- This file documents the final change set and rationale.

Why:

- Reviewers need a concise audit trail: which files changed, what changed, and why.

### `docs/dispider/upstream_diff.patch`

What changed:

- Added a generated patch file containing the diff against the upstream Dispider checkout for:
  - `dispider/model/language_model/builder.py`
  - `dispider/eval/model_ovobench.py`
  - `scripts/eval/ovobench.sh`

Why:

- The user requested the git diff against the original upstream code to be written out explicitly.
- New files are represented as `/dev/null -> file` diffs so the patch is self-contained.

## How The Diff Was Generated

From `models/Dispider`:

```bash
{
  git diff -- dispider/model/language_model/builder.py
  git diff --no-index -- /dev/null dispider/eval/model_ovobench.py || true
  git diff --no-index -- /dev/null scripts/eval/ovobench.sh || true
} > ../../docs/dispider/upstream_diff.patch
```

## Validation

Validated in our internal H20 container with the same dependency versions documented in `reproduce.md`:

- Python 3.10
- CUDA 11.8 toolkit
- PyTorch 2.2.0 cu118
- transformers 4.41.2
- flash-attn 2.5.9.post1
- deepspeed 0.9.5
- accelerate 0.27.2
- pydantic 1.10.13
- timm 0.6.13
- decord

Validation commands run:

```bash
python inference.py \
  --model_path "$MODEL_PATH" \
  --video_path "$CHUNKED_DIR/1637_0.mp4" \
  --prompt "What is happening in the video?"

MAX_SAMPLES=1 TASKS=EPM NUM_GPUS=1 NUM_CHUNKS=1 \
MODEL_PATH="$MODEL_PATH" \
CLIP_CKPT_PATH="$CLIP_CKPT_PATH" \
ANNO_PATH="$ANNO_PATH" \
CHUNKED_DIR="$CHUNKED_DIR" \
bash scripts/eval/ovobench.sh
```

Both smoke tests completed successfully.
