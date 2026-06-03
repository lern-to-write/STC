# ReKV Integration Changes

This document compares the vendored ReKV code in `models/rekv` against upstream
`becomebright/ReKV`.

Upstream snapshot used for comparison:

```text
1fd9a3dbf5dbff7f27069ae2f4463674c495e830 (1fd9a3d, 2025-11-04 15:06:44 +0800, Fix type in prepare.sh)
```

The comparison maps upstream `model/` and root-level `video_qa/` into this
repository's vendored layout under `models/rekv/model/`. Upstream repository
metadata, README files, `data/`, `results/`, `model_zoo/`, and ignored Python
cache files are intentionally excluded from the diff because they are not part
of the vendored runtime code. The STC repository also adds wrapper scripts under
`scripts/eval_rekv/`; those are described below but are not part of the upstream
ReKV directory diff.

## Summary

| Area | Main files | What changed | Why |
| --- | --- | --- | --- |
| STC package integration | `llava_onevision_rekv.py`, `abstract_rekv.py` | ReKV now imports `stc`, initializes `GlobalConfig` from environment variables, optionally monkey-patches the vision tower with STC-Cacher, and compresses visual tokens with `STCPruner`. | Keep STC as a standalone package while letting ReKV run either baseline or STC mode from the same code path. |
| Runtime controls | `scripts/eval_rekv/*.sh`, `llava_onevision_rekv.py`, `abstract_rekv.py` | STC behavior is controlled by `STC_PATCH_VISION`, `STC_TOKEN_PER_FRAME`, `STC_UPDATE_TOKEN_RATIO`, and fixed package defaults. | Remove STC-specific CLI plumbing and make shell scripts reproducible through environment variables. |
| Token budget handling | `llava_onevision_rekv.py` | `n_frame_tokens` comes from `default_config().model.token_per_frame`; compressed features are reshaped using the retained token count instead of hard-coded 196. | Allow full-token baseline (`196`) and compressed STC runs (`64`, etc.) without changing model code. |
| Chunk/cache synchronization | `abstract_rekv.py` | `encode_video()` reads `encode_chunk_size` from STC config and resets the default STC cache per chunk. | Keep STC-Cacher state aligned with ReKV streaming chunks. |
| Distributed/eval refactor | `video_qa/*` | Replaced upstream `run_eval.py`, `base.py`, and task-specific scripts with `run_distributed.py`, `configs.py`, solver classes, merge/eval helpers, and smoke/VideoMME/EgoSchema subset evaluators. | Support one unified distributed entry point and shared dataset configuration across offline benchmarks. |
| Online benchmarks | `online_bench_inference/ovobench/*` | Added OVO-Bench inference, scoring, chunking, and model adapter code. | Run ReKV in online/streaming benchmark workflows from the vendored tree. |
| LongVA layout | `longva/*` | Flattened upstream `model/longva/longva` package into `models/rekv/model/longva` and added custom CLIP/SigLIP encoder files. | Make imports work from the STC vendored package layout and retain local vision encoder experiments. |
| Model loading | `video_qa/utils/model_utils.py` | Added model registry, HF cache snapshot discovery, and env overrides such as `REKV_LLAVA_OV_7B_PATH`. | Let users run from local model paths, `models/rekv/model_zoo`, HF cache, or HF repo IDs. |

## Diff Stat

```text
 /dev/null => models/rekv/model/__init__.py         |   0
 {model => models/rekv/model}/abstract_rekv.py      |  24 +-
 /dev/null => models/rekv/model/launtch.sh          |  38 ++
 .../rekv/model}/llava_onevision_rekv.py            |  86 ++-
 .../rekv/model}/longva/__init__.py                 |   0
 .../rekv/model}/longva/constants.py                |   0
 .../rekv/model}/longva/conversation.py             |   0
 /dev/null => models/rekv/model/longva/hfd.sh       | 328 ++++++++++
 .../clip_encoder.py => /dev/null                   | 175 ------
 .../rekv/model}/longva/mm_utils.py                 |   2 +-
 .../rekv/model}/longva/model/__init__.py           |   0
 .../rekv/model}/longva/model/apply_delta.py        |   0
 .../rekv/model}/longva/model/builder.py            |   0
 .../rekv/model}/longva/model/consolidate.py        |   0
 .../longva/model/language_model/llava_llama.py     |   2 +-
 .../longva/model/language_model/llava_mistral.py   |   0
 .../longva/model/language_model/llava_mpt.py       |   0
 .../longva/model/language_model/llava_qwen.py      |   2 +-
 .../longva/model/language_model/modeling_llama.py  |   0
 .../rekv/model}/longva/model/llava_arch.py         |   6 +-
 .../rekv/model}/longva/model/make_delta.py         |   0
 .../longva/model/multimodal_encoder/builder.py     |   0
 .../model/multimodal_encoder/clip_encoder.py       | 374 ++++++++++++
 .../longva/model/multimodal_encoder/custom_clip.py | 316 ++++++++++
 .../longva/model/multimodal_projector/builder.py   |   0
 .../model/multimodal_projector/pooler_projector.py |   0
 .../longva/model/multimodal_resampler/builder.py   |   0
 .../model/multimodal_resampler/masked_drop.py      |   0
 .../longva/model/multimodal_resampler/perceiver.py |   0
 .../longva/model/multimodal_resampler/qformer.py   |   0
 .../model/multimodal_resampler/spatial_pool.py     |   0
 .../rekv/model}/longva/model/utils.py              |   0
 .../longva => models/rekv/model}/longva/utils.py   |   2 +-
 {model => models/rekv/model}/longva_rekv.py        |  15 +-
 .../online_bench_inference/ovobench/constant.py    |  60 ++
 .../ovobench/inference_distributed.py              | 481 +++++++++++++++
 .../ovobench/models/Dispider.py                    | 281 +++++++++
 .../ovobench/models/FlashVStream.py                | 102 ++++
 .../online_bench_inference/ovobench/models/GPT.py  | 114 ++++
 .../ovobench/models/Gemini.py                      |  57 ++
 .../ovobench/models/LLaVA_OneVision.py             |  95 +++
 .../ovobench/models/LLaVA_Video.py                 |  89 +++
 .../ovobench/models/QWen2VL.py                     |  85 +++
 .../ovobench/models/VideoLLM_Online.py             |  72 +++
 .../online_bench_inference/ovobench/models/rekv.py |  58 ++
 .../model/online_bench_inference/ovobench/score.py |  37 ++
 .../ovobench/utils/OVOBench.py                     | 145 +++++
 .../ovobench/utils/OVOBenchScore.py                | 135 ++++
 .../ovobench/utils/chunk_videos.py                 |  62 ++
 .../ovobench/utils/sample_frames.py                |  63 ++
 {model => models/rekv/model}/patch.py              |   7 +-
 /dev/null => models/rekv/model/video_qa/README.md  | 270 ++++++++
 model/video_qa/base.py => /dev/null                | 231 -------
 .../rekv/model/video_qa/base_refactored.py         | 102 ++++
 /dev/null => models/rekv/model/video_qa/configs.py |  80 +++
 .../model/video_qa/eval/eval_egoschema_subset.py   | 679 +++++++++++++++++++++
 .../rekv/model/video_qa/eval/eval_smoke.py         |  25 +
 .../rekv/model/video_qa/eval/eval_videomme.py      | 168 +++++
 .../rekv/model/video_qa/eval/evaluate.py           |   0
 .../rekv/model/video_qa/rekv_offline_refactored.py |  77 +++
 model/video_qa/rekv_offline_vqa.py => /dev/null    |  80 ---
 .../rekv/model/video_qa/rekv_stream_refactored.py  |  76 +++
 model/video_qa/rekv_stream_vqa.py => /dev/null     |  70 ---
 .../rekv/model/video_qa/run_distributed.py         | 169 +++++
 model/video_qa/run_eval.py => /dev/null            | 276 ---------
 .../rekv/model/video_qa/solver_factory.py          |  26 +
 .../rekv/model/video_qa/utils/__init__.py          |  14 +
 .../rekv/model/video_qa/utils/data_utils.py        |  44 ++
 .../rekv/model/video_qa/utils/merge_utils.py       |  19 +
 .../rekv/model/video_qa/utils/model_utils.py       |  89 +++
 .../rekv/model/video_qa/videomme_refactored.py     |  74 +++
 71 files changed, 5014 insertions(+), 868 deletions(-)
```

## Git Diff Against Upstream ReKV

Generated with:

```bash
git clone --depth 1 https://github.com/becomebright/ReKV.git /tmp/rekv-upstream-compare
# Normalize upstream video_qa into model/video_qa, exclude ignored caches, then compare:
git diff --no-index -- model models/rekv/model
```

```diff
diff --git a/models/rekv/model/__init__.py b/models/rekv/model/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/model/abstract_rekv.py b/models/rekv/model/abstract_rekv.py
index f23daf5..14a3e78 100644
--- a/model/abstract_rekv.py
+++ b/models/rekv/model/abstract_rekv.py
@@ -1,6 +1,8 @@
 import torch
 from logzero import logger
 
+from stc import default_config, reset_default_cache
+
 
 class Abstract_ReKV:
     processor = None
@@ -13,6 +15,10 @@ class Abstract_ReKV:
         self.n_local = n_local
         self.topk = topk
         self.chunk_size = chunk_size
+        self.ram_usage=0
+        self.total_cuda_time=0
+        self.max_mem=0
+        self.total_llm_pre_time=0
 
     def clear_cache(self):
         self.kv_cache = None
@@ -33,23 +39,27 @@ class Abstract_ReKV:
         pixel_values_videos = self.processor.video_processor(video_chunk, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
         video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*196, D)
         assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'
-
         output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
         self.kv_cache = output.past_key_values
+    
+
 
     @torch.inference_mode()
-    def encode_video(self, video, encode_chunk_size=64):  # video: (Nv, H, W, 3)
-        # encode chunk by chunk
+    
+    def encode_video(self, video):  # video: (Nv, H, W, 3)
+        cfg = default_config()
+        encode_chunk_size = cfg.model.encode_chunk_size
         num_frames = video.shape[0]
         num_chunks = num_frames // encode_chunk_size
 
         for chunk_idx in range(num_chunks):
+            reset_default_cache(chunk_idx, cfg.cache.update_token_ratio)
+
             start_idx = chunk_idx * encode_chunk_size
             end_idx = start_idx + encode_chunk_size
             chunk_video = video[start_idx:end_idx]
             self._encode_video_chunk(chunk_video)
-            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
-
+            # logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
         # Handle remaining frames
         remaining_frames = num_frames % encode_chunk_size
         if remaining_frames > 0:
@@ -57,8 +67,7 @@ class Abstract_ReKV:
             end_idx = start_idx + remaining_frames
             remaining_video = video[start_idx:end_idx]
             self._encode_video_chunk(remaining_video)
-        
-        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
+        # logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
 
     @torch.inference_mode()
     def question_answering(self, input_text, max_new_tokens=128):
@@ -68,3 +77,4 @@ class Abstract_ReKV:
         n_layers = len(self.kv_cache)
         memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
         return memory
+
diff --git a/models/rekv/model/launtch.sh b/models/rekv/model/launtch.sh
new file mode 100755
index 0000000..a394d7d
--- /dev/null
+++ b/models/rekv/model/launtch.sh
@@ -0,0 +1,38 @@
+#!/bin/bash
+set -euo pipefail
+
+REKV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
+PROJECT_ROOT="$(cd "$REKV_ROOT/../.." && pwd)"
+export PYTHONPATH="$REKV_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"
+cd "$PROJECT_ROOT"
+
+python -m model.video_qa.run_eval \
+    --num_chunks 1 \
+    --model llava_ov_7b \
+    --dataset qaego4d \
+    --sample_fps 0.5 \
+    --n_local 15000 \
+    --retrieve_size 64
+
+
+
+
+EXPERIMENT=tome python -m model.video_qa.run_eval \
+    --num_chunks 1 \
+    --model llava_ov_7b \
+    --dataset egoschema \
+    --sample_fps 0.5 \
+    --n_local 15000 \
+    --retrieve_size 64
+
+
+
+python -m model.video_qa.run_eval \
+    --num_chunks 1 \
+    --model llava_ov_7b \
+    --dataset videomme \
+    --sample_fps 0.5 \
+    --n_local 15000 \
+    --retrieve_size 64 \
+    --prune_strategy "vidcom2_orignal" \
+    --token_per_frame 49 
diff --git a/model/llava_onevision_rekv.py b/models/rekv/model/llava_onevision_rekv.py
index 691b24e..7f1ddf0 100644
--- a/model/llava_onevision_rekv.py
+++ b/models/rekv/model/llava_onevision_rekv.py
@@ -1,28 +1,52 @@
 import torch
 from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
 from logzero import logger
-
+import torch.distributed as dist
+
+from stc import (
+    GlobalConfig,
+    STCPruner,
+    default_config,
+    register_stc_cacher,
+    reset_default_cache,
+    stc_patch_vision_enabled,
+)
 from model.patch import patch_hf
 from model.abstract_rekv import Abstract_ReKV
 
 
 class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
+
     def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
         LlavaOnevisionForConditionalGeneration.__init__(self, config)
         Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)
 
+        cfg = default_config()
+        if stc_patch_vision_enabled():
+            register_stc_cacher(self.vision_tower, kind="siglip", config=cfg.cache)
+        reset_default_cache(chunk_idx=0, update_token_ratio=cfg.cache.update_token_ratio)
+
+        self.stc_pruner = STCPruner()
+        self.past_memory_mean_token = self.stc_pruner.past_memory_mean_token
+        
+    def get_vision_tower(self):
+        return self.vision_tower
+
     def get_prompt(self, query, mc=False):
         prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
         if mc:
             prompt += 'Best option: ('
         return prompt
 
+        
+        
     def _get_video_features(self, pixel_values_videos):
         batch_size, frames, channels, height, width = pixel_values_videos.shape
         pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
+        
         video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
         selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]
-
+        frames=selected_video_feature.shape[0]
         if self.config.vision_feature_select_strategy == "default":
             selected_video_feature = selected_video_feature[:, 1:]
         elif self.config.vision_feature_select_strategy == "full":
@@ -30,20 +54,36 @@ class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV)
         video_features = self.multi_modal_projector(selected_video_feature)
 
         video_features = self.apply_pooling(video_features)
-        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  # (B, Nv*196, D)
+        
+        reshaped_video_tensor = video_features.reshape(-1, video_features.size(-1))
+        token_per_frame = default_config().model.token_per_frame
+        video_features = self.stc_pruner.compress(reshaped_video_tensor)
+        if dist.get_rank() == 0:
+            logger.info(f"LLM | Vocab size: 196, Tokens to retained: {token_per_frame}")
+        frames = video_features.shape[0] // token_per_frame
+        
+        video_features = video_features.reshape(batch_size, frames * token_per_frame, -1)
         return video_features
 
+
     @torch.inference_mode()
     def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
+        
         device = self.device
         stop_token_ids = [self.processor.tokenizer.eos_token_id]
 
         output_ids = []
         stopped = False
-
-        # NOTE: Only input the question to perform retrieval.
-        input_ids = self.processor.tokenizer(input_text['question']).input_ids
+        if isinstance(input_text, str):
+            question_text = input_text
+            prompt_text = input_text
+        else:
+            question_text = input_text['question']
+            prompt_text = input_text['prompt']
+            
+        input_ids = self.processor.tokenizer(question_text).input_ids
         input_ids = torch.as_tensor([input_ids], device=device)
+        
         for layer_kv in self.kv_cache:  # activate retrieval mode
             layer_kv.set_retrieval()
 
@@ -62,7 +102,7 @@ class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV)
 
         for i in range(max_new_tokens):
             if i == 0:  # prefill
-                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
+                input_ids = self.processor.tokenizer(prompt_text).input_ids
                 input_ids = torch.as_tensor([input_ids], device=device)
                 inputs_embeds = self.get_input_embeddings()(input_ids)
                 out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
@@ -85,6 +125,10 @@ class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV)
             _, indices = torch.topk(last_token_logits, 2)
             tokens = [int(index) for index in indices.tolist()]
             token = tokens[0]
+            if i == 0 and token in stop_token_ids:   # 第一步就算 eos 也继续
+                token = tokens[1] if len(tokens) > 1 else 1 
+            
+
 
             output_ids.append(token)
 
@@ -95,7 +139,7 @@ class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV)
 
             if i == max_new_tokens - 1 or stopped:
                 break
-
+        
         output = self.processor.tokenizer.decode(
             output_ids,
             skip_special_tokens=True,
@@ -106,10 +150,14 @@ class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV)
         return output
 
 
-def load_model(model_path='model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf',
-               n_init=None, n_local=None, topk=64, chunk_size=1):
-    device = 'cuda'
-    n_frame_tokens = 196
+def load_model(model_path='llava-hf/llava-onevision-qwen2-7b-ov-hf',device=None,
+                       n_init=None, n_local=15000, topk=64, chunk_size=1):
+    GlobalConfig.initialize_from_env()
+    if device is None:
+        device = 'cuda' if torch.cuda.is_available() else 'cpu'
+    token_per_frame = default_config().model.token_per_frame
+    n_frame_tokens =int(token_per_frame)
+    
     processor = LlavaOnevisionProcessor.from_pretrained(model_path)
     
     init_prompt = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
@@ -127,7 +175,7 @@ def load_model(model_path='model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf',
     }
     model = LlavaOneVision_ReKV.from_pretrained(
         model_path, 
-        device_map="auto",
+        device_map={"": device}, # <--- 核心修改：禁止 "auto"，强制指定
         low_cpu_mem_usage=True, 
         torch_dtype=torch.float16,
         processor=processor,
@@ -137,12 +185,16 @@ def load_model(model_path='model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf',
         topk=topk,
         chunk_size=chunk_size,
     )
+
     model.language_model = patch_hf(model.language_model, **inf_llm_config)
     
-    for k, v in inf_llm_config.items():
-        logger.info(f'{k}: {v}')
-    logger.info(f'n_frame_tokens: {n_frame_tokens}')
-
+    ######################################################################
+    rank = dist.get_rank()
+    if rank == 0:
+        for k, v in inf_llm_config.items():
+            logger.info(f'{k}: {v}')
+        logger.info(f'n_frame_tokens: {n_frame_tokens}')
+    ######################################################################
     model.eval()
 
     return model, processor
diff --git a/model/longva/longva/__init__.py b/models/rekv/model/longva/__init__.py
old mode 100755
new mode 100644
similarity index 100%
rename from model/longva/longva/__init__.py
rename to models/rekv/model/longva/__init__.py
diff --git a/model/longva/longva/constants.py b/models/rekv/model/longva/constants.py
similarity index 100%
rename from model/longva/longva/constants.py
rename to models/rekv/model/longva/constants.py
diff --git a/model/longva/longva/conversation.py b/models/rekv/model/longva/conversation.py
similarity index 100%
rename from model/longva/longva/conversation.py
rename to models/rekv/model/longva/conversation.py
diff --git a/models/rekv/model/longva/hfd.sh b/models/rekv/model/longva/hfd.sh
new file mode 100755
index 0000000..8ee2f61
--- /dev/null
+++ b/models/rekv/model/longva/hfd.sh
@@ -0,0 +1,328 @@
+#!/usr/bin/env bash
+# Color definitions
+RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m' # No Color
+
+trap 'printf "${YELLOW}\nDownload interrupted. You can resume by re-running the command.\n${NC}"; exit 1' INT
+
+display_help() {
+    cat << EOF
+Usage:
+  hfd <REPO_ID> [--include include_pattern1 include_pattern2 ...] [--exclude exclude_pattern1 exclude_pattern2 ...] [--hf_username username] [--hf_token token] [--tool aria2c|wget] [-x threads] [-j jobs] [--dataset] [--local-dir path] [--revision rev]
+
+Description:
+  Downloads a model or dataset from Hugging Face using the provided repo ID.
+
+Arguments:
+  REPO_ID         The Hugging Face repo ID (Required)
+                  Format: 'org_name/repo_name' or legacy format (e.g., gpt2)
+Options:
+  include/exclude_pattern The patterns to match against file path, supports wildcard characters.
+                  e.g., '--exclude *.safetensor *.md', '--include vae/*'.
+  --include       (Optional) Patterns to include files for downloading (supports multiple patterns).
+  --exclude       (Optional) Patterns to exclude files from downloading (supports multiple patterns).
+  --hf_username   (Optional) Hugging Face username for authentication (not email).
+  --hf_token      (Optional) Hugging Face token for authentication.
+  --tool          (Optional) Download tool to use: aria2c (default) or wget.
+  -x              (Optional) Number of download threads for aria2c (default: 4).
+  -j              (Optional) Number of concurrent downloads for aria2c (default: 5).
+  --dataset       (Optional) Flag to indicate downloading a dataset.
+  --local-dir     (Optional) Directory path to store the downloaded data.
+                             Defaults to the current directory with a subdirectory named 'repo_name'
+                             if REPO_ID is is composed of 'org_name/repo_name'.
+  --revision      (Optional) Model/Dataset revision to download (default: main).
+
+Example:
+  hfd gpt2
+  hfd bigscience/bloom-560m --exclude *.safetensors
+  hfd meta-llama/Llama-2-7b --hf_username myuser --hf_token mytoken -x 4
+  hfd lavita/medical-qa-shared-task-v1-toy --dataset
+  hfd bartowski/Phi-3.5-mini-instruct-exl2 --revision 5_0
+EOF
+    exit 1
+}
+
+[[ -z "$1" || "$1" =~ ^-h || "$1" =~ ^--help ]] && display_help
+
+REPO_ID=$1
+shift
+
+# Default values
+TOOL="aria2c"
+THREADS=4
+CONCURRENT=5
+HF_ENDPOINT=${HF_ENDPOINT:-"https://huggingface.co"}
+INCLUDE_PATTERNS=()
+EXCLUDE_PATTERNS=()
+REVISION="main"
+
+validate_number() {
+    [[ "$2" =~ ^[1-9][0-9]*$ && "$2" -le "$3" ]] || { printf "${RED}[Error] $1 must be 1-$3${NC}\n"; exit 1; }
+}
+
+# Argument parsing
+while [[ $# -gt 0 ]]; do
+    case $1 in
+        --include) shift; while [[ $# -gt 0 && ! ($1 =~ ^--) && ! ($1 =~ ^-[^-]) ]]; do INCLUDE_PATTERNS+=("$1"); shift; done ;;
+        --exclude) shift; while [[ $# -gt 0 && ! ($1 =~ ^--) && ! ($1 =~ ^-[^-]) ]]; do EXCLUDE_PATTERNS+=("$1"); shift; done ;;
+        --hf_username) HF_USERNAME="$2"; shift 2 ;;
+        --hf_token) HF_TOKEN="$2"; shift 2 ;;
+        --tool)
+            case $2 in
+                aria2c|wget)
+                    TOOL="$2"
+                    ;;
+                *)
+                    printf "%b[Error] Invalid tool. Use 'aria2c' or 'wget'.%b\n" "$RED" "$NC"
+                    exit 1
+                    ;;
+            esac
+            shift 2
+            ;;
+        -x) validate_number "threads (-x)" "$2" 10; THREADS="$2"; shift 2 ;;
+        -j) validate_number "concurrent downloads (-j)" "$2" 10; CONCURRENT="$2"; shift 2 ;;
+        --dataset) DATASET=1; shift ;;
+        --local-dir) LOCAL_DIR="$2"; shift 2 ;;
+        --revision) REVISION="$2"; shift 2 ;;
+        *) display_help ;;
+    esac
+done
+
+# Generate current command string
+generate_command_string() {
+    local cmd_string="REPO_ID=$REPO_ID"
+    cmd_string+=" TOOL=$TOOL"
+    cmd_string+=" INCLUDE_PATTERNS=${INCLUDE_PATTERNS[*]}"
+    cmd_string+=" EXCLUDE_PATTERNS=${EXCLUDE_PATTERNS[*]}"
+    cmd_string+=" DATASET=${DATASET:-0}"
+    cmd_string+=" HF_USERNAME=${HF_USERNAME:-}"
+    cmd_string+=" HF_TOKEN=${HF_TOKEN:-}"
+    cmd_string+=" HF_TOKEN=${HF_ENDPOINT:-}"
+    cmd_string+=" REVISION=$REVISION"
+    echo "$cmd_string"
+}
+
+# Check if aria2, wget, curl are installed
+check_command() {
+    if ! command -v $1 &>/dev/null; then
+        printf "%b%s is not installed. Please install it first.%b\n" "$RED" "$1" "$NC"
+        exit 1
+    fi
+}
+
+check_command curl; check_command "$TOOL"
+
+LOCAL_DIR="${LOCAL_DIR:-${REPO_ID#*/}}"
+mkdir -p "$LOCAL_DIR/.hfd"
+
+if [[ "$DATASET" == 1 ]]; then
+    METADATA_API_PATH="datasets/$REPO_ID"
+    DOWNLOAD_API_PATH="datasets/$REPO_ID"
+    CUT_DIRS=5
+else
+    METADATA_API_PATH="models/$REPO_ID"
+    DOWNLOAD_API_PATH="$REPO_ID"
+    CUT_DIRS=4
+fi
+
+# Modify API URL, construct based on revision
+if [[ "$REVISION" != "main" ]]; then
+    METADATA_API_PATH="$METADATA_API_PATH/revision/$REVISION"
+fi
+API_URL="$HF_ENDPOINT/api/$METADATA_API_PATH"
+
+METADATA_FILE="$LOCAL_DIR/.hfd/repo_metadata.json"
+
+# Fetch and save metadata
+fetch_and_save_metadata() {
+    status_code=$(curl -L -s -w "%{http_code}" -o "$METADATA_FILE" ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "$API_URL")
+    RESPONSE=$(cat "$METADATA_FILE")
+    if [ "$status_code" -eq 200 ]; then
+        printf "%s\n" "$RESPONSE"
+    else
+        printf "%b[Error] Failed to fetch metadata from $API_URL. HTTP status code: $status_code.%b\n$RESPONSE\n" "${RED}" "${NC}" >&2
+        rm $METADATA_FILE
+        exit 1
+    fi
+}
+
+check_authentication() {
+    local response="$1"
+    if command -v jq &>/dev/null; then
+        local gated
+        gated=$(echo "$response" | jq -r '.gated // false')
+        if [[ "$gated" != "false" && ( -z "$HF_TOKEN" || -z "$HF_USERNAME" ) ]]; then
+            printf "${RED}The repository requires authentication, but --hf_username and --hf_token is not passed. Please get token from https://huggingface.co/settings/tokens.\nExiting.\n${NC}"
+            exit 1
+        fi
+    else
+        if echo "$response" | grep -q '"gated":[^f]' && [[ -z "$HF_TOKEN" || -z "$HF_USERNAME" ]]; then
+            printf "${RED}The repository requires authentication, but --hf_username and --hf_token is not passed. Please get token from https://huggingface.co/settings/tokens.\nExiting.\n${NC}"
+            exit 1
+        fi
+    fi
+}
+
+if [[ ! -f "$METADATA_FILE" ]]; then
+    printf "%bFetching repo metadata...%b\n" "$YELLOW" "$NC"
+    RESPONSE=$(fetch_and_save_metadata) || exit 1
+    check_authentication "$RESPONSE"
+else
+    printf "%bUsing cached metadata: $METADATA_FILE%b\n" "$GREEN" "$NC"
+    RESPONSE=$(cat "$METADATA_FILE")
+    check_authentication "$RESPONSE"
+fi
+
+should_regenerate_filelist() {
+    local command_file="$LOCAL_DIR/.hfd/last_download_command"
+    local current_command=$(generate_command_string)
+    
+    # If file list doesn't exist, regenerate
+    if [[ ! -f "$LOCAL_DIR/$fileslist_file" ]]; then
+        echo "$current_command" > "$command_file"
+        return 0
+    fi
+    
+    # If command file doesn't exist, regenerate
+    if [[ ! -f "$command_file" ]]; then
+        echo "$current_command" > "$command_file"
+        return 0
+    fi
+    
+    # Compare current command with saved command
+    local saved_command=$(cat "$command_file")
+    if [[ "$current_command" != "$saved_command" ]]; then
+        echo "$current_command" > "$command_file"
+        return 0
+    fi
+    
+    return 1
+}
+
+fileslist_file=".hfd/${TOOL}_urls.txt"
+
+if should_regenerate_filelist; then
+    # Remove existing file list if it exists
+    [[ -f "$LOCAL_DIR/$fileslist_file" ]] && rm "$LOCAL_DIR/$fileslist_file"
+    
+    printf "%bGenerating file list...%b\n" "$YELLOW" "$NC"
+    
+    # Convert include and exclude patterns to regex
+    INCLUDE_REGEX=""
+    EXCLUDE_REGEX=""
+    if ((${#INCLUDE_PATTERNS[@]})); then
+        INCLUDE_REGEX=$(printf '%s\n' "${INCLUDE_PATTERNS[@]}" | sed 's/\./\\./g; s/\*/.*/g' | paste -sd '|' -)
+    fi
+    if ((${#EXCLUDE_PATTERNS[@]})); then
+        EXCLUDE_REGEX=$(printf '%s\n' "${EXCLUDE_PATTERNS[@]}" | sed 's/\./\\./g; s/\*/.*/g' | paste -sd '|' -)
+    fi
+
+    # Check if jq is available
+    if command -v jq &>/dev/null; then
+        process_with_jq() {
+            if [[ "$TOOL" == "aria2c" ]]; then
+                printf "%s" "$RESPONSE" | jq -r \
+                    --arg endpoint "$HF_ENDPOINT" \
+                    --arg repo_id "$DOWNLOAD_API_PATH" \
+                    --arg token "$HF_TOKEN" \
+                    --arg include_regex "$INCLUDE_REGEX" \
+                    --arg exclude_regex "$EXCLUDE_REGEX" \
+                    --arg revision "$REVISION" \
+                    '
+                    .siblings[]
+                    | select(
+                        .rfilename != null
+                        and ($include_regex == "" or (.rfilename | test($include_regex)))
+                        and ($exclude_regex == "" or (.rfilename | test($exclude_regex) | not))
+                      )
+                    | [
+                        ($endpoint + "/" + $repo_id + "/resolve/" + $revision + "/" + .rfilename),
+                        " dir=" + (.rfilename | split("/")[:-1] | join("/")),
+                        " out=" + (.rfilename | split("/")[-1]),
+                        if $token != "" then " header=Authorization: Bearer " + $token else empty end,
+                        ""
+                      ]
+                    | join("\n")
+                    '
+            else
+                printf "%s" "$RESPONSE" | jq -r \
+                    --arg endpoint "$HF_ENDPOINT" \
+                    --arg repo_id "$DOWNLOAD_API_PATH" \
+                    --arg include_regex "$INCLUDE_REGEX" \
+                    --arg exclude_regex "$EXCLUDE_REGEX" \
+                    --arg revision "$REVISION" \
+                    '
+                    .siblings[]
+                    | select(
+                        .rfilename != null
+                        and ($include_regex == "" or (.rfilename | test($include_regex)))
+                        and ($exclude_regex == "" or (.rfilename | test($exclude_regex) | not))
+                      )
+                    | ($endpoint + "/" + $repo_id + "/resolve/" + $revision + "/" + .rfilename)
+                    '
+            fi
+        }
+        result=$(process_with_jq)
+        printf "%s\n" "$result" > "$LOCAL_DIR/$fileslist_file"
+    else
+        printf "%b[Warning] jq not installed, using grep/awk for metadata json parsing (slower). Consider installing jq for better parsing performance.%b\n" "$YELLOW" "$NC"
+        process_with_grep_awk() {
+            local include_pattern=""
+            local exclude_pattern=""
+            local output=""
+            
+            if ((${#INCLUDE_PATTERNS[@]})); then
+                include_pattern=$(printf '%s\n' "${INCLUDE_PATTERNS[@]}" | sed 's/\./\\./g; s/\*/.*/g' | paste -sd '|' -)
+            fi
+            if ((${#EXCLUDE_PATTERNS[@]})); then
+                exclude_pattern=$(printf '%s\n' "${EXCLUDE_PATTERNS[@]}" | sed 's/\./\\./g; s/\*/.*/g' | paste -sd '|' -)
+            fi
+
+            local files=$(printf '%s' "$RESPONSE" | grep -o '"rfilename":"[^"]*"' | awk -F'"' '{print $4}')
+            
+            if [[ -n "$include_pattern" ]]; then
+                files=$(printf '%s\n' "$files" | grep -E "$include_pattern")
+            fi
+            if [[ -n "$exclude_pattern" ]]; then
+                files=$(printf '%s\n' "$files" | grep -vE "$exclude_pattern")
+            fi
+
+            while IFS= read -r file; do
+                if [[ -n "$file" ]]; then
+                    if [[ "$TOOL" == "aria2c" ]]; then
+                        output+="$HF_ENDPOINT/$DOWNLOAD_API_PATH/resolve/$REVISION/$file"$'\n'
+                        output+=" dir=$(dirname "$file")"$'\n'
+                        output+=" out=$(basename "$file")"$'\n'
+                        [[ -n "$HF_TOKEN" ]] && output+=" header=Authorization: Bearer $HF_TOKEN"$'\n'
+                        output+=$'\n'
+                    else
+                        output+="$HF_ENDPOINT/$DOWNLOAD_API_PATH/resolve/$REVISION/$file"$'\n'
+                    fi
+                fi
+            done <<< "$files"
+
+            printf '%s' "$output"
+        }
+
+        result=$(process_with_grep_awk)
+        printf "%s\n" "$result" > "$LOCAL_DIR/$fileslist_file"
+    fi
+else
+    printf "%bResume from file list: $LOCAL_DIR/$fileslist_file%b\n" "$GREEN" "$NC"
+fi
+
+# Perform download
+printf "${YELLOW}Starting download with $TOOL to $LOCAL_DIR...\n${NC}"
+
+cd "$LOCAL_DIR"
+if [[ "$TOOL" == "aria2c" ]]; then
+    aria2c --console-log-level=error --file-allocation=none -x "$THREADS" -j "$CONCURRENT" -s "$THREADS" -k 1M -c -i "$fileslist_file" --save-session="$fileslist_file"
+elif [[ "$TOOL" == "wget" ]]; then
+    wget -x -nH --cut-dirs="$CUT_DIRS" ${HF_TOKEN:+--header="Authorization: Bearer $HF_TOKEN"} --input-file="$fileslist_file" --continue
+fi
+
+if [[ $? -eq 0 ]]; then
+    printf "${GREEN}Download completed successfully. Repo directory: $PWD\n${NC}"
+else
+    printf "${RED}Download encountered errors.\n${NC}"
+    exit 1
+fi
diff --git a/model/longva/longva/model/multimodal_encoder/clip_encoder.py b/model/longva/longva/model/multimodal_encoder/clip_encoder.py
deleted file mode 100755
index e52a51e..0000000
--- a/model/longva/longva/model/multimodal_encoder/clip_encoder.py
+++ /dev/null
@@ -1,175 +0,0 @@
-import torch
-import torch.nn as nn
-from longva.utils import rank0_print
-from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
-
-try:
-    from s2wrapper import forward as multiscale_forward
-except:
-    pass
-
-
-class CLIPVisionTower(nn.Module):
-    def __init__(self, vision_tower, args, delay_load=False):
-        super().__init__()
-
-        self.is_loaded = False
-
-        self.vision_tower_name = vision_tower
-        self.select_layer = args.mm_vision_select_layer
-        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
-
-        if not delay_load:
-            rank0_print(f"Loading vision tower: {vision_tower}")
-            self.load_model()
-        elif getattr(args, "unfreeze_mm_vision_tower", False):
-            # TODO: better detector is needed.
-            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
-            self.load_model()
-        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
-            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
-            self.load_model()
-        else:
-            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
-
-    def load_model(self, device_map=None):
-        if self.is_loaded:
-            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
-            return
-
-        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
-        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
-        self.vision_tower.requires_grad_(False)
-
-        self.is_loaded = True
-
-    def feature_select(self, image_forward_outs):
-        select_feature_type = self.select_feature
-
-        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
-            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
-            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in range(select_every_k_layer + self.select_layer, len(image_forward_outs.hidden_states), select_every_k_layer)], dim=-1)
-            select_feature_type = select_feature_type.replace("slicefour_", "")
-        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
-            select_layers = [-2, -5, -8, -11, 6]
-            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in select_layers], dim=-1)
-            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
-        else:
-            image_features = image_forward_outs.hidden_states[self.select_layer]
-
-        if select_feature_type == "patch":
-            image_features = image_features[:, 1:]
-        elif select_feature_type == "cls_patch":
-            image_features = image_features
-        else:
-            raise ValueError(f"Unexpected select feature: {select_feature_type}")
-        return image_features
-
-    def forward(self, images):
-        if type(images) is list:
-            image_features = []
-            for image in images:
-                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
-                image_feature = self.feature_select(image_forward_out).to(image.dtype)
-                image_features.append(image_feature)
-        else:
-            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
-            image_features = self.feature_select(image_forward_outs).to(images.dtype)
-
-        return image_features
-
-    @property
-    def dummy_feature(self):
-        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
-
-    @property
-    def dtype(self):
-        return self.vision_tower.dtype
-
-    @property
-    def device(self):
-        return self.vision_tower.device
-
-    @property
-    def config(self):
-        if self.is_loaded:
-            return self.vision_tower.config
-        else:
-            return self.cfg_only
-
-    @property
-    def hidden_size(self):
-        _hidden_size = self.config.hidden_size
-        if "slicefour" in self.select_feature:
-            _hidden_size *= 4
-        if "slice_m25811_f6" in self.select_feature:
-            _hidden_size *= 5
-        return _hidden_size
-
-    @property
-    def num_patches_per_side(self):
-        return self.config.image_size // self.config.patch_size
-
-    @property
-    def num_patches(self):
-        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
-        if "cls_patch" in self.select_feature:
-            _num_patches += 1
-        return _num_patches
-
-    @property
-    def image_size(self):
-        return self.config.image_size
-
-
-class CLIPVisionTowerS2(CLIPVisionTower):
-    def __init__(self, vision_tower, args, delay_load=False):
-
-        self.s2_scales = getattr(args, "s2_scales", "336,672,1008")
-        self.s2_scales = list(map(int, self.s2_scales.split(",")))
-        self.s2_scales.sort()
-        self.s2_split_size = self.s2_scales[0]
-        self.s2_image_size = self.s2_scales[-1]
-
-        super().__init__(vision_tower, args, delay_load)
-
-        # change resize/crop size in preprocessing to the largest image size in s2_scale
-        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
-            self.image_processor.size["shortest_edge"] = self.s2_image_size
-            self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size
-
-    def load_model(self, device_map=None):
-        if self.is_loaded:
-            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
-            return
-
-        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
-        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
-        self.vision_tower.requires_grad_(False)
-
-        self.image_processor.size["shortest_edge"] = self.s2_image_size
-        self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size
-
-        self.is_loaded = True
-
-    @torch.no_grad()
-    def forward_feature(self, images):
-        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
-        image_features = self.feature_select(image_forward_outs).to(images.dtype)
-        return image_features
-
-    @torch.no_grad()
-    def forward(self, images):
-        if type(images) is list:
-            image_features = []
-            for image in images:
-                image_feature = multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)
-                image_features.append(image_feature)
-        else:
-            image_features = multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)
-
-        return image_features
-
-    @property
-    def hidden_size(self):
-        return self.config.hidden_size * len(self.s2_scales)
diff --git a/model/longva/longva/mm_utils.py b/models/rekv/model/longva/mm_utils.py
similarity index 99%
rename from model/longva/longva/mm_utils.py
rename to models/rekv/model/longva/mm_utils.py
index e48b4ff..d177969 100755
--- a/model/longva/longva/mm_utils.py
+++ b/models/rekv/model/longva/mm_utils.py
@@ -6,7 +6,7 @@ import ast
 import re
 import torch
 from transformers import StoppingCriteria
-from longva.constants import IMAGE_TOKEN_INDEX
+from model.longva.constants import IMAGE_TOKEN_INDEX
 
 
 def resize_and_center_crop(image, shortest_edge_length):
diff --git a/model/longva/longva/model/__init__.py b/models/rekv/model/longva/model/__init__.py
similarity index 100%
rename from model/longva/longva/model/__init__.py
rename to models/rekv/model/longva/model/__init__.py
diff --git a/model/longva/longva/model/apply_delta.py b/models/rekv/model/longva/model/apply_delta.py
similarity index 100%
rename from model/longva/longva/model/apply_delta.py
rename to models/rekv/model/longva/model/apply_delta.py
diff --git a/model/longva/longva/model/builder.py b/models/rekv/model/longva/model/builder.py
similarity index 100%
rename from model/longva/longva/model/builder.py
rename to models/rekv/model/longva/model/builder.py
diff --git a/model/longva/longva/model/consolidate.py b/models/rekv/model/longva/model/consolidate.py
similarity index 100%
rename from model/longva/longva/model/consolidate.py
rename to models/rekv/model/longva/model/consolidate.py
diff --git a/model/longva/longva/model/language_model/llava_llama.py b/models/rekv/model/longva/model/language_model/llava_llama.py
similarity index 98%
rename from model/longva/longva/model/language_model/llava_llama.py
rename to models/rekv/model/longva/model/language_model/llava_llama.py
index 9376865..cace271 100755
--- a/model/longva/longva/model/language_model/llava_llama.py
+++ b/models/rekv/model/longva/model/language_model/llava_llama.py
@@ -29,7 +29,7 @@ from transformers import LlamaModel, LlamaForCausalLM
 from transformers.modeling_outputs import CausalLMOutputWithPast
 from transformers.generation.utils import GenerateOutput
 
-from longva.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
+from model.longva.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
 
 
 class LlavaConfig(LlamaConfig):
diff --git a/model/longva/longva/model/language_model/llava_mistral.py b/models/rekv/model/longva/model/language_model/llava_mistral.py
similarity index 100%
rename from model/longva/longva/model/language_model/llava_mistral.py
rename to models/rekv/model/longva/model/language_model/llava_mistral.py
diff --git a/model/longva/longva/model/language_model/llava_mpt.py b/models/rekv/model/longva/model/language_model/llava_mpt.py
similarity index 100%
rename from model/longva/longva/model/language_model/llava_mpt.py
rename to models/rekv/model/longva/model/language_model/llava_mpt.py
diff --git a/model/longva/longva/model/language_model/llava_qwen.py b/models/rekv/model/longva/model/language_model/llava_qwen.py
similarity index 98%
rename from model/longva/longva/model/language_model/llava_qwen.py
rename to models/rekv/model/longva/model/language_model/llava_qwen.py
index e481b18..073a072 100755
--- a/model/longva/longva/model/language_model/llava_qwen.py
+++ b/models/rekv/model/longva/model/language_model/llava_qwen.py
@@ -25,7 +25,7 @@ from transformers.modeling_outputs import CausalLMOutputWithPast
 from transformers.generation.utils import GenerateOutput
 
 # from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
-from longva.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
+from model.longva.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
 from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
 
 # from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
diff --git a/model/longva/longva/model/language_model/modeling_llama.py b/models/rekv/model/longva/model/language_model/modeling_llama.py
similarity index 100%
rename from model/longva/longva/model/language_model/modeling_llama.py
rename to models/rekv/model/longva/model/language_model/modeling_llama.py
diff --git a/model/longva/longva/model/llava_arch.py b/models/rekv/model/longva/model/llava_arch.py
similarity index 98%
rename from model/longva/longva/model/llava_arch.py
rename to models/rekv/model/longva/model/llava_arch.py
index baad49b..580c0cc 100755
--- a/model/longva/longva/model/llava_arch.py
+++ b/models/rekv/model/longva/model/llava_arch.py
@@ -24,10 +24,10 @@ from .multimodal_encoder.builder import build_vision_tower
 from .multimodal_resampler.builder import build_vision_resampler
 from .multimodal_projector.builder import build_vision_projector
 
-from longva.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
+from model.longva.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
 
-from longva.mm_utils import get_anyres_image_grid_shape
-from longva.utils import rank0_print
+from model.longva.mm_utils import get_anyres_image_grid_shape
+from model.longva.utils import rank0_print
 import random
 
 
diff --git a/model/longva/longva/model/make_delta.py b/models/rekv/model/longva/model/make_delta.py
similarity index 100%
rename from model/longva/longva/model/make_delta.py
rename to models/rekv/model/longva/model/make_delta.py
diff --git a/model/longva/longva/model/multimodal_encoder/builder.py b/models/rekv/model/longva/model/multimodal_encoder/builder.py
similarity index 100%
rename from model/longva/longva/model/multimodal_encoder/builder.py
rename to models/rekv/model/longva/model/multimodal_encoder/builder.py
diff --git a/models/rekv/model/longva/model/multimodal_encoder/clip_encoder.py b/models/rekv/model/longva/model/multimodal_encoder/clip_encoder.py
new file mode 100755
index 0000000..02fcdf4
--- /dev/null
+++ b/models/rekv/model/longva/model/multimodal_encoder/clip_encoder.py
@@ -0,0 +1,374 @@
+import torch
+import torch.nn as nn
+from model.longva.utils import rank0_print
+from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
+
+try:
+    from s2wrapper import forward as multiscale_forward
+except:
+    pass
+import os
+from model.longva.model.multimodal_encoder.custom_clip import patch_clip_with_token_cache
+
+
+class CLIPVisionTower(nn.Module):
+    def __init__(self, vision_tower, args, delay_load=False):
+        super().__init__()
+
+        self.is_loaded = False
+
+        self.vision_tower_name = vision_tower
+        self.select_layer = args.mm_vision_select_layer
+        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
+
+        if not delay_load:
+            rank0_print(f"Loading vision tower: {vision_tower}")
+            self.load_model()
+        elif getattr(args, "unfreeze_mm_vision_tower", False):
+            # TODO: better detector is needed.
+            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
+            self.load_model()
+        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
+            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
+            self.load_model()
+        else:
+            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
+
+    # def load_model(self, device_map=None):
+    #     if self.is_loaded:
+    #         rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
+    #         return
+
+    #     self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
+    #     self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
+    #     self.vision_tower.requires_grad_(False)
+
+    #     self.is_loaded = True
+    
+    def load_model(self, device_map=None):
+        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
+        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
+        self.vision_tower.requires_grad_(False)
+        
+        # ✅ 替换 encoder layers 为你的自定义层
+        self.skip_token_ratio = 0.5
+        
+        # 设置必要的环境变量来启用缓存
+        os.environ['CACHE_STRATEGY'] = 'token_level_cache_preln'
+        
+        print(f"Using custom CLIP with skip_token_ratio={self.skip_token_ratio}")
+        print(f"CACHE_STRATEGY={os.getenv('CACHE_STRATEGY')}")
+        try:
+            self.vision_tower = patch_clip_with_token_cache(self.vision_tower, self.skip_token_ratio)
+            print("✅ Successfully patched CLIP with token cache!")
+            
+            # 验证是否正确替换
+            first_layer = self.vision_tower.vision_model.encoder.layers[0]
+            print(f"First layer type: {type(first_layer).__name__}")
+            if hasattr(first_layer, 'set_chunk_index'):
+                print("✅ Custom encoder layers detected!")
+            else:
+                print("❌ Custom encoder layers NOT detected!")
+                
+        except Exception as e:
+            print(f"❌ Failed to patch CLIP: {e}")
+            import traceback
+            traceback.print_exc()
+
+        self.is_loaded = True
+
+    # 暴露缓存控制方法
+    def set_chunk_index(self, idx: int):
+        if hasattr(self.vision_tower, 'set_chunk_index'):
+            self.vision_tower.set_chunk_index(idx)
+        else:
+            # 直接设置每层的chunk_index
+            for layer in self.vision_tower.vision_model.encoder.layers:
+                if hasattr(layer, 'set_chunk_index'):
+                    layer.set_chunk_index(idx)
+
+    def clear_all_cache(self):
+        if hasattr(self.vision_tower, 'clear_all_cache'):
+            self.vision_tower.clear_all_cache()
+        else:
+            # 直接清理每层的缓存
+            for layer in self.vision_tower.vision_model.encoder.layers:
+                if hasattr(layer, 'clear_cache'):
+                    layer.clear_cache()
+
+    def get_all_cache_stats(self):
+        if hasattr(self.vision_tower, 'get_all_cache_stats'):
+            return self.vision_tower.get_all_cache_stats()
+        else:
+            # 直接获取每层的统计
+            stats = {}
+            for i, layer in enumerate(self.vision_tower.vision_model.encoder.layers):
+                if hasattr(layer, 'get_cache_stats'):
+                    stats[f"layer_{i}"] = layer.get_cache_stats()
+            return stats
+    ######################################################################################
+
+    def feature_select(self, image_forward_outs):
+        select_feature_type = self.select_feature
+
+        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
+            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
+            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in range(select_every_k_layer + self.select_layer, len(image_forward_outs.hidden_states), select_every_k_layer)], dim=-1)
+            select_feature_type = select_feature_type.replace("slicefour_", "")
+        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
+            select_layers = [-2, -5, -8, -11, 6]
+            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in select_layers], dim=-1)
+            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
+        else:
+            image_features = image_forward_outs.hidden_states[self.select_layer]
+
+        if select_feature_type == "patch":
+            image_features = image_features[:, 1:]
+        elif select_feature_type == "cls_patch":
+            image_features = image_features
+        else:
+            raise ValueError(f"Unexpected select feature: {select_feature_type}")
+        return image_features
+
+    # def forward(self, images):
+    #     if type(images) is list:
+    #         image_features = []
+    #         for image in images:
+    #             image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
+    #             image_feature = self.feature_select(image_forward_out).to(image.dtype)
+    #             image_features.append(image_feature)
+    #     else:
+    #         image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
+    #         image_features = self.feature_select(image_forward_outs).to(images.dtype)
+
+    #     return image_features
+    
+    @torch.no_grad()
+    def forward(self, images):
+        # 设置当前chunk索引
+
+
+        if os.environ.get('RESET_CLIP_CACHE', '0') == '1':
+            # 清除环境变量，确保只在第一次调用时生效
+            self.current_chunk_idx = 0
+            self.clear_all_cache()
+            os.environ.pop('RESET_CLIP_CACHE', None)
+            
+            
+        if type(images) is list:
+            image_features = []
+            for i, image in enumerate(images):
+                # 为每个图像设置不同的chunk索引
+                self.set_chunk_index(self.current_chunk_idx + i)
+                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
+                image_feature = self.feature_select(image_forward_out).to(image.dtype)
+                image_features.append(image_feature)
+            # 更新chunk计数器
+            self.current_chunk_idx += len(images)
+        else:
+            self.set_chunk_index(self.current_chunk_idx)
+            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
+            image_features = self.feature_select(image_forward_outs).to(images.dtype)
+            # 更新chunk计数器
+            self.current_chunk_idx += 1
+            
+        return image_features
+
+    @property
+    def dummy_feature(self):
+        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
+
+    @property
+    def dtype(self):
+        return self.vision_tower.dtype
+
+    @property
+    def device(self):
+        return self.vision_tower.device
+
+    @property
+    def config(self):
+        if self.is_loaded:
+            return self.vision_tower.config
+        else:
+            return self.cfg_only
+
+    @property
+    def hidden_size(self):
+        _hidden_size = self.config.hidden_size
+        if "slicefour" in self.select_feature:
+            _hidden_size *= 4
+        if "slice_m25811_f6" in self.select_feature:
+            _hidden_size *= 5
+        return _hidden_size
+
+    @property
+    def num_patches_per_side(self):
+        return self.config.image_size // self.config.patch_size
+
+    @property
+    def num_patches(self):
+        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
+        if "cls_patch" in self.select_feature:
+            _num_patches += 1
+        return _num_patches
+
+    @property
+    def image_size(self):
+        return self.config.image_size
+
+
+class CLIPVisionTowerS2(CLIPVisionTower):
+    def __init__(self, vision_tower, args, delay_load=False):
+
+        self.s2_scales = getattr(args, "s2_scales", "336,672,1008")
+        self.s2_scales = list(map(int, self.s2_scales.split(",")))
+        self.s2_scales.sort()
+        self.s2_split_size = self.s2_scales[0]
+        self.s2_image_size = self.s2_scales[-1]
+
+        super().__init__(vision_tower, args, delay_load)
+
+        # change resize/crop size in preprocessing to the largest image size in s2_scale
+        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
+            self.image_processor.size["shortest_edge"] = self.s2_image_size
+            self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size
+
+    # def load_model(self, device_map=None):
+    #     if self.is_loaded:
+    #         rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
+    #         return
+
+    #     self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
+    #     self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
+    #     self.vision_tower.requires_grad_(False)
+
+    #     self.image_processor.size["shortest_edge"] = self.s2_image_size
+    #     self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size
+
+    #     self.is_loaded = True
+    
+    
+    #######################################################################
+    def load_model(self,device_map=None):
+        if self.is_loaded:
+            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
+            return
+
+        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
+        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
+        self.vision_tower.requires_grad_(False)
+
+        self.image_processor.size["shortest_edge"] = self.s2_image_size
+        self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size
+
+        self.is_loaded = True
+        
+        # ✅ 替换 encoder layers 为你的自定义层
+        self.skip_token_ratio = 0.5
+        
+        # 设置必要的环境变量来启用缓存
+        os.environ['CACHE_STRATEGY'] = 'token_level_cache_preln'
+        
+        print(f"Using custom CLIP with skip_token_ratio={self.skip_token_ratio}")
+        print(f"CACHE_STRATEGY={os.getenv('CACHE_STRATEGY')}")
+        try:
+            self.vision_tower = patch_clip_with_token_cache(self.vision_tower, self.skip_token_ratio)
+            print("✅ Successfully patched CLIP with token cache!")
+            
+            # 验证是否正确替换
+            first_layer = self.vision_tower.vision_model.encoder.layers[0]
+            print(f"First layer type: {type(first_layer).__name__}")
+            if hasattr(first_layer, 'set_chunk_index'):
+                print("✅ Custom encoder layers detected!")
+            else:
+                print("❌ Custom encoder layers NOT detected!")
+                
+        except Exception as e:
+            print(f"❌ Failed to patch CLIP: {e}")
+            import traceback
+            traceback.print_exc()
+
+        self.is_loaded = True
+
+    # 暴露缓存控制方法
+    def set_chunk_index(self, idx: int):
+        if hasattr(self.vision_tower, 'set_chunk_index'):
+            self.vision_tower.set_chunk_index(idx)
+        else:
+            # 直接设置每层的chunk_index
+            for layer in self.vision_tower.vision_model.encoder.layers:
+                if hasattr(layer, 'set_chunk_index'):
+                    layer.set_chunk_index(idx)
+
+    def clear_all_cache(self):
+        if hasattr(self.vision_tower, 'clear_all_cache'):
+            self.vision_tower.clear_all_cache()
+        else:
+            # 直接清理每层的缓存
+            for layer in self.vision_tower.vision_model.encoder.layers:
+                if hasattr(layer, 'clear_cache'):
+                    layer.clear_cache()
+
+    def get_all_cache_stats(self):
+        if hasattr(self.vision_tower, 'get_all_cache_stats'):
+            return self.vision_tower.get_all_cache_stats()
+        else:
+            # 直接获取每层的统计
+            stats = {}
+            for i, layer in enumerate(self.vision_tower.vision_model.encoder.layers):
+                if hasattr(layer, 'get_cache_stats'):
+                    stats[f"layer_{i}"] = layer.get_cache_stats()
+            return stats
+
+    @torch.no_grad()
+    def forward_feature(self, images):
+        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
+        image_features = self.feature_select(image_forward_outs).to(images.dtype)
+        return image_features
+
+    # @torch.no_grad()
+    # def forward(self, images):
+    #     if type(images) is list:
+    #         image_features = []
+    #         for image in images:
+    #             image_feature = multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)
+    #             image_features.append(image_feature)
+    #     else:
+    #         image_features = multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)
+
+    #     return image_features
+    @torch.no_grad()
+    def forward(self, images):
+        # 设置当前chunk索引
+
+
+        if os.environ.get('RESET_CLIP_CACHE', '0') == '1':
+            # 清除环境变量，确保只在第一次调用时生效
+            self.current_chunk_idx = 0
+            self.clear_all_cache()
+            os.environ.pop('RESET_CLIP_CACHE', None)
+            
+            
+        if type(images) is list:
+            image_features = []
+            for i, image in enumerate(images):
+                # 为每个图像设置不同的chunk索引
+                self.set_chunk_index(self.current_chunk_idx + i)
+                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
+                image_feature = multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)
+                image_features.append(image_feature)
+            # 更新chunk计数器
+            self.current_chunk_idx += len(images)
+        else:
+            self.set_chunk_index(self.current_chunk_idx)
+            image_features = multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)
+
+            # 更新chunk计数器
+            self.current_chunk_idx += 1
+            
+        return image_features
+
+    @property
+    def hidden_size(self):
+        return self.config.hidden_size * len(self.s2_scales)
diff --git a/models/rekv/model/longva/model/multimodal_encoder/custom_clip.py b/models/rekv/model/longva/model/multimodal_encoder/custom_clip.py
new file mode 100644
index 0000000..dfe2d60
--- /dev/null
+++ b/models/rekv/model/longva/model/multimodal_encoder/custom_clip.py
@@ -0,0 +1,316 @@
+import os
+from typing import Optional, Tuple
+
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
+
+
+from transformers import CLIPVisionModel, CLIPVisionConfig
+
+from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPEncoder, CLIPConfig
+from transformers.modeling_outputs import BaseModelOutput
+
+from typing import Optional, Tuple,Union
+
+
+# ✅ 简单的全局重置标志
+RESET_CACHE_FLAG = False
+
+    
+class LayerRatioAllocator:
+    """
+    简化版层级 skip ratio 分配器。支持:
+    - uniform: 各层同一比例
+    - linear_increasing: 由浅到深线性递增（默认）
+    """
+    def __init__(self, num_layers: int, target_ratio: float = 0.3):
+        self.num_layers = num_layers
+        self.target_ratio = float(target_ratio)
+        self.layer_ratios = self._initialize_layer_ratios()
+
+    def _initialize_layer_ratios(self):
+        strategy = os.getenv("LAYER_RATIO_STRATEGY", "uniform")
+        if strategy == "uniform":
+            return [self.target_ratio] * self.num_layers
+        # linear_increasing
+        ratios = []
+        for i in range(self.num_layers):
+            ratio = self.target_ratio * (0.2 + 1.6 * (i / max(self.num_layers - 1, 1)))
+            ratios.append(ratio)
+        avg = sum(ratios) / len(ratios)
+        if avg > 0:
+            ratios = [r * (self.target_ratio / avg) for r in ratios]
+        return ratios
+
+    def get_layer_ratio(self, layer_idx: int) -> float:
+        if layer_idx >= self.num_layers:
+            return self.target_ratio
+        return float(self.layer_ratios[layer_idx])
+
+
+class TokenLevelCacheCLIPEncoderLayer(CLIPEncoderLayer):
+    """
+    参考 custom_siglip 的结构，为 CLIP 的每层增加：
+    - 偶数 chunk：全量计算，缓存最后一帧 pre-LN2 以及 MLP 输出
+    - 奇数 chunk：LN1+Attention 全量，LN2+MLP 阶段按相似度跳过部分 token，直接复用缓存结果
+    说明：
+    - 这里将 batch 维度视作“帧数” F（对图像也一样工作，只是 F=批大小）
+    - token_per_frame = 序列长度（patch 数+cls），通常含 cls；若你只对 patch 操作，可在上游做裁剪
+    """
+    def __init__(self, config: CLIPVisionConfig):
+        super().__init__(config)
+        self.layer_idx: Optional[int] = None
+        self.ratio_allocator: Optional[LayerRatioAllocator] = None
+
+        # 缓存
+        self.reference_frame_pre_ln2 = None  # [T, C]
+        self.reference_frame_mlp_post = None  # [T, C]
+
+        # 统计
+        self.total_tokens_processed = 0
+        self.total_tokens_skipped = 0
+
+        # 配置
+        self.base_skip_token_ratio = float(os.getenv("SKIP_TOKEN_RATIO", "0.8"))
+        self._current_chunk_idx = 0
+
+    # ------- 基础工具 -------
+    def set_chunk_index(self, chunk_idx: int):
+        self._current_chunk_idx = int(chunk_idx)
+        if self._current_chunk_idx == 0:
+            self.clear_cache()
+
+    def get_chunk_index(self) -> int:
+        return int(getattr(self, "_current_chunk_idx", 0))
+
+    def clear_cache(self):
+        self.reference_frame_pre_ln2 = None
+        self.reference_frame_mlp_post = None
+        self.total_tokens_processed = 0
+        self.total_tokens_skipped = 0
+        if torch.cuda.is_available():
+            torch.cuda.empty_cache()
+
+    def get_cache_stats(self):
+        total = max(self.total_tokens_processed, 1)
+        return {
+            "layer_idx": self.layer_idx,
+            "total_tokens_processed": int(self.total_tokens_processed),
+            "total_tokens_skipped": int(self.total_tokens_skipped),
+            "actual_skip_ratio": float(self.total_tokens_skipped) / total,
+        }
+
+    def get_layer_skip_ratio(self) -> float:
+        if self.ratio_allocator is not None:
+            return float(self.ratio_allocator.get_layer_ratio(self.layer_idx))
+        return float(self.base_skip_token_ratio)
+
+    # ------- 相似度与索引选择 -------
+    @staticmethod
+    def _cosine_sim(hidden_sel: torch.Tensor, ref_sel: torch.Tensor) -> torch.Tensor:
+        # hidden_sel: [F, T, C], ref_sel: [T, C]
+        return F.cosine_similarity(hidden_sel, ref_sel.unsqueeze(0), dim=-1, eps=1e-8)
+
+    def _compute_preln_skip_indices(
+        self, hidden_states_after_attn_residual: torch.Tensor
+    ):
+        """
+        输入 residual2，也就是 Attention 段之后、进入 LN2 前的特征:
+        hidden_states_after_attn_residual: [F, T, C]
+        返回: skip_indices [F, S], compute_indices [F, K]
+        """
+        Fn, T, C = hidden_states_after_attn_residual.shape
+
+        # 没有参考帧则不跳过
+        if self.reference_frame_pre_ln2 is None:
+            all_idx = torch.arange(T, device=hidden_states_after_attn_residual.device)[None, :].expand(Fn, -1)
+            return (
+                torch.empty(Fn, 0, dtype=torch.long, device=hidden_states_after_attn_residual.device),
+                all_idx,
+            )
+
+        with torch.no_grad():
+            ref = self.reference_frame_pre_ln2  # [T, C]
+            sim = self._cosine_sim(hidden_states_after_attn_residual, ref)  # [F, T]
+
+            # 层级 skip ratio
+            use_layer_ratio = os.getenv("LAYER_RATIO_ENABLED", "0").lower() in ("1", "true", "yes")
+            skip_ratio = self.get_layer_skip_ratio() if use_layer_ratio else self.base_skip_token_ratio
+            num_skip = int(max(0, min(T, int(T * skip_ratio))))
+            num_comp = T - num_skip
+
+            if num_skip > 0:
+                skip_indices = torch.topk(sim, k=num_skip, dim=1, largest=True).indices  # [F, S]
+            else:
+                skip_indices = torch.empty(Fn, 0, dtype=torch.long, device=sim.device)
+
+            if num_comp > 0:
+                all_idx = torch.arange(T, device=sim.device)[None, :].expand(Fn, -1)
+                comp_mask = torch.ones_like(all_idx, dtype=torch.bool).scatter(1, skip_indices, False)
+                compute_indices = all_idx[comp_mask].view(Fn, num_comp)  # [F, K]
+            else:
+                compute_indices = torch.empty(Fn, 0, dtype=torch.long, device=sim.device)
+
+        return skip_indices, compute_indices
+
+    # ------- 两种执行路径 -------
+    def _forward_preln_token_cache(
+        self,
+        hidden_states: torch.Tensor,
+        attention_mask: Optional[torch.Tensor],
+        output_attentions: bool,
+    ):
+        """
+        偶数 chunk：全量计算并更新缓存
+        奇数 chunk：LN1+Attn 全量；LN2+MLP 使用缓存按 token 跳过
+        """
+        os.environ['RESET_CLIP_CACHE'] = '0'
+        
+        chunk_idx = self.get_chunk_index() 
+        Fn, T, C = hidden_states.shape
+
+        # LN1
+        residual1 = hidden_states
+        hidden_states_ln1 = self.layer_norm1(hidden_states)
+
+        # Attention 全量
+        attn_out, attn_weights = self.self_attn(
+            hidden_states=hidden_states_ln1,
+            attention_mask=attention_mask,
+            output_attentions=output_attentions,
+        )
+        hidden_states = residual1 + attn_out
+
+        # 进入 LN2/MLP
+        residual2 = hidden_states
+
+        is_even = (chunk_idx % 2 == 0)
+        if is_even:
+            # 全量 LN2 + MLP
+            hs_ln2 = self.layer_norm2(hidden_states)
+            mlp_out_full = self.mlp(hs_ln2)
+            hidden_states = residual2 + mlp_out_full
+
+            # 更新缓存（取最后一“帧”作为参考）
+            with torch.no_grad():
+                self.reference_frame_pre_ln2 = residual2[-1].detach()           # [T, C]
+                self.reference_frame_mlp_post = mlp_out_full[-1].detach()       # [T, C]
+
+            self.total_tokens_processed += Fn * T
+            return (hidden_states, attn_weights) if output_attentions else (hidden_states,)
+        else:
+            # 奇数 chunk：只在 LN2+MLP 阶段跳过
+            skip_indices, compute_indices = self._compute_preln_skip_indices(residual2)
+
+            if self.reference_frame_mlp_post is None or compute_indices.shape[1] == T:
+                # 无缓存或无可跳过：走全量
+                hs_ln2 = self.layer_norm2(hidden_states)
+                mlp_out_full = self.mlp(hs_ln2)
+                hidden_states = residual2 + mlp_out_full
+            else:
+                # 仅计算 compute 的 token，其余用参考帧 MLP 输出
+                out = self.reference_frame_mlp_post.unsqueeze(0).expand(Fn, -1, -1).clone()  # [F, T, C]
+                num_comp = compute_indices.shape[1] if compute_indices.numel() > 0 else 0
+                if num_comp > 0:
+                    idx_comp = compute_indices.unsqueeze(-1).expand(-1, -1, C)             # [F, K, C]
+                    tokens_to_ln2 = hidden_states.gather(1, idx_comp)                      # [F, K, C]
+                    tokens_ln2 = self.layer_norm2(tokens_to_ln2)                           # [F, K, C]
+                    tokens_mlp = self.mlp(tokens_ln2)                                      # [F, K, C]
+                    out.scatter_(1, idx_comp, tokens_mlp)
+                hidden_states = residual2 + out
+
+            # 统计
+            num_skip = skip_indices.shape[1] if skip_indices.numel() > 0 else 0
+            print("num_skip",num_skip)
+            
+            self.total_tokens_processed += Fn * T
+            self.total_tokens_skipped += Fn * num_skip
+
+            return (hidden_states, attn_weights) if output_attentions else (hidden_states,)
+
+    def forward(
+        self,
+        hidden_states: torch.Tensor,
+        attention_mask: Optional[torch.Tensor],
+        causal_attention_mask: torch.Tensor,
+        output_attentions: bool = False,
+    ) -> Tuple[torch.FloatTensor]:
+
+        """
+        默认行为保持与 CLIP 一致；当 CACHE_STRATEGY=token_level_cache_preln 时启用缓存路径
+        """
+        cache_strategy = os.getenv("CACHE_STRATEGY", "none").lower()
+        if cache_strategy == "token_level_cache_preln":
+            return self._forward_preln_token_cache(hidden_states, attention_mask, output_attentions)
+
+        # 原始路径（不缓存）
+        residual = hidden_states
+        hidden_states = self.layer_norm1(hidden_states)
+        hidden_states, attn_weights = self.self_attn(
+            hidden_states=hidden_states,
+            attention_mask=attention_mask,
+            causal_attention_mask=causal_attention_mask,
+            output_attentions=output_attentions,
+        )
+        hidden_states = residual + hidden_states
+        residual = hidden_states
+        hidden_states = self.layer_norm2(hidden_states)
+        hidden_states = self.mlp(hidden_states)
+        hidden_states = residual + hidden_states
+        outputs = (hidden_states,)
+
+        if output_attentions:
+            outputs += (attn_weights,)
+
+        return outputs
+
+def patch_clip_with_token_cache(model, skip_token_ratio=0.3):
+    """
+    将传入的 CLIPVisionModel 的 encoder.layers 替换为 TokenLevelCacheCLIPEncoderLayer，
+    并完成权重拷贝和层级 ratio 分配。
+    """
+    if not isinstance(model, CLIPVisionModel):
+        raise TypeError("patch_clip_with_token_cache expects a transformers.CLIPVisionModel")
+
+    # 取到 encoder 层列表
+    layers = model.vision_model.encoder.layers
+    
+    num_layers = len(layers)
+    allocator = LayerRatioAllocator(num_layers=num_layers, target_ratio=float(skip_token_ratio))
+
+    new_layers = nn.ModuleList()
+    for i, old_layer in enumerate(layers):
+        new_layer = TokenLevelCacheCLIPEncoderLayer(model.config)  # CLIPVisionConfig
+        new_layer.load_state_dict(old_layer.state_dict())
+        new_layer.layer_idx = i
+        new_layer.ratio_allocator = allocator
+        new_layers.append(new_layer)
+
+    # 替换
+    model.vision_model.encoder.layers = new_layers
+
+    # 给 model 添加便捷方法（可选）
+    def _set_chunk_index_all(idx: int):
+        for lyr in model.vision_model.encoder.layers:
+            if hasattr(lyr, "set_chunk_index"):
+                lyr.set_chunk_index(int(idx))
+
+    def _clear_all_cache():
+        for lyr in model.vision_model.encoder.layers:
+            if hasattr(lyr, "clear_cache"):
+                lyr.clear_cache()
+
+    def _get_all_cache_stats():
+        stats = {}
+        for lyr in model.vision_model.encoder.layers:
+            if hasattr(lyr, "get_cache_stats"):
+                stats[f"layer_{lyr.layer_idx}"] = lyr.get_cache_stats()
+        return stats
+
+    model.set_chunk_index = _set_chunk_index_all
+    model.clear_all_cache = _clear_all_cache
+    model.get_all_cache_stats = _get_all_cache_stats
+    
+    return model
+
diff --git a/model/longva/longva/model/multimodal_projector/builder.py b/models/rekv/model/longva/model/multimodal_projector/builder.py
similarity index 100%
rename from model/longva/longva/model/multimodal_projector/builder.py
rename to models/rekv/model/longva/model/multimodal_projector/builder.py
diff --git a/model/longva/longva/model/multimodal_projector/pooler_projector.py b/models/rekv/model/longva/model/multimodal_projector/pooler_projector.py
similarity index 100%
rename from model/longva/longva/model/multimodal_projector/pooler_projector.py
rename to models/rekv/model/longva/model/multimodal_projector/pooler_projector.py
diff --git a/model/longva/longva/model/multimodal_resampler/builder.py b/models/rekv/model/longva/model/multimodal_resampler/builder.py
similarity index 100%
rename from model/longva/longva/model/multimodal_resampler/builder.py
rename to models/rekv/model/longva/model/multimodal_resampler/builder.py
diff --git a/model/longva/longva/model/multimodal_resampler/masked_drop.py b/models/rekv/model/longva/model/multimodal_resampler/masked_drop.py
similarity index 100%
rename from model/longva/longva/model/multimodal_resampler/masked_drop.py
rename to models/rekv/model/longva/model/multimodal_resampler/masked_drop.py
diff --git a/model/longva/longva/model/multimodal_resampler/perceiver.py b/models/rekv/model/longva/model/multimodal_resampler/perceiver.py
similarity index 100%
rename from model/longva/longva/model/multimodal_resampler/perceiver.py
rename to models/rekv/model/longva/model/multimodal_resampler/perceiver.py
diff --git a/model/longva/longva/model/multimodal_resampler/qformer.py b/models/rekv/model/longva/model/multimodal_resampler/qformer.py
similarity index 100%
rename from model/longva/longva/model/multimodal_resampler/qformer.py
rename to models/rekv/model/longva/model/multimodal_resampler/qformer.py
diff --git a/model/longva/longva/model/multimodal_resampler/spatial_pool.py b/models/rekv/model/longva/model/multimodal_resampler/spatial_pool.py
similarity index 100%
rename from model/longva/longva/model/multimodal_resampler/spatial_pool.py
rename to models/rekv/model/longva/model/multimodal_resampler/spatial_pool.py
diff --git a/model/longva/longva/model/utils.py b/models/rekv/model/longva/model/utils.py
similarity index 100%
rename from model/longva/longva/model/utils.py
rename to models/rekv/model/longva/model/utils.py
diff --git a/model/longva/longva/utils.py b/models/rekv/model/longva/utils.py
similarity index 99%
rename from model/longva/longva/utils.py
rename to models/rekv/model/longva/utils.py
index 1135427..2bde528 100755
--- a/model/longva/longva/utils.py
+++ b/models/rekv/model/longva/utils.py
@@ -7,7 +7,7 @@ import numpy as np
 
 import requests
 
-from longva.constants import LOGDIR
+from model.longva.constants import LOGDIR
 
 server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
 moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."
diff --git a/model/longva_rekv.py b/models/rekv/model/longva_rekv.py
index c66e20b..5efb6a3 100644
--- a/model/longva_rekv.py
+++ b/models/rekv/model/longva_rekv.py
@@ -2,11 +2,12 @@ import torch
 from logzero import logger
 
 from transformers import AutoTokenizer
-from longva.model import LlavaQwenForCausalLM
+from model.longva.model import LlavaQwenForCausalLM
 
 from model.patch import patch_hf
 from model.abstract_rekv import Abstract_ReKV
 
+import os
 
 class LongVA_ReKV(LlavaQwenForCausalLM, Abstract_ReKV):
     def __init__(self, config, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
@@ -24,7 +25,14 @@ class LongVA_ReKV(LlavaQwenForCausalLM, Abstract_ReKV):
         video_features = self.get_model().get_vision_tower()(pixel_values_videos)  # (Nv, 576, 1024)
         video_features = self.get_model().mm_projector(video_features)  # (Nv, 576, 3584)
         video_features = self.get_2dPool(video_features)  # (Nv, 144, 3584)
-        video_features = video_features.flatten(0, 1).unsqueeze(0)  # (1, Nv*144, 3584)
+        
+        #######################################################
+        reshaped_video_tensor=video_features.reshape(-1, video_features.size(-1))  
+        token_per_frame = int(os.getenv("TOKEN_PER_FRAME", 144))
+        retention_ratio=float(token_per_frame/144)
+        video_features = video_features.unsqueeze(0)  # (1, Nv*144, 3584)
+            #############################################################
+        # video_features = video_features.flatten(0, 1).unsqueeze(0)  # (1, Nv*144, 3584)
         return video_features
 
     def _encode_video_chunk(self, video_chunk):  # (Nv, H, W, 3)
@@ -110,7 +118,8 @@ class LongVA_ReKV(LlavaQwenForCausalLM, Abstract_ReKV):
 
 def load_model(model_path='model_zoo/LongVA-7B',
                n_init=None, n_local=8000, topk=32, chunk_size=1):
-    n_frame_tokens = 144
+    token_per_frame=int(os.getenv("TOKEN_PER_FRAME", default=144))
+    n_frame_tokens =int(token_per_frame)
     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
     
     init_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
diff --git a/models/rekv/model/online_bench_inference/ovobench/constant.py b/models/rekv/model/online_bench_inference/ovobench/constant.py
new file mode 100644
index 0000000..b9a03ca
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/constant.py
@@ -0,0 +1,60 @@
+BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
+REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
+FORWARD_TASKS = ["REC", "SSR", "CRR"]
+
+# Prompt template for backward-tracing and real-time visual perception task
+BR_PROMPT_TEMPLATE = """
+Question: {}
+Options:
+{}
+
+Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
+Do not include any additional text or explanation in your response.
+"""
+
+# Prompt template for REC task
+# REC_PROMPT_TEMPLATE = """ 
+# You're provided with multiple images which are frames extracted from a video, in which the man/woman are performing an action repetitively.
+
+# Now, answer the following question: Have the person in the video {} {} times?
+
+# Answer only with “Yes” or “No”.
+# Do not include any additional text or explanation in your response.
+# """
+REC_PROMPT_TEMPLATE = """
+You're watching a video in which people may perform a certain type of action repetively. 
+The person performing this kind of action are referred to as 'they' in the following statement.
+You're task is to count how many times have different people in the video perform this kind of action in total.
+One complete motion counts as one. 
+Now, answer the following question: {}
+Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
+Do not include any additional text or explanation in your response.
+"""
+
+# Prompt template for SSR task
+# SSR_PROMPT_TEMPLATE = """
+# You're provided with multiple images which are frames extracted from a tutorial video, in which the whole process may contain multiple different steps.
+
+# Now, answer the following question: Have the person in the video complete {}?
+
+# Answer only with “Yes” or “No”.
+# Do not include any additional text or explanation in your response.
+# """
+SSR_PROMPT_TEMPLATE = """
+You're watching a tutorial video which contain a sequential of steps. 
+The following is one step from the whole procedures: 
+{}
+Your task is to determine if the man or woman in the video is currently performing this step.
+Answer only with “Yes” or “No”.
+Do not include any additional text or explanation in your response.
+"""
+
+# Prompt template for CRR task
+CRR_PROMPT_TEMPLATE = """
+You're responsible of answering questions based on the video content. 
+The following question are relevant to the latest frames, i.e. the end of the video.
+{}
+Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
+Answer only with “Yes” or “No”.
+Do not include any additional text or explanation in your response.
+"""
\ No newline at end of file
diff --git a/models/rekv/model/online_bench_inference/ovobench/inference_distributed.py b/models/rekv/model/online_bench_inference/ovobench/inference_distributed.py
new file mode 100644
index 0000000..07231ec
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/inference_distributed.py
@@ -0,0 +1,481 @@
+"""
+    分布式推理版本 - 使用 torch.distributed.run
+    支持多GPU并行推理，提升OVBench推理速度
+"""
+
+import argparse
+import os
+import json
+import sys
+import warnings
+import torch
+import torch.distributed as dist
+from pathlib import Path
+from tqdm import tqdm
+from logzero import logger
+from stc.config import GlobalConfig
+
+warnings.filterwarnings(
+    "ignore",
+    message=".*do_sample.*is set to.*However.*temperature.*top_p.*",
+    category=UserWarning,
+    module="transformers.generation.configuration_utils"
+)
+
+warnings.filterwarnings(
+    "ignore",
+    message=".*copying from a non-meta parameter.*meta parameter.*no-op.*",
+    category=UserWarning,
+    module="torch.nn.modules.module"
+)
+
+current_dir = os.path.dirname(os.path.abspath(__file__))
+sys.path.append(os.path.join(current_dir, "models"))
+
+
+def main():
+    """主函数"""
+    args = parse_args()
+    GlobalConfig.initialize_from_env()
+    
+    ###############################################################################
+    # 初始化分布式环境
+    assert torch.cuda.is_available(), "DDP推理需要至少一个GPU"
+    torch.backends.cuda.matmul.allow_tf32 = getattr(args, 'tf32', False)
+    torch.set_grad_enabled(False)
+    
+    # Setup DDP - 使用gloo后端支持多进程共享GPU
+    dist.init_process_group("gloo")
+    rank = dist.get_rank()
+    world_size = dist.get_world_size()
+    device = rank % torch.cuda.device_count()
+    
+    # 设置随机种子
+    seed = getattr(args, 'global_seed', 42) * world_size + rank
+    torch.manual_seed(seed)
+    torch.cuda.set_device(device)
+    
+    if rank == 0:
+        logger.info(f"Starting distributed inference with {world_size} processes on {torch.cuda.device_count()} GPUs")
+        logger.info(f"Model: {args.model}; Tasks: {args.task}")
+    
+    dist.barrier()
+    
+    #########################################################################################
+    # 加载annotations
+    with open(args.anno_path, "r") as f:
+        annotations = json.load(f)
+    
+    # 处理视频路径
+    for i, item in enumerate(annotations):
+        annotations[i]["video"] = os.path.join(args.video_dir, item["video"])
+    
+    # 按任务类型分组
+    backward_tasks = ["EPM", "ASI", "HLD"]
+    realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
+    forward_tasks = ["REC", "SSR", "CRR"]
+    
+    backward_anno = []
+    realtime_anno = []
+    forward_anno = []
+    
+    for anno in annotations:
+        if anno["task"] in args.task:
+            if anno["task"] in backward_tasks:
+                backward_anno.append(anno)
+            if anno["task"] in realtime_tasks:
+                realtime_anno.append(anno)
+            if anno["task"] in forward_tasks:
+                forward_anno.append(anno)
+    
+    # 按rank分片数据
+    backward_anno_split = split_data(backward_anno, world_size, rank)
+    realtime_anno_split = split_data(realtime_anno, world_size, rank)
+    forward_anno_split = split_data(forward_anno, world_size, rank)
+    
+    anno = {
+        "backward": backward_anno_split,
+        "realtime": realtime_anno_split,
+        "forward": forward_anno_split
+    }
+    
+    if rank == 0:
+        logger.info(f"Total samples - Backward: {len(backward_anno)}, "
+                   f"Realtime: {len(realtime_anno)}, Forward: {len(forward_anno)}")
+        logger.info(f"[Rank {rank}] Processing - Backward: {len(backward_anno_split)}, "
+                   f"Realtime: {len(realtime_anno_split)}, Forward: {len(forward_anno_split)}")
+    
+    #########################################################################################
+    # 初始化模型（根据device参数传递给模型）
+    model = initialize_model(args, device, rank)
+    
+    ######################################################################
+    # 同步所有进程
+    dist.barrier()
+    
+    # 运行推理
+    results = run_inference(model, anno, args, rank, world_size)
+    
+    # 收集结果
+    if rank == 0:
+        logger.info(f"[Rank {rank}] Gathering results from all ranks...")
+    
+    all_results = gather_results_pipeline(results, rank, world_size)
+    # Rank 0 保存合并后的结果
+    if rank == 0:
+        save_merged_results(all_results, args)
+    
+    dist.destroy_process_group()
+
+
+def initialize_model(args, device, rank):
+    """初始化模型"""
+    # 根据模型类型初始化
+    if args.model == "GPT":
+        from models.GPT import EvalGPT
+        assert args.gpt_api is not None, "GPT API key is required"
+        model = EvalGPT(args)
+        
+    elif args.model == "Gemini":
+        from models.Gemini import EvalGemini
+        assert args.gemini_project is not None, "Gemini project is required"
+        model = EvalGemini(args)
+        
+    elif args.model == "InternVL2":
+        from models.InternVL2 import EvalInternVL2
+        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
+        model = EvalInternVL2(args)
+        
+    elif args.model == "QWen2VL_7B" or args.model == "QWen2VL_72B":
+        from models.QWen2VL import EvalQWen2VL
+        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
+        model = EvalQWen2VL(args)
+        
+    elif args.model == "LongVU":
+        from models.LongVU import EvalLongVU
+        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
+        model = EvalLongVU(args)
+        
+    elif args.model == "LLaVA_OneVision":
+        from models.LLaVA_OneVision import EvalLLaVAOneVision
+        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
+        model = EvalLLaVAOneVision(args)
+        
+    elif args.model == "LLaVA_Video":
+        from models.LLaVA_Video import EvalLLaVAVideo
+        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
+        model = EvalLLaVAVideo(args)
+        
+    elif args.model == "videollm_online":
+        from models.VideoLLM_Online import EvalVideollmOnline
+        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
+        model = EvalVideollmOnline(args)
+        
+    elif args.model == "FlashVStream":
+        from models.FlashVStream import EvalFlashVStream
+        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
+        model = EvalFlashVStream(args)
+        
+    elif args.model == "MiniCPM_o":
+        from models.MiniCPM_o import EvalMiniCPM
+        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
+        model = EvalMiniCPM(args)
+        
+    elif args.model == "Dispider":
+        from models.Dispider import EvalDispider
+        assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
+        model = EvalDispider(args)
+        
+    elif args.model == "rekv":
+        from models.rekv import Evalrekv
+        model = Evalrekv(args)
+        
+    else:
+        raise ValueError(f"Unsupported model: {args.model}. Please implement the model.")
+    
+    if rank == 0:
+        logger.info(f"Model {args.model} initialized on device {device}")
+    
+    return model
+
+
+def split_data(data_list, world_size, rank):
+    """按rank分片数据"""
+    if len(data_list) == 0:
+        return []
+    
+    # 计算每个rank处理的数据范围
+    total_samples = len(data_list)
+    samples_per_rank = (total_samples + world_size - 1) // world_size
+    start_idx = rank * samples_per_rank
+    end_idx = min(start_idx + samples_per_rank, total_samples)
+    
+    return data_list[start_idx:end_idx]
+
+
+def run_inference(model, anno, args, rank, world_size):
+    """运行推理"""
+    backward_results = []
+    realtime_results = []
+    forward_results = []
+    
+    # 处理backward任务
+    if len(anno["backward"]) > 0:
+        desc = f"[Rank {rank}/{world_size}] Backward Tasks"
+        for _anno_ in tqdm(anno["backward"], desc=desc, disable=rank != 0):
+            try:
+                result = process_backward_or_realtime(_anno_, model, args)
+                backward_results.append(result)
+            except Exception as e:
+                logger.error(f"[Rank {rank}] Error processing backward sample {_anno_['id']}: {e}")
+                # 添加失败的结果
+                result = {
+                    "id": _anno_["id"],
+                    "video": _anno_["video"],
+                    "task": _anno_["task"],
+                    "question": _anno_["question"],
+                    "response": None,
+                    "ground_truth": chr(65 + _anno_["gt"]),
+                    "error": str(e)
+                }
+                backward_results.append(result)
+    
+    # 处理realtime任务
+    if len(anno["realtime"]) > 0:
+        desc = f"[Rank {rank}/{world_size}] Realtime Tasks"
+        for _anno_ in tqdm(anno["realtime"], desc=desc, disable=rank != 0):
+            try:
+                result = process_backward_or_realtime(_anno_, model, args)
+                realtime_results.append(result)
+            except Exception as e:
+                logger.error(f"[Rank {rank}] Error processing realtime sample {_anno_['id']}: {e}")
+                result = {
+                    "id": _anno_["id"],
+                    "video": _anno_["video"],
+                    "task": _anno_["task"],
+                    "question": _anno_["question"],
+                    "response": None,
+                    "ground_truth": chr(65 + _anno_["gt"]),
+                    "error": str(e)
+                }
+                realtime_results.append(result)
+    
+    # 处理forward任务
+    if len(anno["forward"]) > 0:
+        desc = f"[Rank {rank}/{world_size}] Forward Tasks"
+        for _anno_ in tqdm(anno["forward"], desc=desc, disable=rank != 0):
+            try:
+                result = process_forward(_anno_, model, args)
+                forward_results.append(result)
+            except Exception as e:
+                logger.error(f"[Rank {rank}] Error processing forward sample {_anno_['id']}: {e}")
+                # 为forward任务添加错误标记
+                _anno_copy = _anno_.copy()
+                for i in range(len(_anno_copy.get("test_info", []))):
+                    _anno_copy["test_info"][i]["response"] = None
+                    _anno_copy["test_info"][i]["error"] = str(e)
+                forward_results.append(_anno_copy)
+    
+    if rank == 0:
+        logger.info(f"[Rank {rank}] Processed - Backward: {len(backward_results)}, "
+                   f"Realtime: {len(realtime_results)}, Forward: {len(forward_results)}")
+    
+    return {
+        "backward": backward_results,
+        "realtime": realtime_results,
+        "forward": forward_results
+    }
+
+
+def process_backward_or_realtime(_anno_, model, args):
+    """处理backward或realtime任务"""
+    id = _anno_["id"]
+    video = _anno_["video"]
+    task = _anno_["task"]
+    question = _anno_["question"]
+    options = _anno_["options"]
+    
+    assert question is not None
+    assert options is not None
+    
+    prompt = model.build_prompt(task=task, question=question, options=options, _anno_=None, index=None)
+    chunk_video_path = os.path.join(args.chunked_dir, f"{id}.mp4")
+    
+    assert os.path.exists(chunk_video_path), f"Video not found: {chunk_video_path}"
+    
+    response = model.inference(chunk_video_path, prompt)
+    
+    result = {
+        "id": id,
+        "video": video,
+        "task": task,
+        "question": question,
+        "response": response,
+        "ground_truth": chr(65 + _anno_["gt"])
+    }
+    
+    return result
+
+
+def process_forward(_anno_, model, args):
+    """处理forward任务"""
+    id = _anno_["id"]
+    task = _anno_["task"]
+    test_info = _anno_["test_info"]
+    
+    _anno_copy = _anno_.copy()
+    
+    for i in range(len(test_info)):
+        prompt = model.build_prompt(task=task, question=None, options=None, _anno_=_anno_, index=i)
+        chunk_video_path = os.path.join(args.chunked_dir, f"{id}_{i}.mp4")
+        
+        assert os.path.exists(chunk_video_path), f"Video not found: {chunk_video_path}"
+        
+        response = model.inference(chunk_video_path, prompt)
+        _anno_copy["test_info"][i]["response"] = response
+    
+    return _anno_copy
+
+def gather_results_pipeline(results, rank, world_size):
+    """
+    Pipeline式all_gather - 最优雅的PyTorch原生方案
+    
+    策略：
+    1. 每个rank统计自己的数据量
+    2. 使用all_gather交换元数据
+    3. 使用send/recv点对点传输大数据
+    4. 避免gather_object的全局同步瓶颈
+    """
+    import pickle
+    
+    # 1. 序列化数据
+    results_bytes = pickle.dumps(results, protocol=pickle.HIGHEST_PROTOCOL)
+    local_size = len(results_bytes)
+    
+    # 2. 交换所有rank的数据大小（小数据，很快）
+    size_tensor = torch.tensor([local_size], dtype=torch.long, device='cpu')
+    all_sizes = [torch.zeros(1, dtype=torch.long, device='cpu') for _ in range(world_size)]
+    
+    dist.all_gather(all_sizes, size_tensor)
+    
+    all_sizes = [s.item() for s in all_sizes]
+    total_size_mb = sum(all_sizes) / (1024**2)
+    
+    if rank == 0:
+        logger.info(f"Total data: {total_size_mb:.2f} MB across {world_size} ranks")
+        max_size_mb = max(all_sizes) / (1024**2)
+        logger.info(f"Max rank size: {max_size_mb:.2f} MB")
+    
+    # 3. Pipeline传输：rank 0逐个从其他rank接收
+    if rank == 0:
+        all_data = []
+        
+        # 首先处理自己的数据
+        all_data.append(results)
+        
+        # 接收其他rank的数据
+        for src_rank in range(1, world_size):
+            # 准备接收buffer
+            recv_size = all_sizes[src_rank]
+            recv_buffer = torch.zeros(recv_size, dtype=torch.uint8, device='cpu')
+            
+            # 接收数据
+            logger.info(f"Receiving from rank {src_rank} ({recv_size / (1024**2):.2f} MB)...")
+            dist.recv(recv_buffer, src=src_rank, tag=src_rank)
+            
+            # 反序列化
+            recv_bytes = recv_buffer.numpy().tobytes()
+            rank_results = pickle.loads(recv_bytes)
+            all_data.append(rank_results)
+            
+            logger.info(f"✓ Received from rank {src_rank} ({src_rank}/{world_size-1})")
+        
+        # 合并所有结果
+        merged = {"backward": [], "realtime": [], "forward": []}
+        
+        for rank_results in all_data:
+            merged["backward"].extend(rank_results["backward"])
+            merged["realtime"].extend(rank_results["realtime"])
+            merged["forward"].extend(rank_results["forward"])
+        
+        logger.info(f"✓ Final results - Backward: {len(merged['backward'])}, "
+                   f"Realtime: {len(merged['realtime'])}, Forward: {len(merged['forward'])}")
+        
+        return merged
+        
+    else:
+        # 其他rank发送数据给rank 0
+        send_buffer = torch.tensor(list(results_bytes), dtype=torch.uint8, device='cpu')
+        
+        logger.info(f"[Rank {rank}] Sending {local_size / (1024**2):.2f} MB to rank 0...")
+        dist.send(send_buffer, dst=0, tag=rank)
+        logger.info(f"[Rank {rank}] Send completed")
+        
+        return None
+
+
+def save_merged_results(results, args):
+    """保存合并后的结果"""
+    if args.save_results:
+        result_dir = Path(args.result_dir) / args.model
+        result_dir.mkdir(parents=True, exist_ok=True)
+        
+        result_file = result_dir / f"{args.model}_{'_'.join(args.task)}_{args.mode}_distributed.json"
+        
+        with open(result_file, "w") as f:
+            json.dump(results, f, indent=4)
+        
+        logger.info(f"Results saved to: {result_file}")
+        logger.info(f"Total samples saved - Backward: {len(results['backward'])}, "
+                   f"Realtime: {len(results['realtime'])}, Forward: {len(results['forward'])}")
+
+
+def parse_args():
+    """解析命令行参数"""
+    parser = argparse.ArgumentParser(description='Run OVBench with Distributed Inference')
+    
+    # 数据路径参数
+    parser.add_argument("--anno_path", type=str, default="data/ovo_bench_new.json", 
+                       help="Path to the annotations")
+    parser.add_argument("--video_dir", type=str, default="data/src_videos", 
+                       help="Root directory of source videos")
+    parser.add_argument("--chunked_dir", type=str, default="data/chunked_videos", 
+                       help="Root directory of chunked videos")
+    parser.add_argument("--result_dir", type=str, default="results", 
+                       help="Root directory of results")
+    
+    # 任务参数
+    parser.add_argument("--mode", type=str, required=True, choices=["online", "offline"], 
+                       help="Online or Offline model for testing")
+    parser.add_argument("--task", type=str, required=False, nargs="+",
+                       choices=["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"],
+                       default=["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"],
+                       help="Tasks to evaluate")
+    
+    # 模型参数
+    parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
+    parser.add_argument("--model_path", type=str, required=False, default=None,
+                       help="Path to the model checkpoint")
+    parser.add_argument("--save_results", type=bool, default=True, 
+                       help="Save results to a file")
+    
+    # API参数（用于GPT和Gemini）
+    parser.add_argument("--gpt_api", type=str, required=False, default=None,
+                       help="GPT API key")
+    parser.add_argument("--gemini_project", type=str, required=False, default=None,
+                       help="Gemini project name")
+    
+    # ReKV相关参数
+    parser.add_argument("--retrieve_size", type=int, required=False, default=64, 
+                       help="Retrieval window size for ReKV and related models")
+    # 分布式参数
+    parser.add_argument("--global_seed", type=int, default=42,
+                       help="Global random seed")
+    parser.add_argument("--tf32", action="store_true", 
+                       help="Enable TF32 acceleration")
+    
+    return parser.parse_args()
+
+
+if __name__ == "__main__":
+    main()
diff --git a/models/rekv/model/online_bench_inference/ovobench/models/Dispider.py b/models/rekv/model/online_bench_inference/ovobench/models/Dispider.py
new file mode 100644
index 0000000..6d7aa22
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/models/Dispider.py
@@ -0,0 +1,281 @@
+"""
+Flash-VStream Eval Code
+
+Weight from: 
+- https://huggingface.co/Mar2Ding/Dispider
+
+Inference Code from:
+- https://github.com/Mark12Ding/Dispider/blob/master/inference.py
+
+Inference Platform:
+- 1*A100 80GB
+"""
+
+import os
+import transformers
+import torch
+import sys
+import argparse
+
+from dispider.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_ANS_TOKEN, DEFAULT_TODO_TOKEN
+from dispider.conversation import conv_templates, SeparatorStyle
+from dispider.model.builder import load_pretrained_model
+from dispider.utils import disable_torch_init
+from dispider.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
+import pdb
+from PIL import Image
+import math
+import pickle
+from decord import VideoReader
+import numpy as np
+
+from transformers import StoppingCriteria, StoppingCriteriaList
+
+from utils.OVOBench import OVOBenchOffline
+class StoppingCriteriaSub(StoppingCriteria):
+    def __init__(self, stops=[], encounters=1):
+        super().__init__()
+        self.stops = stops
+
+    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
+        for stop in self.stops:
+            if torch.all((stop == input_ids[0][-len(stop):])).item():
+                return True
+
+        return False
+
+
+def get_seq_frames(total_num_frames, desired_num_frames):
+    seg_size = float(total_num_frames - 1) / desired_num_frames
+    seq = []
+    for i in range(desired_num_frames):
+        start = int(np.round(seg_size * i))
+        end = int(np.round(seg_size * (i + 1)))
+        seq.append((start + end) // 2)
+
+    return seq
+
+
+def get_seq_time(vr, frame_idx, num_clip):
+    frm_per_clip = len(frame_idx) // num_clip
+    key_frame = [[frame_idx[i*frm_per_clip], frame_idx[i*frm_per_clip+frm_per_clip-1]] for i in range(num_clip)]
+    time = vr.get_frame_timestamp(key_frame)
+    return np.hstack([time[:, 0, 0], time[:, 1, 1]])
+
+
+def calculate_diff(scene_sep, start_frame):
+    diff = [scene_sep[0]-start_frame]
+    for i in range(len(scene_sep)-1):
+        diff.append(scene_sep[i+1]-scene_sep[i])
+    return diff
+
+
+def load_video(vis_path, scene_sep, num_frm=1, max_clip=64, sample_frame=None):
+    block_size = 1
+    vr = VideoReader(vis_path, num_threads=1)
+    total_frame_num = len(vr) if sample_frame is None else (sample_frame[0][1]-sample_frame[0][0])
+    fps = vr.get_avg_fps()
+    total_time = total_frame_num / fps
+
+    if len(scene_sep) == 0:
+        num_clip = total_time / num_frm
+        num_clip = int(block_size*np.round(num_clip/block_size)) if num_clip > block_size else int(np.round(num_clip))
+        num_clip = max(num_clip, 1) ### default 5
+        num_clip = min(num_clip, max_clip)
+        total_num_frm = num_frm * num_clip
+        start_frame = 0 if sample_frame is None else sample_frame[0][0]
+        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
+    else:
+        end_frame = total_frame_num if sample_frame is None else sample_frame[0][1]
+        new_scene_sep = []
+        for ele in scene_sep:
+            sep = int(fps*(ele+1))
+            sep = min(sep, end_frame-1)
+            new_scene_sep.append(sep)
+        new_scene_sep += [end_frame-1]
+        scene_sep = new_scene_sep
+        if len(scene_sep) > max_clip:
+            diff = calculate_diff(scene_sep, start_frame=0)
+            min_idx = np.argsort(diff[:-1])[:len(scene_sep)-max_clip] ##minimum diff to remove
+            for i in np.sort(min_idx)[::-1]:
+                del scene_sep[i]        
+        start_ = 0
+        for end_frame in scene_sep:
+            idx_list = np.linspace(start_, end_frame, num=num_frm, endpoint=False)
+            frame_idx.extend([int(id) for id in idx_list])
+            start_ = end_frame
+
+    time_idx = get_seq_time(vr, frame_idx, num_clip)
+    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)
+
+    a, H, W, _ = img_array.shape
+    if H != W:
+        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
+        img_array = torch.nn.functional.interpolate(img_array, size=(min(H, W), min(H, W)))
+        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
+
+    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
+
+    clip_imgs = []
+    for j in range(total_num_frm):
+        clip_imgs.append(Image.fromarray(img_array[0, j]))
+
+    return clip_imgs, time_idx, num_clip
+
+
+def preprocess_time(time, num_clip, tokenizer):
+    time = time.reshape(2, num_clip)
+    seq = []
+
+    block_size = 1
+    for i in range(num_clip):
+        start, end = time[:, i]
+        start = int(np.round(start))
+        end = int(np.round(end))
+        if (i+1) % block_size == 0:
+            history_end = end
+        sentence = 'This contains a clip sampled in %d to %d seconds' % (start, end) + DEFAULT_IMAGE_TOKEN
+        sentence = tokenizer_image_token(sentence, tokenizer, return_tensors='pt')
+        seq.append(sentence)
+    return seq
+
+
+def preprocess_question(questions, tokenizer):
+    seq = []
+    for q in questions:
+        sentence = tokenizer_image_token(q+DEFAULT_TODO_TOKEN, tokenizer, return_tensors='pt')
+        seq.append(sentence)
+    
+    return seq
+
+
+def process_data(video_id, scene_sep, question, model_config, tokenizer, processor, processor_large, time_tokenizer):
+    num_frames = int(os.getenv("DISPIDER_NUMFRAMES", 16))
+    
+    num_clips = 10000
+    if model_config.mm_use_im_start_end:
+        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
+    else:
+        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
+    conv = conv_templates['qwen'].copy()
+    conv.append_message(conv.roles[0], qs)
+    conv.append_message(conv.roles[1], None)
+    prompt = conv.get_prompt()
+
+    image = video_id
+    presigned_url = image
+    frames, time_idx, num_clips = load_video(presigned_url, scene_sep, num_frames, num_clips)
+    ##############################################################################
+    # video_clips = []
+    # for i in range(num_clips):
+    #     # 提取每个clip的frames
+    #     start_idx = i * num_frames
+    #     end_idx = (i + 1) * num_frames
+    #     clip_frames = frames[start_idx:end_idx]
+        
+    #     # 对当前clip进行预处理
+    #     # processor.set_vision_encoder_chunk_index(i)
+
+    #     clip_tensor = processor.preprocess(clip_frames, return_tensors='pt')['pixel_values']
+    #     video_clips.append(clip_tensor)
+
+    # # 将所有clip连接成一个tensor
+    # video = torch.stack(video_clips, dim=0)[0]
+    ####################################################################################
+    
+    video = processor.preprocess(frames, return_tensors='pt')['pixel_values']
+    
+    video = video.view(num_clips, num_frames, *video.shape[1:])
+    video_large = processor_large.preprocess(frames, return_tensors='pt')['pixel_values']
+    video_large = video_large.view(num_clips, num_frames, *video_large.shape[1:])[:, :1].contiguous()
+    seqs = preprocess_time(time_idx, num_clips, time_tokenizer)
+    seqs = torch.nn.utils.rnn.pad_sequence(
+        seqs, 
+        batch_first=True,
+        padding_value=time_tokenizer.pad_token_id)
+    compress_mask = seqs.ne(time_tokenizer.pad_token_id)
+    question = preprocess_question([question], time_tokenizer)
+    question = torch.nn.utils.rnn.pad_sequence(
+        question, 
+        batch_first=True,
+        padding_value=time_tokenizer.pad_token_id)
+    qs_mask = question.ne(time_tokenizer.pad_token_id)
+
+    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
+
+    return input_ids, video, video_large, seqs, compress_mask, question, qs_mask
+
+
+class EvalDispider(OVOBenchOffline):
+    def __init__(self, args):
+        super().__init__(args)
+
+        self.args = args
+        self._model_init()
+
+    def _model_init(self):
+        model_path = self.args.model_path
+
+        model_path = os.path.expanduser(model_path)
+        model_name = get_model_name_from_path(model_path)
+
+        self.tokenizer, self.model, image_processor, self.context_len = load_pretrained_model(model_path, None, model_name)
+
+        self.image_processor, self.time_tokenizer = image_processor
+        self.image_processor_large = self.image_processor
+        if self.time_tokenizer.pad_token is None:
+            self.time_tokenizer.pad_token = '<pad>'
+
+
+        stop_words_ids = [
+            torch.tensor(self.tokenizer('<|im_end|>').input_ids).cuda(),
+        ]
+
+        self.stopping_criteria = StoppingCriteriaList(
+                [StoppingCriteriaSub(stops=stop_words_ids)])
+    
+    def inference(self, video_file_name, prompt):
+        os.environ['RESET_CLIP_CACHE'] = '1'
+        file = video_file_name
+        prompt = prompt
+                # ✅ 直接调用重置方法
+
+        """
+        Given the video file and input prompt, run the model and return the response
+        file: Video file path
+        inp: Input prompt
+        """
+        input_ids, image_tensor, image_tensor_large, seqs, compress_mask, qs, qs_mask = process_data(file, 
+                                [], 
+                                prompt, 
+                                self.model.config, 
+                                self.tokenizer, 
+                                self.image_processor, 
+                                self.image_processor_large, 
+                                self.time_tokenizer,
+                                )
+        input_ids = input_ids.unsqueeze(0).to(device='cuda', non_blocking=True)
+        with torch.inference_mode():
+            output_ids = self.model.generate(
+                input_ids,
+                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
+                images_large=image_tensor_large.to(dtype=torch.float16, device='cuda', non_blocking=True),
+                seqs=seqs.to(device='cuda', non_blocking=True),
+                compress_mask=compress_mask.to(device='cuda', non_blocking=True),
+                qs=qs.to(device='cuda', non_blocking=True),
+                qs_mask=qs_mask.to(device='cuda', non_blocking=True),
+                ans_token=self.time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors="pt").input_ids.to(device='cuda', non_blocking=True),
+                todo_token=self.time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors="pt").input_ids.to(device='cuda', non_blocking=True),
+                q_id=None,
+                insert_position=0,
+                ans_position=[],
+                do_sample=False,
+                max_new_tokens=1024,
+                pad_token_id=self.tokenizer.eos_token_id,
+                stopping_criteria=self.stopping_criteria,
+                use_cache=True)
+
+        input_token_len = input_ids.shape[1]
+        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
+        outputs = outputs.strip()
+        return outputs
\ No newline at end of file
diff --git a/models/rekv/model/online_bench_inference/ovobench/models/FlashVStream.py b/models/rekv/model/online_bench_inference/ovobench/models/FlashVStream.py
new file mode 100644
index 0000000..64b1861
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/models/FlashVStream.py
@@ -0,0 +1,102 @@
+"""
+Flash-VStream Eval Code
+
+Weight from: 
+- https://huggingface.co/IVGSZ/Flash-VStream-7b
+
+Inference Code from:
+- https://github.com/IVGSZ/Flash-VStream/blob/main/flash_vstream/serve/cli_video_stream.py
+- https://github.com/THUNLP-MT/StreamingBench/blob/main/src/model/FlashVstream.py
+
+Inference Platform:
+- 1*A100 80GB
+"""
+
+import argparse
+import requests
+import logging
+import torch
+import numpy as np
+import time
+import os
+
+from flash_vstream.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
+from flash_vstream.conversation import conv_templates, SeparatorStyle
+from flash_vstream.model.builder import load_pretrained_model
+from flash_vstream.utils import disable_torch_init
+from flash_vstream.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
+
+from torch.multiprocessing import Process, Queue, Manager
+from transformers import TextStreamer
+from decord import VideoReader
+from datetime import datetime
+from PIL import Image
+from io import BytesIO
+
+from utils.OVOBench import OVOBenchOffline
+
+def load_video(video_path):
+    vr = VideoReader(video_path)
+    total_frame_num = len(vr)
+    fps = round(vr.get_avg_fps())
+    frame_idx = [i for i in range(0, len(vr), fps)]
+    spare_frames = vr.get_batch(frame_idx).asnumpy()
+    return spare_frames
+
+class EvalFlashVStream(OVOBenchOffline):
+    def __init__(self, args):
+        super().__init__(args)
+
+        self.args = args
+        self._model_init()
+
+    def _model_init(self):
+        model_name = get_model_name_from_path(self.args.model_path)
+        model_base = None
+        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.args.model_path, model_base, model_name, device="cuda", device_map="auto")
+    
+    def inference(self, video_file_name, prompt):
+        try:
+            video = load_video(video_file_name)
+            video = self.image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
+            video = [video]
+
+            qs = prompt
+            if self.model.config.mm_use_im_start_end:
+                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
+            else:
+                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
+
+            conv = conv_templates["vicuna_v1"].copy()
+            conv.append_message(conv.roles[0], prompt)
+            conv.append_message(conv.roles[1], None)
+            prompt = conv.get_prompt()
+
+            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
+
+            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
+            keywords = [stop_str]
+            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
+
+            with torch.inference_mode():
+                output_ids = self.model.generate(
+                    input_ids,
+                    images=video,
+                    do_sample=True,
+                    temperature=0.002,
+                    max_new_tokens=1024,
+                    use_cache=True,
+                    stopping_criteria=[stopping_criteria])
+                
+            input_token_len = input_ids.shape[1]
+                
+            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
+            outputs = outputs.strip()
+            if outputs.endswith(stop_str):
+                outputs = outputs[:-len(stop_str)]
+            outputs = outputs.strip()
+        except Exception as e:
+            print(e)
+            outputs = None
+
+        return outputs
diff --git a/models/rekv/model/online_bench_inference/ovobench/models/GPT.py b/models/rekv/model/online_bench_inference/ovobench/models/GPT.py
new file mode 100644
index 0000000..8c6c22a
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/models/GPT.py
@@ -0,0 +1,114 @@
+from openai import OpenAI
+import os
+from PIL import Image
+import time
+from utils.OVOBench import OVOBenchOffline
+import base64
+import io
+import numpy as np
+from decord import cpu, VideoReader
+
+base_url = "https://api.openai.com/v1"
+
+class EvalGPT(OVOBenchOffline):
+    def __init__(self, args, model="gpt-4o"):
+        super().__init__(args)
+        self.args = args
+        self.model_name = model
+        self.api_key = args.gpt_api
+        print(self.api_key)
+        
+        self._init_model()
+    
+    def _init_model(self):
+        self.proxy_on()
+        self.client = OpenAI(base_url= base_url, api_key=self.api_key)
+
+    def proxy_on(self):
+        os.environ['http_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
+        os.environ['https_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
+        os.environ['HTTP_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
+        os.environ['HTTPS_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
+        print(os.environ['http_proxy'])
+
+    def load_video(self, video_path, max_frames_num):
+        vr = VideoReader(video_path, ctx=cpu(0))
+        total_frame_num = len(vr)
+        fps = float(vr.get_avg_fps())
+        
+        end_frame = total_frame_num
+        if total_frame_num > max_frames_num:
+            max_frames_num = max_frames_num
+        elif total_frame_num < max_frames_num:
+            max_frames_num = total_frame_num - 2
+        
+        uniform_sampled_frames = np.linspace(0, end_frame - 1, max_frames_num, dtype=int)
+        frame_idx = uniform_sampled_frames.tolist()
+        spare_frames = vr.get_batch(frame_idx)
+        spare_frames = spare_frames.asnumpy()
+        return spare_frames
+    
+    def encode_image(self, image):
+        buffered = io.BytesIO()
+        image.save(buffered, format="PNG")
+        return base64.b64encode(buffered.getvalue()).decode('utf-8')
+
+    def build_messages(self, question, urls):
+        message = []
+        for url in urls:
+            message.append(
+                {
+                    "type": "image_url",
+                    "image_url": {
+                        "url": url,
+                        "detail": "low"
+                    },
+                }
+            )
+        message.append(
+            {
+                "type": "text",
+                "text": question,
+            }
+        )
+
+        prompt =  [
+            {
+                "role": "user",
+                "content": message
+            }
+        ]
+        return prompt
+    
+    def call_gpt_eval(self, message, model_name, retries=10, wait_time=1):
+        for i in range(retries):
+            try:
+                result = self.client.beta.chat.completions.parse(
+                    model=model_name,
+                    messages=message,
+                    max_tokens=128
+                )
+                response_message = result.choices[0].message.content 
+                return response_message
+            except Exception as e:
+                if i < retries - 1:
+                    print(f"Failed to call the API {i+1}/{retries}, will retry after {wait_time} seconds.")
+                    print(e)
+                    time.sleep(wait_time)
+                    continue
+                else:
+                    print(f"Failed to call the API after {retries} attempts.")
+                    print(e)
+                    raise
+    
+    def inference(self, video_file_name, prompt):
+        urls = []
+        frames = self.load_video(video_path=video_file_name, max_frames_num=64)
+        for frame in frames:
+            frame_image = Image.fromarray(frame)
+            base64_image = self.encode_image(frame_image)
+            urls.append(f"data:image/png;base64,{base64_image}")
+        
+        prompt = self.build_messages(prompt, urls)
+        response = self.call_gpt_eval(prompt, self.model_name)
+        return response
diff --git a/models/rekv/model/online_bench_inference/ovobench/models/Gemini.py b/models/rekv/model/online_bench_inference/ovobench/models/Gemini.py
new file mode 100644
index 0000000..09713b2
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/models/Gemini.py
@@ -0,0 +1,57 @@
+import vertexai
+from vertexai.generative_models import GenerativeModel, Part
+import os
+import base64
+from utils.OVOBench import OVOBenchOffline
+
+class EvalGemini(OVOBenchOffline):
+    def __init__(self, args):
+        super().__init__(args)
+        self.args = args
+        self.project = args.gemini_project
+        self._init_model()
+
+    def _init_model(self):
+        self.proxy_on()
+        vertexai.init(project=self.project, location="us-central1")
+        self.vision_model = GenerativeModel(model_name="gemini-1.5-pro")
+
+    def proxy_on(self):
+        os.environ['http_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
+        os.environ['https_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
+        os.environ['HTTP_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
+        os.environ['HTTPS_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
+        print(os.environ['http_proxy'])
+
+    def video_to_base64(self, video_path):
+        # 读取视频文件的二进制数据
+        with open(video_path, 'rb') as video_file:
+            video_data = video_file.read()
+        
+        # 将二进制数据编码为 Base64
+        base64_encoded = base64.b64encode(video_data)
+        
+        # 将 Base64 编码的数据转换为字符串
+        base64_string = base64_encoded.decode('utf-8')
+        
+        return base64_string
+
+    def inference(self, video_file_name, prompt):
+        video_file = self.video_to_base64(video_file_name)
+        
+        try:
+            response = self.vision_model.generate_content(
+                [
+                    Part.from_data(
+                        data=video_file, mime_type="video/mp4"
+                    ),
+                    prompt,
+                ],
+                generation_config={
+                    "temperature": 0
+                }
+            )
+            return response.text
+        except Exception as e:
+            print(e)
+            return None
\ No newline at end of file
diff --git a/models/rekv/model/online_bench_inference/ovobench/models/LLaVA_OneVision.py b/models/rekv/model/online_bench_inference/ovobench/models/LLaVA_OneVision.py
new file mode 100644
index 0000000..1c1cffa
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/models/LLaVA_OneVision.py
@@ -0,0 +1,95 @@
+"""
+LLaVA-OneVision Eval Code
+
+Weight from: 
+- https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov
+
+Inference Code from:
+- https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/playground/demo/video_demo.py
+
+Inference Platform:
+- 4*A100 80GB
+"""
+
+from llava.model.builder import load_pretrained_model
+from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
+from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
+from llava.conversation import conv_templates, SeparatorStyle
+
+import numpy as np
+import copy
+import warnings
+from decord import VideoReader, cpu
+
+warnings.filterwarnings("ignore")
+device = "cuda"
+
+from utils.OVOBench import OVOBenchOffline
+
+def load_video(video_path, max_frames_num,fps=1,force_sample=False):
+    if max_frames_num == 0:
+        return np.zeros((1, 336, 336, 3))
+    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
+    total_frame_num = len(vr)
+    video_time = total_frame_num / vr.get_avg_fps()
+    fps = round(vr.get_avg_fps()/fps)
+    frame_idx = [i for i in range(0, len(vr), fps)]
+    frame_time = [i/fps for i in frame_idx]
+    if len(frame_idx) > max_frames_num or force_sample:
+        sample_fps = max_frames_num
+        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
+        frame_idx = uniform_sampled_frames.tolist()
+        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
+    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
+    spare_frames = vr.get_batch(frame_idx).asnumpy()
+    # import pdb;pdb.set_trace()
+    return spare_frames,frame_time,video_time
+
+class EvalLLaVAOneVision(OVOBenchOffline):
+    def __init__(self, args):
+        super().__init__(args)
+
+        self.args = args
+        self._model_init()
+    
+    def _model_init(self):
+        pretrained = self.args.model_path
+        model_name = "llava_qwen"
+        device = "cuda"
+        device_map = "auto"
+        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
+
+    def inference(self, video_file_name, prompt):
+        image_tensors = []
+        image_sizes = []
+
+        video,frame_time,video_time = load_video(video_file_name, 64, 1, force_sample=True)
+        frames = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
+        image_tensors.append(frames)
+        image_sizes = [frame.size for frame in video]
+        modality = "video"
+
+        # Prepare conversation input
+        conv_template = "qwen_1_5"
+        question = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
+
+        conv = copy.deepcopy(conv_templates[conv_template])
+        conv.append_message(conv.roles[0], question)
+        conv.append_message(conv.roles[1], None)
+        prompt_question = conv.get_prompt()
+
+        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
+
+        # Generate response
+        cont = self.model.generate(
+            input_ids,
+            images=image_tensors,
+            image_sizes=image_sizes,
+            do_sample=False,
+            temperature=0,
+            max_new_tokens=4096,
+            modalities=[modality],
+        )
+        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
+        response = text_outputs[0]
+        return response
diff --git a/models/rekv/model/online_bench_inference/ovobench/models/LLaVA_Video.py b/models/rekv/model/online_bench_inference/ovobench/models/LLaVA_Video.py
new file mode 100644
index 0000000..eef399d
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/models/LLaVA_Video.py
@@ -0,0 +1,89 @@
+"""
+LLaVA Video Eval Code
+
+Weight from: 
+- https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2
+
+Inference Code from:
+- https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_Video_1003.md
+
+Inference Platform:
+- 4*A100 80GB
+"""
+
+from llava.model.builder import load_pretrained_model
+from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
+from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
+from llava.conversation import conv_templates, SeparatorStyle
+from PIL import Image
+import requests
+import copy
+import torch
+import sys
+import warnings
+from decord import VideoReader, cpu
+import numpy as np
+warnings.filterwarnings("ignore")
+
+from utils.OVOBench import OVOBenchOffline
+
+device = "cuda"
+
+def load_video(video_path, max_frames_num,fps=1,force_sample=False):
+    if max_frames_num == 0:
+        return np.zeros((1, 336, 336, 3))
+    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
+    total_frame_num = len(vr)
+    video_time = total_frame_num / vr.get_avg_fps()
+    fps = round(vr.get_avg_fps()/fps)
+    frame_idx = [i for i in range(0, len(vr), fps)]
+    frame_time = [i/fps for i in frame_idx]
+    if len(frame_idx) > max_frames_num or force_sample:
+        sample_fps = max_frames_num
+        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
+        frame_idx = uniform_sampled_frames.tolist()
+        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
+    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
+    spare_frames = vr.get_batch(frame_idx).asnumpy()
+    # import pdb;pdb.set_trace()
+    return spare_frames,frame_time,video_time
+
+class EvalLLaVAVideo(OVOBenchOffline):
+    def __init__(self, args):
+        super().__init__(args)
+
+        self.args = args
+        self._model_init()
+    
+    def _model_init(self):
+        pretrained = self.args.model_path
+        model_name = "llava_qwen"
+        device = "cuda"
+        device_map = "auto"
+        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
+        self.model.eval()
+    
+    def inference(self, video_file_name, prompt):
+        video_path = video_file_name
+        max_frames_num = 64
+        video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
+        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
+        video = [video]
+        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
+        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
+        question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n{prompt}"
+        conv = copy.deepcopy(conv_templates[conv_template])
+        conv.append_message(conv.roles[0], question)
+        conv.append_message(conv.roles[1], None)
+        prompt_question = conv.get_prompt()
+        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
+        cont = self.model.generate(
+            input_ids,
+            images=video,
+            modalities= ["video"],
+            do_sample=False,
+            temperature=0,
+            max_new_tokens=4096,
+        )
+        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
+        return text_outputs
\ No newline at end of file
diff --git a/models/rekv/model/online_bench_inference/ovobench/models/QWen2VL.py b/models/rekv/model/online_bench_inference/ovobench/models/QWen2VL.py
new file mode 100644
index 0000000..1862df6
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/models/QWen2VL.py
@@ -0,0 +1,85 @@
+"""
+Qwen2VL Eval Code
+
+Weight from: 
+- https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
+- https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct
+
+Inference Code from:
+- https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
+
+Inference Platform:
+- 7B: 4*A100 80GB
+- 72B: 8*A100 80GB
+"""
+from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
+from qwen_vl_utils import process_vision_info
+
+from utils.OVOBench import OVOBenchOffline
+from decord import VideoReader
+
+def get_max_frames(video_file_name, max_frames):
+    video = VideoReader(video_file_name)
+    return min(max_frames, len(video) - 2)
+
+class EvalQWen2VL(OVOBenchOffline):
+    def __init__(self, args) -> None:
+        super().__init__(args)
+
+        self.args = args
+        self._model_init()
+
+    def _model_init(self):
+        model_path = self.args.model_path
+        self.model =  Qwen2VLForConditionalGeneration.from_pretrained(
+            model_path, 
+            torch_dtype="auto", 
+            device_map="auto", 
+            attn_implementation="flash_attention_2"
+        )
+
+        self.processor = AutoProcessor.from_pretrained(model_path)
+
+    def inference(self, video_file_name, prompt):
+        frames_num = get_max_frames(video_file_name, max_frames=64)
+        messages = [
+            {
+                "role": "user",
+                "content": [
+                    {
+                        "type": "video",
+                        "video": video_file_name,
+                        "max_pixels": 360 * 420,
+                        "nframes": frames_num,
+                    },
+                    {
+                        "type": "text",
+                        "text": prompt
+                    }
+                ]
+            }
+        ]
+
+        # Preparation for inference
+        text = self.processor.apply_chat_template(
+            messages, tokenize=False, add_generation_prompt=True
+        )
+        image_inputs, video_inputs = process_vision_info(messages)
+        inputs = self.processor(
+            text=[text],
+            images=image_inputs,
+            videos=video_inputs,
+            padding=True,
+            return_tensors="pt",
+        )
+        inputs = inputs.to("cuda")
+
+        # Inference
+        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
+        generated_ids_trimmed = [
+            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
+        ]
+        output_text = self.processor.batch_decode(
+            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
+        )
+        return output_text
\ No newline at end of file
diff --git a/models/rekv/model/online_bench_inference/ovobench/models/VideoLLM_Online.py b/models/rekv/model/online_bench_inference/ovobench/models/VideoLLM_Online.py
new file mode 100644
index 0000000..2d1ecaa
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/models/VideoLLM_Online.py
@@ -0,0 +1,72 @@
+"""
+videollm-online Eval Code
+
+Weight from: 
+- https://huggingface.co/chenjoya/videollm-online-8b-v1plus
+
+Inference Code from:
+- https://github.com/showlab/videollm-online/blob/main/demo/cli.py
+- https://github.com/THUNLP-MT/StreamingBench/blob/main/src/model/VideollmOnline.py
+"""
+
+import os
+import transformers
+import subprocess
+logger = transformers.logging.get_logger('liveinfer')
+from moviepy.editor import VideoFileClip
+from videollm_online.demo.inference import LiveInfer
+
+from utils.OVOBench import OVOBenchOffline
+
+def ffmpeg_once(src_path: str, dst_path: str, *, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
+    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
+    command = [
+        './ffmpeg/ffmpeg',
+        '-y',
+        '-sws_flags', mode,
+        '-i', src_path,
+        '-an',
+        '-threads', '10',
+    ]
+    if fps is not None:
+        command += ['-r', str(fps)]
+    if resolution is not None:
+        command += ['-vf', f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"]
+    command += [dst_path]
+    subprocess.run(command, check=True)
+
+class EvalVideollmOnline(OVOBenchOffline):
+    def __init__(self, args):
+        super().__init__(args)
+
+        self.args = args
+        self._model_init()
+
+    def _model_init(self):
+        self.liveinfer = LiveInfer()
+
+    def inference(self, video_file_name, prompt):
+        file = video_file_name
+        inp = prompt
+        duration = VideoFileClip(video_file_name).duration
+        timestamp = duration
+
+        self.liveinfer.reset()
+        name, ext = os.path.splitext(file)
+        name = name.split('/')[-1]
+        ffmpeg_video_path = os.path.join('./cache', name + f'_{self.liveinfer.frame_fps}fps_{self.liveinfer.frame_resolution}' + ext)
+        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
+        ffmpeg_once(file, ffmpeg_video_path, fps=self.liveinfer.frame_fps, resolution=self.liveinfer.frame_resolution)
+        logger.warning(f'{file} -> {ffmpeg_video_path}, {self.liveinfer.frame_fps} FPS, {self.liveinfer.frame_resolution} Resolution')
+
+        self.liveinfer.load_video(ffmpeg_video_path)
+        self.liveinfer.input_query_stream(inp, video_time=timestamp)
+
+        for i in range(self.liveinfer.num_video_frames):
+            self.liveinfer.input_video_stream(i / self.liveinfer.frame_fps)
+            query, response = self.liveinfer()
+
+            if response:
+                print(response)
+                return response
+        return None
\ No newline at end of file
diff --git a/models/rekv/model/online_bench_inference/ovobench/models/rekv.py b/models/rekv/model/online_bench_inference/ovobench/models/rekv.py
new file mode 100644
index 0000000..6ecdd50
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/models/rekv.py
@@ -0,0 +1,58 @@
+# ===== file: model/rekv.py =====
+from operator import attrgetter
+
+
+import torch
+import os
+import numpy as np
+from PIL import Image
+import requests
+import copy
+import warnings
+from decord import VideoReader, cpu
+from model.video_qa.rekv_offline_refactored import ReKVOfflineVQA
+from model.llava_onevision_rekv import load_model
+
+
+from utils.OVOBench import OVOBenchOffline
+
+
+
+class Evalrekv(ReKVOfflineVQA,OVOBenchOffline):
+
+
+    def __init__(self, args):
+        self.args = args
+        self.sample_fps = 1
+        self.retrieve_size = getattr(args, 'retrieve_size', 64) if getattr(args, 'retrieve_size', None) is not None else 64
+        self.chunk_size = 1
+        self._model_init()
+        #################################
+        self.total_cuda_time =0
+        self.max_mem=0
+        ##################################
+    def _model_init(self):
+        self.qa_model, self.processor = load_model()
+
+
+
+    def inference(self,file, inp):
+        self.qa_model.past_memory_mean_token=[]
+        video = self.load_video(file)
+        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2) 
+        if not isinstance(video, torch.Tensor):
+            video_tensor = torch.from_numpy(video)
+        else:
+            video_tensor = video_tensor
+        self.qa_model.clear_cache()
+        self.qa_model.encode_init_prompt()
+
+        self.qa_model.encode_video(video_tensor)
+
+        response=self.qa_model.question_answering(inp)
+        response_lines = response.strip().splitlines()
+        final_answer = response_lines[-1] if response_lines else ""
+        print("model_final_answer:",final_answer)
+        
+        return final_answer
+
diff --git a/models/rekv/model/online_bench_inference/ovobench/score.py b/models/rekv/model/online_bench_inference/ovobench/score.py
new file mode 100644
index 0000000..376e436
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/score.py
@@ -0,0 +1,37 @@
+"""
+    Calculate scores given inference results JSON
+"""
+
+import argparse
+import os
+import json
+import os
+from utils.OVOBenchScore import OVOBenchOfflineScore, OVOBenchOnlineScore
+
+parser = argparse.ArgumentParser(description='Eval OVBench')
+parser.add_argument("--result_dir", type=str, default="results", help="Root directory of results")
+parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
+parser.add_argument("--mode", type=str, required=True, choices=["online", "offline"], help="Online of Offline model for testing")
+args = parser.parse_args()
+
+# assert os.path.exists(os.path.join(args.result_dir, args.model))
+
+results_paths = os.listdir(os.path.join(args.result_dir, args.model))
+results = {
+    "backward": [],
+    "realtime": [],
+    "forward": []
+}
+for result_path in results_paths:
+    with open(os.path.join(args.result_dir, args.model, result_path), "r") as f:
+        result = json.load(f)
+        results["backward"] += result["backward"]
+        results["realtime"] += result["realtime"]
+        results["forward"] += result["forward"]
+
+if args.model in ["GPT","rekv", "Gemini","Dispider", "InternVL2", "QWen2VL_7B", "QWen2VL_72B", "QWen2VL_7B_", "QWen2VL_72B_", "LongVU", "LLaVA_OneVision", "LLaVA_Video", "videollm_online", "FlashVStream", "MiniCPM_o"]:
+    score_model = OVOBenchOfflineScore(args, results)
+else:
+    raise ValueError(f"Unsupported model: {args.model}. Please implement the model.")
+
+score_model.score()
\ No newline at end of file
diff --git a/models/rekv/model/online_bench_inference/ovobench/utils/OVOBench.py b/models/rekv/model/online_bench_inference/ovobench/utils/OVOBench.py
new file mode 100644
index 0000000..958b5a6
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/utils/OVOBench.py
@@ -0,0 +1,145 @@
+import abc
+from tqdm import tqdm
+import json
+import os
+import sys
+sys.path.append("..")
+from constant import BR_PROMPT_TEMPLATE, REC_PROMPT_TEMPLATE, SSR_PROMPT_TEMPLATE, CRR_PROMPT_TEMPLATE
+
+class OVOBenchOnline():
+    def __init__(self) -> None:
+        pass
+
+    def inference():
+        pass
+
+class OVOBenchOffline():
+    def __init__(self, args):
+        self.args = args
+
+    def eval(self, anno, task_list, mode = "offline"):
+        # Inference
+        if len(anno["backward"]) > 0:
+            backward_results = []
+            for _anno_ in tqdm(anno["backward"], desc="Backward Tasks"):
+                id = _anno_["id"]
+                video = _anno_["video"]
+                task = _anno_["task"]
+                question = _anno_["question"]
+                options = _anno_["options"]
+                realtime = _anno_["realtime"]
+                assert not question == None
+                assert not options == None
+                prompt = self.build_prompt(task = task, question = question, options = options, _anno_ = None, index = None)
+
+                chunk_video_path = os.path.join(self.args.chunked_dir, f"{id}.mp4")
+
+                assert os.path.exists(chunk_video_path)
+                try:
+                    response = self.inference(chunk_video_path, prompt)
+                except Exception as e:
+                    print(f"Error during inference: {e}")
+                    response = None
+
+                result = {
+                    "id": id,
+                    "video": video,
+                    "task": task,
+                    "question": question,
+                    "response": response,
+                    "ground_truth": chr(65 + _anno_["gt"])
+                }
+                backward_results.append(result)
+
+        if len(anno["realtime"]) > 0:
+            realtime_results = []
+            for _anno_ in tqdm(anno["realtime"], desc="Realtime Tasks"):
+                id = _anno_["id"]
+                video = _anno_["video"]
+                task = _anno_["task"]
+                question = _anno_["question"]
+                options = _anno_["options"]
+                realtime = _anno_["realtime"]
+                assert not question == None
+                assert not options == None
+                prompt = self.build_prompt(task = task, question = question, options = options, _anno_ = None, index = None)
+
+                chunk_video_path = os.path.join(self.args.chunked_dir, f"{id}.mp4")
+                assert os.path.exists(chunk_video_path)
+
+                try:
+                    response = self.inference(chunk_video_path, prompt)
+                except Exception as e:
+                    print(f"Error during inference: {e}")
+                    response = None
+
+                result = {
+                    "id": id,
+                    "video": video,
+                    "task": task,
+                    "question": question,
+                    "response": response,
+                    "ground_truth": chr(65 + _anno_["gt"])
+                }
+                realtime_results.append(result)
+
+        if len(anno["forward"]) > 0:
+            forward_results = []
+            for _anno_ in tqdm(anno["forward"], desc="Forward Tasks"):
+                id = _anno_["id"]
+                video = _anno_["video"]
+                task = _anno_["task"]
+                test_info = _anno_["test_info"]
+                for i in range(len(test_info)):
+                    prompt = self.build_prompt(task = task, question = None, options = None, _anno_ = _anno_, index = i)
+                    realtime = test_info[i]["realtime"]
+
+                    chunk_video_path = os.path.join(self.args.chunked_dir, f"{id}_{i}.mp4")
+                    assert os.path.exists(chunk_video_path)
+                    try:
+                        response = self.inference(chunk_video_path, prompt)
+                    except Exception as e:
+                        print(f"Error during inference: {e}")
+                        response = None
+                    
+                    _anno_["test_info"][i]["response"] = response
+                forward_results.append(_anno_)
+        
+        # Calculate Score
+        if len(anno["backward"]) == 0:
+            backward_results = []
+        if len(anno["realtime"]) == 0:
+            realtime_results = []
+        if len(anno["forward"]) == 0:
+            forward_results = []
+
+        # Save Results
+        if self.args.save_results:
+            os.makedirs(f"{self.args.result_dir}/{self.args.model}", exist_ok=True)
+            with open(f"{self.args.result_dir}/{self.args.model}/{self.args.model}_{'_'.join(task_list)}_{mode}_1.json", "w") as f:
+                json.dump({
+                    "backward": backward_results,
+                    "realtime": realtime_results,
+                    "forward": forward_results
+                }, f, indent=4)
+
+    def build_prompt(self, task, question, options, _anno_, index):
+        if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
+            formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
+            prompt = BR_PROMPT_TEMPLATE.format(question, formatted_options)
+            
+        elif task == "REC":
+            activity = _anno_["activity"]
+            question = "How many times did they " + activity + "?"
+            prompt = REC_PROMPT_TEMPLATE.format(question)
+        elif task == "SSR":
+            step = _anno_["test_info"][index]["step"]
+            prompt = SSR_PROMPT_TEMPLATE.format(step)
+        elif task == "CRR":
+            question = _anno_["question"]
+            prompt = CRR_PROMPT_TEMPLATE.format(question)
+        return prompt
+
+    @abc.abstractmethod
+    def inference(self, video_file_name, prompt, start_time=0, end_time=0):
+        pass
\ No newline at end of file
diff --git a/models/rekv/model/online_bench_inference/ovobench/utils/OVOBenchScore.py b/models/rekv/model/online_bench_inference/ovobench/utils/OVOBenchScore.py
new file mode 100644
index 0000000..983ca1f
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/utils/OVOBenchScore.py
@@ -0,0 +1,135 @@
+import os
+class OVOBenchOnlineScore():
+    def __init__(self) -> None:
+        pass
+
+    def eval():
+        pass
+
+class OVOBenchOfflineScore():
+    def __init__(self, args, results):
+        self.args = args
+        self.results = results
+
+    def calculate_score_backward_realtime(self, results):
+        def get_score(response, gt):
+            if response == None:
+                return 0
+            return int(gt in response)
+        # Calculate Score for Every Result
+        for i in range(len(results)):
+            results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])
+        
+        scores = {}
+        for i in range(len(results)):
+            if not results[i]["task"] in scores.keys():
+                scores[results[i]["task"]] = [results[i]["score"]]
+            else:
+                scores[results[i]["task"]].append(results[i]["score"])
+        return results, scores
+
+    def calculate_score_forward(self, results):
+        def get_score_REC(response, gt):
+            if response == None:
+                return 0
+            import re
+            response = re.findall(r'\d+', response)
+            response = "".join(response)
+            return response == str(gt)
+        
+        def get_score_SSR_CRR(response, gt):
+            if response == None:
+                return 0
+            return int(gt in response)
+        
+        scores = {}
+        tasks = list(set([result["task"] for result in results]))
+        for task in tasks:
+            scores[task] = []
+        for i, result in enumerate(results):
+            # Calculate score for REC
+            if result["task"] == "REC":
+                for j, test_info_ in enumerate(result["test_info"]):
+                    scores["REC"].append(get_score_REC(test_info_["response"], test_info_["count"]))
+            # Calculate score for SSR
+            if result["task"] == "SSR":
+                for j, test_info_ in enumerate(result["test_info"]):
+                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
+                        scores["SSR"].append(1)
+                        continue
+                    gt = "No" if test_info_["type"] == 0 else "Yes"
+                    scores["SSR"].append(get_score_SSR_CRR(test_info_["response"], gt))
+            # Calculate score for CRR
+            if result["task"] == "CRR":
+                for j, test_info_ in enumerate(result["test_info"]):
+                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
+                        scores["CRR"].append(1)
+                        continue
+                    gt = "No" if test_info_["type"] == 0 else "Yes"
+                    scores["CRR"].append(get_score_SSR_CRR(test_info_["response"], gt))
+        return results, scores
+    
+    def score(self):
+        print(f"Offline Model: {self.args.model}")
+        backward_results = self.results["backward"]
+        realtime_results = self.results["realtime"]
+        forward_results = self.results["forward"]
+        avg_scores = {
+            "backward": [],
+            "realtime": [],
+            "forward": []
+        }
+
+        if len(backward_results) > 0:
+            print("Evaluate Backward Tracing...")
+            backward_results, backward_scores = self.calculate_score_backward_realtime(backward_results)
+            # correct_backward, total_backward = 0, 0
+            for k, v in backward_scores.items():
+                print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
+                # correct_backward += sum(v)
+                # total_backward += len(v)
+                avg_scores["backward"].append(sum(v)/len(v))
+            # print(f"Backward Avg.: {100 * correct_backward / total_backward:.2f}\n")
+            backward_score = 100 * sum(avg_scores['backward'])/len(avg_scores['backward'])
+            print(f"Backward Avg.: {100 * sum(avg_scores['backward'])/len(avg_scores['backward']):.2f}\n")
+        else:
+            # correct_backward = 0
+            # total_backward = 0
+            pass
+            
+        if len(realtime_results) > 0:
+            print("Evaluate Real-time Visual Perception...")
+            realtime_results, realtime_scores = self.calculate_score_backward_realtime(realtime_results)
+            # correct_realtime, total_realtime = 0, 0
+            for k, v in realtime_scores.items():
+                print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
+                # correct_realtime += sum(v)
+                # total_realtime += len(v)
+                avg_scores["realtime"].append(sum(v)/len(v))
+            # print(f"Realtime Avg.: {100 * correct_realtime / total_realtime:.2f}\n")
+                realtime_score = 100 * sum(avg_scores['realtime'])/len(avg_scores['realtime'])
+            print(f"Realtime Avg.: {100 * sum(avg_scores['realtime'])/len(avg_scores['realtime']):.2f}\n")
+        else:
+            # correct_realtime = 0
+            # total_realtime = 0
+            pass
+
+        if len(forward_results) > 0:
+            print("Evaluate Forward Active Responding...")
+            forward_results, forward_scores = self.calculate_score_forward(forward_results)
+            # correct_forward, total_forward = 0, 0
+            for k, v in forward_scores.items():
+                print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
+                # correct_forward += sum(v)
+                # total_forward += len(v)
+                avg_scores["forward"].append(sum(v)/len(v))
+            # print(f"Forward Avg.: {100 * correct_forward / total_forward:.2f}\n")
+                forward_score = 100 * sum(avg_scores['forward'])/len(avg_scores['forward'])
+            print(f"Forward Avg.: {100 * sum(avg_scores['forward'])/len(avg_scores['forward']):.2f}\n")
+        else:
+            # correct_forward = 0
+            # total_forward = 0
+            pass
+
+        print(f"Total Avg.: {(backward_score + realtime_score + forward_score) / 3:.2f}")
+
diff --git a/models/rekv/model/online_bench_inference/ovobench/utils/chunk_videos.py b/models/rekv/model/online_bench_inference/ovobench/utils/chunk_videos.py
new file mode 100644
index 0000000..afbded6
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/utils/chunk_videos.py
@@ -0,0 +1,62 @@
+import argparse
+import os
+import json
+from moviepy.editor import VideoFileClip
+import sys
+sys.path.append("..")
+import math
+from tqdm import tqdm
+
+BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
+REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
+FORWARD_TASKS = ["REC", "SSR", "CRR"]
+
+parser = argparse.ArgumentParser(description="Chunk_Video")
+parser.add_argument("--anno_path", type=str, default="data/ovo_bench_new.json", help="Path to the annotations")
+parser.add_argument("--video_dir", type=str, default="data/src_videos", help="Root directory of source videos")
+parser.add_argument("--output_dir", type=str, default="data/chunked_videos", help="Root directory to save the chunked videos")
+
+args = parser.parse_args()
+os.makedirs(args.output_dir, exist_ok=True)
+
+with open(args.anno_path, "r") as file:
+    data = json.load(file)
+
+for i in tqdm(range(len(data))):
+    if not (data[i]["task"] in FORWARD_TASKS):
+        continue
+    if data[i]["task"] in BACKWARD_TASKS or data[i]["task"] in REAL_TIME_TASKS:
+        output_path = os.path.join(args.output_dir, f"{data[i]['id']}.mp4")
+        end_time = math.ceil(data[i]["realtime"])
+        if os.path.exists(output_path):
+            print(f"Chunked video path {output_path} exists. Pass.")
+
+        if True:
+            video = VideoFileClip(os.path.join(args.video_dir, data[i]["video"]))
+            video_duration = video.duration
+            if end_time > video_duration:
+                end_time = video_duration
+            clip = video.subclip(0, end_time)
+            clip.write_videofile(output_path)
+
+            video.close()
+    elif data[i]["task"] in FORWARD_TASKS:
+        for j in range(len(data[i]["test_info"])):
+            output_path = os.path.join(args.output_dir, f"{data[i]['id']}_{j}.mp4")
+            end_time = math.ceil(data[i]["test_info"][j]["realtime"])
+            
+            if os.path.exists(output_path):
+                print(f"Chunked video path {output_path} exists. Pass.")
+
+            if True:
+                video = VideoFileClip(os.path.join(args.video_dir, data[i]["video"]))
+                video_duration = video.duration
+                if end_time > video_duration:
+                    end_time = video_duration
+                clip = video.subclip(0, end_time)
+                clip.write_videofile(output_path)
+
+                video.close()
+    
+    
+
diff --git a/models/rekv/model/online_bench_inference/ovobench/utils/sample_frames.py b/models/rekv/model/online_bench_inference/ovobench/utils/sample_frames.py
new file mode 100644
index 0000000..85e269c
--- /dev/null
+++ b/models/rekv/model/online_bench_inference/ovobench/utils/sample_frames.py
@@ -0,0 +1,63 @@
+from decord import VideoReader, cpu
+import argparse
+import os
+import numpy as np
+from tqdm import tqdm
+import json
+from PIL import Image
+import sys
+sys.path.append("..")
+
+BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
+REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
+FORWARD_TASKS = ["REC", "SSR", "CRR"]
+
+parser = argparse.ArgumentParser(description='Run OVBench')
+parser.add_argument("--anno_path", type=str, default="data/ovo_bench.json", help="Path to the annotations")
+parser.add_argument("--video_dir", type=str, default="data/src_videos", help="Root directory of source videos")
+parser.add_argument("--chunked_dir", type=str, default="data/chunked_videos", help="Root directory of chunked videos")
+parser.add_argument("--sampled_frames_dir", type=str, default="data/sampled_frames", help="Root dir to save sampled frames")
+args = parser.parse_args()
+
+def load_video(video_path, max_frames_num=64):
+    vr = VideoReader(video_path, ctx=cpu(0))
+    total_frame_num = len(vr)
+        
+    end_frame = total_frame_num
+    if total_frame_num > max_frames_num:
+        max_frames_num = max_frames_num
+    elif total_frame_num < max_frames_num:
+        max_frames_num = total_frame_num - 2
+        
+    uniform_sampled_frames = np.linspace(0, end_frame - 1, max_frames_num, dtype=int)
+    frame_idx = uniform_sampled_frames.tolist()
+    spare_frames = vr.get_batch(frame_idx)
+    spare_frames = spare_frames.asnumpy()
+
+    return spare_frames
+
+with open(args.anno_path, "r") as file:
+    data = json.load(file)
+
+for i in tqdm(range(len(data))):
+    if data[i]["task"] in BACKWARD_TASKS or data[i]["task"] in REAL_TIME_TASKS:
+        chunked_video_path = os.path.join(args.chunked_dir, f"{data[i]['id']}.mp4")
+        output_dir = os.path.join(args.sampled_frames_dir, f"{data[i]['id']}")
+
+    elif data[i]["task"] in FORWARD_TASKS:
+        for j in range(len(data[i]["test_info"])):
+            chunked_video_path = os.path.join(args.chunked_dir, f"{data[i]['id']}.mp4")
+            output_dir = os.path.join(args.sampled_frames_dir, f"{data[i]['id']}_{j}")
+    
+    os.makedirs(output_dir, exist_ok=True)
+    assert os.path.exists(chunked_video_path)
+
+    spare_frames = load_video(chunked_video_path)
+    for j in range(len(spare_frames)):
+        save_path = os.path.join(output_dir, f"{j}.jpg")
+        if os.path.exists(save_path):
+            print(f"Sampled frames path {save_path} exists. Pass.")
+        else:
+            # Save sampled frames to path
+            img = Image.fromarray(spare_frames[j])
+            img.save(save_path)
\ No newline at end of file
diff --git a/model/patch.py b/models/rekv/model/patch.py
index 9131d44..f7e9c04 100644
--- a/model/patch.py
+++ b/models/rekv/model/patch.py
@@ -150,9 +150,12 @@ def patch_hf(
 
     hf_rope = model.model.layers[0].self_attn.rotary_emb 
     if isinstance(hf_rope, Qwen2RotaryEmbedding):
-        base = hf_rope.base
+        base = getattr(hf_rope, "base", getattr(hf_rope.config, "rope_theta", 10000.0))
         distance_scale = 1.0
-        dim = hf_rope.dim
+        dim = getattr(hf_rope, "dim", None)
+        if dim is None:
+            partial_rotary_factor = getattr(hf_rope.config, "partial_rotary_factor", 1.0)
+            dim = int((hf_rope.config.hidden_size // hf_rope.config.num_attention_heads) * partial_rotary_factor)
     else:
         base = hf_rope.config.rope_theta
         distance_scale = distance_scale if distance_scale is not None else 1.0
diff --git a/models/rekv/model/video_qa/README.md b/models/rekv/model/video_qa/README.md
new file mode 100644
index 0000000..ac4d57e
--- /dev/null
+++ b/models/rekv/model/video_qa/README.md
@@ -0,0 +1,270 @@
+# Video QA 模块说明
+
+## 📝 概述
+
+这是一个统一的视频问答推理模块，支持多种数据集和推理模式。
+
+## 🏗️ 架构设计
+
+### 核心组件
+
+```
+video_qa/
+├── base_refactored.py          # 基类：所有solver的通用逻辑
+├── rekv_offline_refactored.py  # 离线推理：标准视频问答
+├── videomme_refactored.py      # VideoMME专用：带性能统计
+├── rekv_stream_refactored.py  # 流式推理：增量编码
+├── solver_factory.py           # Solver工厂：根据配置创建实例
+├── configs.py                  # 数据集配置：统一管理
+└── run_distributed.py          # 分布式推理：多卡并行
+```
+
+### 设计模式
+
+1. **工厂模式** (`solver_factory.py`)
+   - 根据数据集配置自动选择正确的solver
+   - 解耦配置和实现
+
+2. **模板方法** (`base_refactored.py`)
+   - 定义通用流程
+   - 子类实现特定逻辑
+
+3. **策略模式** (三种solver)
+   - 不同数据集使用不同策略
+   - 灵活扩展
+
+## 🚀 使用方法
+
+### 1. 配置数据集
+
+在 `configs.py` 中添加或修改数据集配置：
+
+```python
+DATASETS = {
+    'my_dataset': DatasetConfig(
+        name='my_dataset',
+        anno_path='data/my_dataset/test.json',
+        solver='rekv_offline_vqa',  # 选择solver类型
+        eval_script='models/rekv/model/video_qa/eval/eval_my_dataset.py'
+    ),
+}
+```
+
+### 2. 选择Solver类型
+
+支持三种solver：
+
+| Solver | 用途 | 特性 |
+|--------|------|------|
+| `rekv_offline_vqa` | 标准视频问答 | 支持多选题和开放式问答 |
+| `videomme_rekv_offline_vqa` | VideoMME数据集 | 带GPU时间/内存统计 |
+| `rekv_stream_vqa` | 流式视频问答 | 增量编码，支持时间窗口 |
+
+### 3. 运行推理
+
+#### 单卡推理
+
+```bash
+python -m model.video_qa.run_distributed \
+    --dataset egoschema \
+    --save_dir results/egoschema \
+    --model llava_ov_7b
+```
+
+#### 多卡推理
+
+```bash
+torchrun --nproc_per_node=4 \
+    -m model.video_qa.run_distributed \
+    --dataset videomme \
+    --save_dir results/videomme \
+    --model llava_ov_7b \
+    --retrieve_size 64
+```
+
+## 📊 Solver详细说明
+
+### ReKVOfflineVQA (标准离线推理)
+
+**适用数据集**: EgoSchema, MLVU, CG-Bench, ActivityNet-QA
+
+**核心功能**:
+- 编码整个视频到KV缓存
+- 支持多选题和开放式问答
+- 自动提取选项字母
+
+**数据格式**:
+```json
+{
+  "video_id": "xxx",
+  "video_path": "path/to/video.mp4",
+  "conversations": [
+    {
+      "question": "What happened?",
+      "answer": "Something",
+      "choices": ["A", "B", "C", "D"]  // 可选
+    }
+  ]
+}
+```
+
+### VideoMMEReKVOfflineVQA (VideoMME专用)
+
+**适用数据集**: Video-MME, Video-MME Subset
+
+**特殊功能**:
+- ✅ GPU编码时间统计
+- ✅ 显存峰值监控
+- ✅ 累积时间追踪
+- ✅ 支持duration字段
+
+**特殊数据格式**:
+```json
+{
+  "video_id": "xxx",
+  "duration": 120.5,  // 视频时长
+  "conversations": [
+    {
+      "question": "What is shown?",
+      "answer": "A",  // 直接是选项字母，不是文本
+      "choices": ["A", "B", "C", "D"]
+    }
+  ]
+}
+```
+
+**输出字段**:
+```python
+{
+    'video_id': 'xxx',
+    'question': '...',
+    'pred_answer': '...',
+    'pred_choice': 'A',
+    'qa_acc': 100.0,
+    'duration': 120.5  # 额外的duration字段
+}
+```
+
+### ReKVStreamVQA (流式推理)
+
+**适用数据集**: RVS-Ego, RVS-Movie
+
+**核心特性**:
+- 增量编码视频帧
+- 支持时间窗口查询
+- 内存效率高
+
+**数据格式**:
+```json
+{
+  "video_id": "xxx",
+  "video_path": "path/to/video.npy",
+  "conversations": [
+    {
+      "question": "What happened?",
+      "answer": "Something",
+      "start_time": 10.0,  // 时间窗口开始
+      "end_time": 20.0     // 时间窗口结束
+    }
+  ]
+}
+```
+
+## 🔧 扩展指南
+
+### 添加新的Solver
+
+1. **创建新的solver类**:
+
+```python
+# my_custom_solver.py
+from .rekv_offline_refactored import ReKVOfflineVQA
+
+class MyCustomVQA(ReKVOfflineVQA):
+    """自定义solver"""
+    
+    def answer_single(self, qa_pair, video_id):
+        # 实现你的逻辑
+        pass
+```
+
+2. **注册到工厂**:
+
+```python
+# solver_factory.py
+SOLVER_MAP = {
+    'rekv_offline_vqa': ReKVOfflineVQA,
+    'videomme_rekv_offline_vqa': VideoMMEReKVOfflineVQA,
+    'rekv_stream_vqa': ReKVStreamVQA,
+    'my_custom_vqa': MyCustomVQA,  # 添加这行
+}
+```
+
+3. **配置数据集**:
+
+```python
+# configs.py
+DATASETS = {
+    'my_dataset': DatasetConfig(
+        name='my_dataset',
+        anno_path='...',
+        solver='my_custom_vqa',  # 使用新solver
+        eval_script='...'
+    ),
+}
+```
+
+## 📝 最佳实践
+
+### 1. 保持函数简洁
+- 每个函数 < 15行
+- 单一职责原则
+- 清晰的命名
+
+### 2. 使用统一的接口
+- 所有solver继承自`BaseVQA`
+- 实现`answer_single()`方法
+- 返回标准化的字典
+
+### 3. 配置驱动
+- 所有数据集配置在`configs.py`
+- 通过solver名称选择实现
+- 避免硬编码
+
+## 🐛 常见问题
+
+### Q: 如何添加新的数据集？
+
+A: 在 `configs.py` 中添加配置，选择合适的solver即可。
+
+### Q: solver选择错误怎么办？
+
+A: `solver_factory.py` 会自动fallback到`rekv_offline_vqa`，并记录warning日志。
+
+### Q: 如何自定义输出字段？
+
+A: 重写 `_format_mc_result()` 或 `_format_open_result()` 方法。
+
+### Q: 多选题的正确答案如何处理？
+
+A: 
+- 标准数据集：answer是文本，自动匹配choices得到字母
+- VideoMME：answer直接是字母（A/B/C/D）
+
+## 📊 性能优化
+
+### 内存优化
+- 使用流式推理处理长视频
+- 设置合适的`retrieve_size`
+- 控制`chunk_size`
+
+### 速度优化
+- 使用多卡并行（`torchrun`）
+- 启用TF32加速（`--tf32`）
+- 调整`sample_fps`降低帧数
+
+## 📚 相关文档
+
+- [分布式推理详解](../../docs/distributed.md)
+- [数据集准备](../../data/README.md)
+- [模型配置](../config.py)
diff --git a/model/video_qa/base.py b/model/video_qa/base.py
deleted file mode 100644
index 608b7f9..0000000
--- a/model/video_qa/base.py
+++ /dev/null
@@ -1,231 +0,0 @@
-import warnings
-import random
-import json
-import os
-import math
-import argparse
-
-import pandas as pd
-import torch
-from tqdm import tqdm
-from decord import VideoReader, cpu
-from transformers import (
-    logging,
-    LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor,
-    VideoLlavaForConditionalGeneration, VideoLlavaProcessor
-)
-import logzero
-from logzero import logger
-
-from model import llava_onevision_rekv, video_llava_rekv, longva_rekv
-
-
-MODELS = {
-    'llava_ov_0.5b': {
-        'load_func': llava_onevision_rekv.load_model,
-        'model_class': LlavaOnevisionForConditionalGeneration,
-        'processor_class': LlavaOnevisionProcessor,
-        'model_path': 'model_zoo/llava-onevision-qwen2-0.5b-ov-hf',
-    },
-    'llava_ov_7b': {
-        'load_func': llava_onevision_rekv.load_model,
-        'model_class': LlavaOnevisionForConditionalGeneration,
-        'processor_class': LlavaOnevisionProcessor,
-        'model_path': 'model_zoo/llava-onevision-qwen2-7b-ov-hf',
-    },
-    'llava_ov_72b': {
-        'load_func': llava_onevision_rekv.load_model,
-        'model_class': LlavaOnevisionForConditionalGeneration,
-        'processor_class': LlavaOnevisionProcessor,
-        'model_path': 'model_zoo/llava-onevision-qwen2-72b-ov-hf',
-    },
-    'video_llava_7b': {
-        'load_func': video_llava_rekv.load_model,
-        'model_class': VideoLlavaForConditionalGeneration,
-        'processor_class': VideoLlavaProcessor,
-        'model_path': 'model_zoo/Video-LLaVA-7B-hf',
-    },
-    'longva_7b': {
-        'load_func': longva_rekv.load_model,
-        'model_path': 'model_zoo/LongVA-7B',
-    },
-}
-
-
-class BaseVQA:
-    def __init__(self, anno, save_dir, sample_fps,
-                 qa_model, qa_processor=None,
-                 num_chunks=None, chunk_idx=None,
-                 retrieve_size=64, chunk_size=1) -> None:
-        
-        self.sample_fps = sample_fps
-
-        self.qa_model = qa_model
-        self.qa_processor = qa_processor
-
-        # Retrieval Hyperparams
-        assert chunk_size <= retrieve_size, f'chunk_size: {chunk_size}, retrieve_size: {retrieve_size}'
-        self.retrieve_size = retrieve_size
-        self.chunk_size = chunk_size
-
-        self.num_chunks = num_chunks
-        self.chunk_idx = chunk_idx
-        if num_chunks is not None:
-            anno = self.get_chunk(anno, num_chunks, chunk_idx)
-        self.anno = anno
-        self.eval_grounding = 'temporal_windows' in anno[0]['conversations'][0]
-
-        self.save_dir = save_dir
-        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
-        self.record = {(self.retrieve_size, self.chunk_size): []}
-
-    def split_list(self, lst, n):
-        """Split a list into n (roughly) equal-sized chunks"""
-        chunk_size = math.ceil(len(lst) / n)  # integer division
-        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
-
-    def get_chunk(self, lst, n, k):
-        chunks = self.split_list(lst, n)
-        return chunks[k]
-
-    def load_video(self, video_path):
-        vr = VideoReader(video_path, ctx=cpu(0))
-        fps = round(vr.get_avg_fps())
-        frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
-        video = vr.get_batch(frame_idx).asnumpy()
-        logger.debug(f'video shape: {video.shape}')
-        return video
-    
-    def calc_recall_precision(self, gt_temporal_windows, retrieved_mask):
-        total_intersection_length = 0.0
-    
-        for (start_sec, end_sec) in gt_temporal_windows:
-            start = math.floor(start_sec)
-            end = math.ceil(end_sec)
-            for i in range(start, end):
-                if i < len(retrieved_mask) and retrieved_mask[i]:
-                    intersection_start = max(start_sec, i)
-                    intersection_end = min(end_sec, i + 1)
-                    total_intersection_length += intersection_end - intersection_start
-
-        gt_len = sum([end_sec - start_sec for start_sec, end_sec in gt_temporal_windows])
-        retrieved_len = sum(retrieved_mask).item()
-
-        recall = total_intersection_length / gt_len if gt_len > 0 else 0
-        precision = total_intersection_length / retrieved_len if retrieved_len > 0 else 0
-        if precision + recall > 0:
-            f1 = 2 * (precision * recall) / (precision + recall)
-        else:
-            f1 = 0
-        return recall, precision, f1
-    
-    def format_mcqa_prompt(self, question, candidates):
-        assert len(question) > 0, f"Q: {question}"
-
-        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
-        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."
-
-        return {
-            "question": f"{question}",
-            "formatted_question": formatted_question,
-            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
-        }
-
-    def extract_characters_regex(self, s):
-        s = s.strip()
-        if ")" in s:
-            index = s.index(")")
-            pred = s[index - 1 : index]
-            return pred
-        else:
-            return s[0]
-
-    def video_open_qa(self, question, max_new_tokens=1024):
-        pass
-
-    def video_close_qa(self, question, candidates, correct_choice):
-        pass
-
-    @torch.inference_mode()
-    def analyze_a_video(self, video_sample):
-        pass
-
-    def analyze(self, debug=False):
-        video_annos = self.anno[:1] if debug else self.anno
-        for video_sample in tqdm(video_annos):
-            logger.debug(f'video_id: {video_sample["video_id"]}')
-            self.analyze_a_video(video_sample)
-
-        dfs = []
-        for (retrieve_size, chunk_size), dict_list in self.record.items():
-            df = pd.DataFrame(dict_list)
-            df['retrieve_size'] = retrieve_size
-            df['chunk_size'] = chunk_size
-            dfs.append(df)
-        final_df = pd.concat(dfs, ignore_index=True)
-        final_df.to_csv(f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv', index=False)
-
-
-def str2bool(value):
-    if isinstance(value, bool):
-        return value
-    if value.lower() in ('true', '1', 'yes'):
-        return True
-    elif value.lower() in ('false', '0', 'no'):
-        return False
-    else:
-        raise argparse.ArgumentTypeError('Boolean value expected.')
-
-def work(QA_CLASS):
-    logging.set_verbosity_error()
-
-    parser = argparse.ArgumentParser()
-    parser.add_argument("--sample_fps", type=float, default=1)
-    parser.add_argument("--num_chunks", type=int, default=1)
-    parser.add_argument("--chunk_idx", type=int, default=0)
-    parser.add_argument("--save_dir", type=str, required=True)
-    parser.add_argument("--anno_path", type=str, required=True)
-    parser.add_argument("--model", type=str, default="llava_ov_7b")
-    parser.add_argument("--n_local", type=int, default=15000)
-    parser.add_argument("--retrieve_size", type=int, default=64)
-    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
-    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
-    args = parser.parse_args()
-
-    if not args.debug:
-        logzero.loglevel(logging.INFO)
-        warnings.filterwarnings('ignore')
-
-    os.makedirs(args.save_dir, exist_ok=True)
-
-    # fix random seed
-    random.seed(2024)
-    logger.info('seed: 2024')
-
-    # VideoQA model
-    model_path = MODELS[args.model]['model_path']
-    load_func = MODELS[args.model]['load_func']
-    logger.info(f"Loading VideoQA model: {model_path}")
-    videoqa_model, videoqa_processor = load_func(
-        model_path=model_path,
-        n_local=args.n_local,
-        topk=args.retrieve_size,
-        chunk_size=args.retrieve_chunk_size,
-    )
-
-    # Load ground truth file
-    anno = json.load(open(args.anno_path))
-
-    retrieve_analyzer = QA_CLASS(
-        anno=anno,
-        sample_fps=args.sample_fps,
-        qa_model=videoqa_model,
-        qa_processor=videoqa_processor,
-        retrieve_size=args.retrieve_size,
-        chunk_size=args.retrieve_chunk_size,
-        num_chunks=args.num_chunks,
-        chunk_idx=args.chunk_idx,
-        save_dir=args.save_dir,
-    )
-
-    retrieve_analyzer.analyze(debug=args.debug)
diff --git a/models/rekv/model/video_qa/base_refactored.py b/models/rekv/model/video_qa/base_refactored.py
new file mode 100644
index 0000000..17c20ee
--- /dev/null
+++ b/models/rekv/model/video_qa/base_refactored.py
@@ -0,0 +1,102 @@
+"""重构的BaseVQA - 简洁优雅的视频问答基类"""
+import re
+import torch
+from logzero import logger
+from .utils.data_utils import chunk_video
+
+import torch.distributed as dist
+from decord import VideoReader, cpu
+
+class BaseVQA:
+    """视频问答基类 - 所有函数<15行"""
+    
+    choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
+    
+    def __init__(self, model, processor, args):
+        self.model = model
+        self.processor = processor
+        self.args = args
+        self.results = []
+    
+    def __call__(self, video_sample):
+        """前向传播 - 处理一个视频样本"""
+        video = self.load_video(video_sample['video_path'], self.args.sample_fps)
+        video_tensor = self._to_tensor(video)
+        self.encode_video(video_tensor)
+        return self.answer_questions(video_sample)
+    def load_video(self,video_path, sample_fps=1):
+        vr = VideoReader(video_path, ctx=cpu(0))
+        fps = round(vr.get_avg_fps())
+        frame_idx = [i for i in range(0, len(vr), int(fps / sample_fps))]
+        video = vr.get_batch(frame_idx).asnumpy()
+        logger.debug(f'Loaded video: {video.shape}')
+        return video
+    def _to_tensor(self, video):
+        """转换为tensor"""
+        if isinstance(video, torch.Tensor):
+            return video
+        return torch.from_numpy(video)
+    
+    def encode_video(self, video):
+        """编码视频为KV缓存"""
+        self.model.clear_cache()
+        self.model.encode_init_prompt()
+        self.model.encode_video(video)
+        
+        ########################################
+        rank = dist.get_rank()
+        if rank == 0:
+            logger.debug(f'Video encoded, cache size: {self._get_cache_size():.1f} GB')
+        ########################################
+    
+    def _get_cache_size(self):
+        """获取缓存大小（GB）"""
+        return self.model.calc_memory_usage() / (1024**3)
+    
+    def answer_questions(self, video_sample):
+        """批量回答问题"""
+        results = []
+        for qa in video_sample['conversations']:
+            result = self.answer_single(qa, video_sample['video_id'])
+            results.append(result)
+            self.results.append(result)
+        return results
+    
+    def answer_single(self, qa_pair, video_id):
+        """回答单个问题 - 子类实现"""
+        raise NotImplementedError
+    
+    def format_mcqa_prompt(self, question, choices):
+        """格式化多选题提示"""
+        formatted_choices = "\n".join([
+            f"({self.choice_letters[i]}) {choice}" 
+            for i, choice in enumerate(choices)
+        ])
+        formatted_q = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."
+        return self.model.get_prompt(formatted_q, mc=True)
+    
+    def format_openqa_prompt(self, question):
+        """格式化开放式问题提示"""
+        return self.model.get_prompt(question)
+    
+    def extract_choice(self, pred_text):
+        """从预测文本提取选项"""
+        pred_text = pred_text.strip()
+        if ")" in pred_text:
+            idx = pred_text.index(")")
+            return pred_text[idx - 1:idx]
+        return pred_text[0] if pred_text else 'A'
+    
+    def save_results(self, save_path):
+        """保存结果到CSV"""
+        import pandas as pd
+        
+        from pathlib import Path
+
+        # In your save_results method
+        save_dir = Path('results/eval')
+        df = pd.DataFrame(self.results)
+        
+        save_dir.mkdir(parents=True, exist_ok=True)
+        df.to_csv(save_path, index=False)
+        logger.info(f"Saved {len(self.results)} results to {save_path}")
\ No newline at end of file
diff --git a/models/rekv/model/video_qa/configs.py b/models/rekv/model/video_qa/configs.py
new file mode 100644
index 0000000..cca33c9
--- /dev/null
+++ b/models/rekv/model/video_qa/configs.py
@@ -0,0 +1,80 @@
+"""数据集配置 - 统一管理所有数据集参数"""
+from dataclasses import dataclass
+
+@dataclass
+class DatasetConfig:
+    """数据集配置类"""
+    name: str
+    anno_path: str
+    solver: str
+    eval_script: str
+
+# 所有支持的数据集配置
+DATASETS = {
+    'smoke': DatasetConfig(
+        name='smoke',
+        anno_path='benchmarks/offline/smoke/smoke_rekv.json',
+        solver='rekv_offline_vqa',
+        eval_script='models/rekv/model/video_qa/eval/eval_smoke.py'
+    ),
+    'videomme': DatasetConfig(
+        name='videomme',
+        anno_path='benchmarks/offline/videomme/random_videomme.json',
+        solver='videomme_rekv_offline_vqa',
+        eval_script='models/rekv/model/video_qa/eval/videomme_rekv_offline_vqa.py'
+    ),
+    'videomme_subset': DatasetConfig(
+        name='videomme_subset',
+        anno_path='benchmarks/offline/videomme/videomme_subset.json',
+        solver='videomme_rekv_offline_vqa',
+        eval_script='models/rekv/model/video_qa/eval/eval_videomme.py'
+    ),
+    'mlvu': DatasetConfig(
+        name='mlvu',
+        anno_path='benchmarks/offline/mlvu/dev_debug_mc.json',
+        solver='rekv_offline_vqa',
+        eval_script='models/rekv/model/video_qa/eval/eval_multiple_choice.py'
+    ),
+    'egoschema': DatasetConfig(
+        name='egoschema',
+        anno_path='benchmarks/offline/egoschema/full.json',
+        solver='rekv_offline_vqa',
+        eval_script='models/rekv/model/video_qa/eval/eval_egoschema.py'
+    ),
+    'egoschema_subset': DatasetConfig(
+        name='egoschema_subset',
+        anno_path='benchmarks/offline/egoschema_subset/egoschema_subset.json',
+        solver='videomme_rekv_offline_vqa',
+        eval_script='models/rekv/model/video_qa/eval/eval_egoschema_subset.py'
+    ),
+    'qaego4d': DatasetConfig(
+        name='qaego4d',
+        anno_path='benchmarks/offline/qaego4d/test_mc.json',
+        solver='rekv_offline_vqa',
+        eval_script='models/rekv/model/video_qa/eval/eval_multiple_choice.py'
+    ),
+    'cgbench': DatasetConfig(
+        name='cgbench',
+        anno_path='benchmarks/offline/cgbench/full_mc.json',
+        solver='rekv_offline_vqa',
+        eval_script='models/rekv/model/video_qa/eval/eval_multiple_choice.py'
+    ),
+    'activitynet_qa': DatasetConfig(
+        name='activitynet_qa',
+        anno_path='benchmarks/offline/activitynet_qa/test.json',
+        solver='rekv_offline_vqa',
+        eval_script='models/rekv/model/video_qa/eval/eval_open_ended.py'
+    ),
+    'rvs_ego': DatasetConfig(
+        name='rvs_ego',
+        anno_path='benchmarks/offline/rvs/ego/ego4d_oe.json',
+        solver='rekv_stream_vqa',
+        eval_script='models/rekv/model/video_qa/eval/eval_open_ended.py'
+    ),
+    'rvs_movie': DatasetConfig(
+        name='rvs_movie',
+        anno_path='benchmarks/offline/rvs/movie/movienet_oe.json',
+        solver='rekv_stream_vqa',
+        eval_script='models/rekv/model/video_qa/eval/eval_open_ended.py'
+    ),
+}
diff --git a/models/rekv/model/video_qa/eval/eval_egoschema_subset.py b/models/rekv/model/video_qa/eval/eval_egoschema_subset.py
new file mode 100644
index 0000000..a473330
--- /dev/null
+++ b/models/rekv/model/video_qa/eval/eval_egoschema_subset.py
@@ -0,0 +1,679 @@
+#!/usr/bin/env python3
+"""
+视频问答(Video QA)评测脚本
+用于分析模型输出结果并生成详细的评测报告
+
+使用方法：
+    python evaluate_results.py --result_file results.csv
+    python evaluate_results.py --result_dir results/batch_20251010_155403
+    
+"""
+
+import os
+import pandas as pd
+import numpy as np
+import argparse
+import json
+from pathlib import Path
+from datetime import datetime
+from typing import Dict, List, Optional
+import matplotlib.pyplot as plt
+import seaborn as sns
+
+
+class VideoQAEvaluator:
+    """视频问答评测器"""
+    
+    # 答案索引到选项字母的映射
+    INDEX_TO_CHOICE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
+    CHOICE_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
+    
+    def __init__(self, result_file: str):
+        """
+        初始化评测器
+        
+        Args:
+            result_file: 结果CSV文件路径
+        """
+        self.result_file = result_file
+        self.df = None
+        self.metrics = {}
+        
+        # 加载数据
+        self._load_data()
+    
+    def _load_data(self):
+        """加载结果数据"""
+        print(f"📂 加载结果文件: {self.result_file}")
+        
+        if not os.path.exists(self.result_file):
+            raise FileNotFoundError(f"结果文件不存在: {self.result_file}")
+        
+        self.df = pd.read_csv(self.result_file)
+        print(f"✅ 成功加载 {len(self.df)} 条数据")
+        
+        # 检查必需的列
+        required_columns = ['video_id', 'question', 'answer', 'pred_choice', 'qa_acc']
+        missing_columns = [col for col in required_columns if col not in self.df.columns]
+        
+        if missing_columns:
+            raise ValueError(f"缺少必需的列: {missing_columns}")
+        
+        print(f"📊 数据列: {list(self.df.columns)}")
+    
+    def calculate_metrics(self) -> Dict:
+        """
+        计算各种评测指标
+        
+        Returns:
+            包含所有指标的字典
+        """
+        print("\n" + "="*60)
+        print("🔍 计算评测指标...")
+        print("="*60)
+        
+        # 基础统计
+        total_samples = len(self.df)
+        correct_samples = (self.df['qa_acc'] == 1.0).sum()
+        accuracy = self.df['qa_acc'].mean() * 100
+        
+        self.metrics['basic'] = {
+            'total_samples': total_samples,
+            'correct_samples': int(correct_samples),
+            'wrong_samples': int(total_samples - correct_samples),
+            'accuracy': accuracy,
+            'error_rate': 100 - accuracy
+        }
+        
+        # 按视频统计（如果有多个视频）
+        video_stats = self.df.groupby('video_id').agg({
+            'qa_acc': ['count', 'sum', 'mean']
+        }).round(4)
+        
+        self.metrics['per_video'] = video_stats
+        
+        # 答案分布分析 - 转换为字母
+        if 'answer' in self.df.columns:
+            # 将索引转换为字母
+            answer_letters = self.df['answer'].map(self.INDEX_TO_CHOICE)
+            answer_dist = answer_letters.value_counts().sort_index()
+            self.metrics['answer_distribution'] = answer_dist.to_dict()
+        
+        # 预测答案分布分析
+        if 'pred_choice' in self.df.columns:
+            pred_dist = self.df['pred_choice'].value_counts().sort_index()
+            self.metrics['pred_distribution'] = pred_dist.to_dict()
+        
+        # 混淆矩阵 - 正确答案 vs 预测答案
+        if 'answer' in self.df.columns and 'pred_choice' in self.df.columns:
+            confusion = pd.crosstab(
+                self.df['answer'].map(self.INDEX_TO_CHOICE),
+                self.df['pred_choice'],
+                rownames=['Ground Truth'],
+                colnames=['Predicted']
+            )
+            self.metrics['confusion_matrix'] = confusion.to_dict()
+        
+        # 配置参数统计
+        if 'retrieve_size' in self.df.columns:
+            self.metrics['config'] = {
+                'retrieve_size': self.df['retrieve_size'].iloc[0] if len(self.df) > 0 else None,
+                'chunk_size': self.df['chunk_size'].iloc[0] if 'chunk_size' in self.df.columns and len(self.df) > 0 else None
+            }
+        
+        return self.metrics
+    
+    def print_summary(self):
+        """打印评测摘要"""
+        if not self.metrics:
+            self.calculate_metrics()
+        
+        basic = self.metrics['basic']
+        
+        print("\n" + "="*60)
+        print("📊 评测结果摘要")
+        print("="*60)
+        print(f"总样本数:        {basic['total_samples']}")
+        print(f"正确数量:        {basic['correct_samples']} ✅")
+        print(f"错误数量:        {basic['wrong_samples']} ❌")
+        print(f"准确率:          {basic['accuracy']:.2f}%")
+        print(f"错误率:          {basic['error_rate']:.2f}%")
+        print("="*60)
+        
+        # 配置信息
+        if 'config' in self.metrics and self.metrics['config']['retrieve_size']:
+            print(f"\n📝 配置参数:")
+            print(f"检索大小 (retrieve_size): {self.metrics['config']['retrieve_size']}")
+            if self.metrics['config']['chunk_size']:
+                print(f"块大小 (chunk_size): {self.metrics['config']['chunk_size']}")
+        
+        # 答案分布
+        if 'answer_distribution' in self.metrics:
+            print(f"\n📈 正确答案分布:")
+            for ans, count in sorted(self.metrics['answer_distribution'].items()):
+                percentage = (count / basic['total_samples']) * 100
+                print(f"  选项 {ans}: {count} 次 ({percentage:.1f}%)")
+        
+        # 预测分布
+        if 'pred_distribution' in self.metrics:
+            print(f"\n🎯 模型预测分布:")
+            for choice, count in sorted(self.metrics['pred_distribution'].items()):
+                percentage = (count / basic['total_samples']) * 100
+                print(f"  选项 {choice}: {count} 次 ({percentage:.1f}%)")
+        
+        # 混淆矩阵
+        if 'confusion_matrix' in self.metrics:
+            print(f"\n🔀 混淆矩阵 (Ground Truth vs Predicted):")
+            confusion_df = pd.DataFrame(self.metrics['confusion_matrix']).fillna(0).astype(int)
+            print(confusion_df.to_string())
+    
+    def analyze_errors(self, top_n: int = 10) -> pd.DataFrame:
+        """
+        分析错误样本
+        
+        Args:
+            top_n: 显示前N个错误样本
+            
+        Returns:
+            错误样本的DataFrame
+        """
+        print(f"\n🔎 分析错误样本 (显示前{top_n}个)...")
+        print("="*60)
+        
+        # 获取错误样本
+        error_df = self.df[self.df['qa_acc'] == 0.0].copy()
+        
+        if len(error_df) == 0:
+            print("🎉 没有错误样本！所有预测都正确！")
+            return error_df
+        
+        print(f"总错误数: {len(error_df)}\n")
+        
+        # 显示前N个错误
+        for i, (idx, row) in enumerate(error_df.head(top_n).iterrows(), 1):
+            print(f"错误样本 #{i}")
+            print(f"  视频ID: {row['video_id']}")
+            print(f"  问题: {row['question'][:100]}...")  # 只显示前100个字符
+            
+            # 转换索引为字母
+            correct_letter = self.INDEX_TO_CHOICE.get(row['answer'], '?')
+            correct_text = str(row.get('correct_choice', 'N/A'))
+            
+            print(f"  正确答案: {correct_letter}) {correct_text[:80]}...")
+            
+            # 预测答案
+            pred_letter = str(row.get('pred_choice', '?'))
+            pred_text = str(row.get('pred_answer', 'N/A'))
+            print(f"  模型预测: {pred_letter}) {pred_text[:80]}...")
+            print()
+        
+        # 错误分析统计
+        print("\n📊 错误分析统计:")
+        
+        # 统计每个正确答案的错误率
+        if 'answer' in error_df.columns:
+            error_by_answer = error_df['answer'].map(self.INDEX_TO_CHOICE).value_counts().sort_index()
+            total_by_answer = self.df['answer'].map(self.INDEX_TO_CHOICE).value_counts().sort_index()
+            
+            print("\n各选项的错误分布:")
+            for choice in sorted(set(list(error_by_answer.index) + list(total_by_answer.index))):
+                errors = error_by_answer.get(choice, 0)
+                total = total_by_answer.get(choice, 0)
+                error_rate = (errors / total * 100) if total > 0 else 0
+                print(f"  正确答案为{choice}: {errors}/{total} 错误 ({error_rate:.1f}%)")
+        
+        # 统计最常见的错误预测
+        if 'pred_choice' in error_df.columns:
+            print("\n错误样本中最常见的预测:")
+            pred_counts = error_df['pred_choice'].value_counts().head(5)
+            for pred, count in pred_counts.items():
+                percentage = (count / len(error_df)) * 100
+                print(f"  预测{pred}: {count} 次 ({percentage:.1f}%)")
+        
+        return error_df
+    
+    def save_detailed_report(self, output_dir: Optional[str] = None):
+        """
+        保存详细报告
+        
+        Args:
+            output_dir: 输出目录，默认与结果文件同目录
+        """
+        if output_dir is None:
+            output_dir = os.path.dirname(self.result_file)
+        
+        os.makedirs(output_dir, exist_ok=True)
+        
+        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
+        
+        # 1. 保存JSON格式的指标
+        metrics_file = os.path.join(output_dir, f'metrics_{timestamp}.json')
+        
+        # 转换不可序列化的对象
+        json_metrics = {}
+        for key, value in self.metrics.items():
+            if key == 'per_video':
+                # 将多级索引的 DataFrame 转换为可序列化的格式
+                per_video_dict = {}
+                for video_id, row in value.iterrows():
+                    per_video_dict[str(video_id)] = {
+                        'count': int(row[('qa_acc', 'count')]),
+                        'correct': int(row[('qa_acc', 'sum')]),
+                        'accuracy': float(row[('qa_acc', 'mean')])
+                    }
+                json_metrics[key] = per_video_dict
+            elif key == 'confusion_matrix':
+                # 确保混淆矩阵的键都是字符串
+                if isinstance(value, dict):
+                    json_metrics[key] = {str(k): v for k, v in value.items()}
+                else:
+                    json_metrics[key] = value
+            else:
+                json_metrics[key] = value
+        
+        with open(metrics_file, 'w', encoding='utf-8') as f:
+            json.dump(json_metrics, f, indent=2, ensure_ascii=False)
+        
+        print(f"💾 指标已保存: {metrics_file}")
+        
+        # 2. 保存Markdown格式的报告
+        report_file = os.path.join(output_dir, f'evaluation_report_{timestamp}.md')
+        self._generate_markdown_report(report_file)
+        print(f"📝 报告已保存: {report_file}")
+        
+        # 3. 保存错误样本
+        error_df = self.df[self.df['qa_acc'] == 0.0]
+        if len(error_df) > 0:
+            error_file = os.path.join(output_dir, f'error_samples_{timestamp}.csv')
+            error_df.to_csv(error_file, index=False)
+            print(f"❌ 错误样本已保存: {error_file}")
+    def _generate_markdown_report(self, output_file: str):
+        """生成Markdown格式的报告"""
+        basic = self.metrics['basic']
+        
+        md_content = f"""# 📊 视频问答评测报告
+
+    **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
+    **结果文件**: `{self.result_file}`
+
+    ---
+
+    ## 1. 总体评测结果
+
+    | 指标 | 数值 |
+    |------|------|
+    | 总样本数 | {basic['total_samples']} |
+    | 正确数量 | {basic['correct_samples']} ✅ |
+    | 错误数量 | {basic['wrong_samples']} ❌ |
+    | **准确率** | **{basic['accuracy']:.2f}%** |
+    | 错误率 | {basic['error_rate']:.2f}% |
+
+    ---
+
+    ## 2. 配置参数
+
+    """
+        
+        if 'config' in self.metrics:
+            config = self.metrics['config']
+            md_content += f"- **检索大小 (retrieve_size)**: {config.get('retrieve_size', 'N/A')}\n"
+            md_content += f"- **块大小 (chunk_size)**: {config.get('chunk_size', 'N/A')}\n"
+        
+        md_content += "\n---\n\n## 3. 答案分布分析\n\n"
+        
+        # 正确答案分布
+        if 'answer_distribution' in self.metrics:
+            md_content += "### 3.1 正确答案分布\n\n"
+            md_content += "| 选项 | 出现次数 | 占比 |\n"
+            md_content += "|------|---------|------|\n"
+            
+            for ans, count in sorted(self.metrics['answer_distribution'].items()):
+                percentage = (count / basic['total_samples']) * 100
+                md_content += f"| {ans} | {count} | {percentage:.1f}% |\n"
+        
+        # 预测分布
+        if 'pred_distribution' in self.metrics:
+            md_content += "\n### 3.2 模型预测分布\n\n"
+            md_content += "| 预测选项 | 次数 | 占比 |\n"
+            md_content += "|---------|------|------|\n"
+            
+            for choice, count in sorted(self.metrics['pred_distribution'].items()):
+                percentage = (count / basic['total_samples']) * 100
+                md_content += f"| {choice} | {count} | {percentage:.1f}% |\n"
+        
+        # 混淆矩阵
+        if 'confusion_matrix' in self.metrics:
+            md_content += "\n### 3.3 混淆矩阵 (Ground Truth vs Predicted)\n\n"
+            try:
+                confusion_df = pd.DataFrame(self.metrics['confusion_matrix']).fillna(0).astype(int)
+                # 确保行和列都按字母顺序排列
+                all_choices = sorted(set(list(confusion_df.index) + list(confusion_df.columns)))
+                confusion_df = confusion_df.reindex(index=all_choices, columns=all_choices, fill_value=0)
+                md_content += confusion_df.to_markdown() + "\n"
+            except Exception as e:
+                md_content += f"无法生成混淆矩阵: {e}\n"
+        
+        # 每个视频的统计
+        if 'per_video' in self.metrics and len(self.metrics['per_video']) > 1:
+            md_content += "\n---\n\n## 4. 按视频统计\n\n"
+            md_content += "| 视频ID | 样本数 | 正确数 | 准确率 |\n"
+            md_content += "|--------|--------|--------|--------|\n"
+            
+            for video_id, row in self.metrics['per_video'].iterrows():
+                count = int(row[('qa_acc', 'count')])
+                correct = int(row[('qa_acc', 'sum')])
+                acc = row[('qa_acc', 'mean')] * 100
+                video_id_short = str(video_id)[:30] + "..." if len(str(video_id)) > 30 else str(video_id)
+                md_content += f"| {video_id_short} | {count} | {correct} | {acc:.2f}% |\n"
+        
+        md_content += "\n---\n\n## 5. 性能分析\n\n"
+        
+        if basic['accuracy'] >= 80:
+            md_content += "✅ **优秀**: 模型表现出色，准确率超过80%\n"
+        elif basic['accuracy'] >= 60:
+            md_content += "⚠️ **良好**: 模型表现尚可，但仍有改进空间\n"
+        elif basic['accuracy'] >= 40:
+            md_content += "⚠️ **一般**: 模型表现中等，需要优化\n"
+        elif basic['accuracy'] > 0:
+            md_content += "❌ **较差**: 模型表现较差，需要重大改进\n"
+        else:
+            md_content += "🚨 **完全失败**: 所有预测都错误！请检查:\n"
+            md_content += "   - 数据格式是否正确\n"
+            md_content += "   - 答案索引是否对齐\n"
+            md_content += "   - 模型输出是否有效\n"
+        
+        md_content += f"\n### 改进建议\n\n"
+        
+        if basic['accuracy'] == 0:
+            md_content += "🚨 **紧急**: 模型完全没有预测正确，请立即检查:\n"
+            md_content += "1. 检查答案格式和索引是否正确对齐\n"
+            md_content += "2. 验证模型输出格式是否符合预期\n"
+            md_content += "3. 检查数据预处理流程是否有误\n"
+            md_content += "4. 确认评测脚本的逻辑是否正确\n"
+        elif basic['error_rate'] > 50:
+            md_content += "1. 检查模型架构和训练数据质量\n"
+            md_content += "2. 考虑增加训练数据或改进数据增强策略\n"
+            md_content += "3. 调整超参数或训练策略\n"
+        elif basic['error_rate'] > 20:
+            md_content += "1. 分析错误样本，找出模型的薄弱环节\n"
+            md_content += "2. 考虑针对性地改进模型或数据\n"
+            md_content += "3. 可以尝试集成学习或模型融合\n"
+        else:
+            md_content += "1. 继续保持当前策略\n"
+            md_content += "2. 可以尝试更复杂的场景或数据集\n"
+            md_content += "3. 考虑模型压缩和效率优化\n"
+        
+        # 保存文件
+        with open(output_file, 'w', encoding='utf-8') as f:
+            f.write(md_content)
+    
+    def visualize_results(self, output_dir: Optional[str] = None, show: bool = False):
+        """
+        可视化评测结果
+        
+        Args:
+            output_dir: 输出目录
+            show: 是否显示图表
+        """
+        if output_dir is None:
+            output_dir = os.path.dirname(self.result_file)
+        
+        os.makedirs(output_dir, exist_ok=True)
+        
+        # 设置中文字体
+        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
+        plt.rcParams['axes.unicode_minus'] = False
+        
+        # 创建图表
+        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
+        fig.suptitle('Video QA Evaluation Results', fontsize=16, fontweight='bold')
+        
+        basic = self.metrics['basic']
+        
+        # 1. 准确率饼图
+        ax1 = axes[0, 0]
+        sizes = [basic['correct_samples'], basic['wrong_samples']]
+        labels = ['Correct', 'Wrong']
+        colors = ['#4CAF50', '#F44336']
+        explode = (0.1, 0)
+        
+        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
+                autopct='%1.1f%%', shadow=True, startangle=90)
+        ax1.set_title(f"Accuracy: {basic['accuracy']:.2f}%")
+        
+        # 2. 答案分布柱状图
+        ax2 = axes[0, 1]
+        if 'answer_distribution' in self.metrics:
+            ans_dist = self.metrics['answer_distribution']
+            choices = sorted(ans_dist.keys())
+            counts = [ans_dist[c] for c in choices]
+            ax2.bar(choices, counts, color='#2196F3')
+            ax2.set_xlabel('Answer Choice')
+            ax2.set_ylabel('Count')
+            ax2.set_title('Ground Truth Answer Distribution')
+            ax2.grid(axis='y', alpha=0.3)
+        
+        # 3. 预测分布柱状图
+        ax3 = axes[1, 0]
+        if 'pred_distribution' in self.metrics:
+            pred_dist = self.metrics['pred_distribution']
+            choices = sorted(pred_dist.keys())
+            counts = [pred_dist[c] for c in choices]
+            ax3.bar(choices, counts, color='#FF9800')
+            ax3.set_xlabel('Predicted Choice')
+            ax3.set_ylabel('Count')
+            ax3.set_title('Model Prediction Distribution')
+            ax3.grid(axis='y', alpha=0.3)
+        
+        # 4. 统计信息文本
+        ax4 = axes[1, 1]
+        ax4.axis('off')
+        
+        stats_text = f"""
+Total Samples: {basic['total_samples']}
+Correct: {basic['correct_samples']}
+Wrong: {basic['wrong_samples']}
+Accuracy: {basic['accuracy']:.2f}%
+Error Rate: {basic['error_rate']:.2f}%
+        """
+        
+        if 'config' in self.metrics:
+            config = self.metrics['config']
+            stats_text += f"""
+Retrieve Size: {config.get('retrieve_size', 'N/A')}
+Chunk Size: {config.get('chunk_size', 'N/A')}
+            """
+        
+        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
+                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
+        
+        plt.tight_layout()
+        
+        # 保存图表
+        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
+        plot_file = os.path.join(output_dir, f'evaluation_plot_{timestamp}.png')
+        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
+        print(f"📈 图表已保存: {plot_file}")
+        
+        if show:
+            plt.show()
+        else:
+            plt.close()
+
+
+def compare_experiments(result_files: List[str], output_dir: str):
+    """
+    比较多个实验结果
+    
+    Args:
+        result_files: 结果文件列表
+        output_dir: 输出目录
+    """
+    print("\n" + "="*60)
+    print("🔄 比较多个实验结果...")
+    print("="*60)
+    
+    results = []
+    
+    for result_file in result_files:
+        try:
+            evaluator = VideoQAEvaluator(result_file)
+            evaluator.calculate_metrics()
+            
+            exp_name = Path(result_file).parent.name
+            results.append({
+                'experiment': exp_name,
+                'file': result_file,
+                'metrics': evaluator.metrics['basic']
+            })
+        except Exception as e:
+            print(f"⚠️ 无法加载 {result_file}: {e}")
+    
+    if not results:
+        print("❌ 没有有效的结果文件")
+        return
+    
+    # 创建对比表
+    comparison_df = pd.DataFrame([
+        {
+            'Experiment': r['experiment'],
+            'Total': r['metrics']['total_samples'],
+            'Correct': r['metrics']['correct_samples'],
+            'Wrong': r['metrics']['wrong_samples'],
+            'Accuracy (%)': f"{r['metrics']['accuracy']:.2f}",
+            'Error Rate (%)': f"{r['metrics']['error_rate']:.2f}"
+        }
+        for r in results
+    ])
+    
+    print("\n📊 实验对比:")
+    print(comparison_df.to_string(index=False))
+    
+    # 保存对比报告
+    os.makedirs(output_dir, exist_ok=True)
+    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
+    
+    comparison_file = os.path.join(output_dir, f'comparison_{timestamp}.csv')
+    comparison_df.to_csv(comparison_file, index=False)
+    print(f"\n💾 对比结果已保存: {comparison_file}")
+    
+    # 生成对比图表
+    if len(results) > 1:
+        fig, ax = plt.subplots(figsize=(12, 6))
+        
+        experiments = [r['experiment'] for r in results]
+        accuracies = [r['metrics']['accuracy'] for r in results]
+        
+        bars = ax.bar(experiments, accuracies, color='#4CAF50')
+        ax.set_ylabel('Accuracy (%)', fontsize=12)
+        ax.set_title('Experiment Comparison', fontsize=14, fontweight='bold')
+        ax.set_ylim(0, 100)
+        ax.grid(axis='y', alpha=0.3)
+        
+        # 在柱状图上添加数值标签
+        for bar in bars:
+            height = bar.get_height()
+            ax.text(bar.get_x() + bar.get_width()/2., height,
+                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
+        
+        plt.xticks(rotation=45, ha='right')
+        plt.tight_layout()
+        
+        plot_file = os.path.join(output_dir, f'comparison_plot_{timestamp}.png')
+        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
+        print(f"📈 对比图表已保存: {plot_file}")
+        plt.close()
+
+
+def evaluate_batch_results(batch_dir: str):
+    """
+    评测批量实验结果
+    
+    Args:
+        batch_dir: 批量实验结果目录（包含多个子目录）
+    """
+    print(f"\n🔍 扫描批量实验目录: {batch_dir}")
+    
+    result_files = []
+    
+    # 递归查找所有results.csv文件
+    for root, dirs, files in os.walk(batch_dir):
+        for file in files:
+            if file == 'results.csv':
+                result_file = os.path.join(root, file)
+                result_files.append(result_file)
+    
+    if not result_files:
+        print(f"❌ 在 {batch_dir} 中未找到results.csv文件")
+        return
+    
+    print(f"✅ 找到 {len(result_files)} 个结果文件\n")
+    
+    # 评测每个实验
+    for result_file in result_files:
+        print("\n" + "="*80)
+        exp_dir = os.path.dirname(result_file)
+        exp_name = os.path.basename(exp_dir)
+        print(f"📊 评测实验: {exp_name}")
+        print("="*80)
+        
+        try:
+            evaluator = VideoQAEvaluator(result_file)
+            evaluator.calculate_metrics()
+            evaluator.print_summary()
+            evaluator.analyze_errors(top_n=3)
+            evaluator.save_detailed_report(output_dir=exp_dir)
+            # evaluator.visualize_results(output_dir=exp_dir)  # 可选：生成图表
+        except Exception as e:
+            print(f"❌ 评测失败: {e}")
+            import traceback
+            traceback.print_exc()
+    
+    # 对比所有实验
+    if len(result_files) > 1:
+        compare_experiments(result_files, batch_dir)
+
+
+def main():
+    parser = argparse.ArgumentParser(description='视频问答评测脚本')
+    parser.add_argument('--result_file', type=str, help='单个结果CSV文件路径')
+    parser.add_argument('--result_dir', type=str, help='批量结果目录路径')
+    parser.add_argument('--output_dir', type=str, help='输出目录（可选）')
+    parser.add_argument('--visualize', action='store_true', help='生成可视化图表')
+    parser.add_argument('--show_plot', action='store_true', help='显示图表')
+    parser.add_argument('--top_errors', type=int, default=10, help='显示的错误样本数量')
+    
+    args = parser.parse_args()
+    
+    if args.result_dir:
+        # 批量评测模式
+        evaluate_batch_results(args.result_dir)
+    
+    elif args.result_file:
+        # 单文件评测模式
+        print("="*60)
+        print("🚀 开始评测...")
+        print("="*60)
+        
+        evaluator = VideoQAEvaluator(args.result_file)
+        evaluator.calculate_metrics()
+        evaluator.print_summary()
+        evaluator.analyze_errors(top_n=args.top_errors)
+        
+        output_dir = args.output_dir if args.output_dir else os.path.dirname(args.result_file)
+        evaluator.save_detailed_report(output_dir=output_dir)
+        
+        if args.visualize:
+            evaluator.visualize_results(output_dir=output_dir, show=args.show_plot)
+        
+        print("\n" + "="*60)
+        print("✅ 评测完成!")
+        print("="*60)
+    
+    else:
+        print("❌ 请指定 --result_file 或 --result_dir 参数")
+        parser.print_help()
+
+
+if __name__ == "__main__":
+    main()
\ No newline at end of file
diff --git a/models/rekv/model/video_qa/eval/eval_smoke.py b/models/rekv/model/video_qa/eval/eval_smoke.py
new file mode 100644
index 0000000..ec778f0
--- /dev/null
+++ b/models/rekv/model/video_qa/eval/eval_smoke.py
@@ -0,0 +1,25 @@
+"""Minimal smoke evaluator without optional plotting dependencies."""
+
+from __future__ import annotations
+
+import argparse
+from pathlib import Path
+
+import pandas as pd
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--save_dir", required=True)
+    args = parser.parse_args()
+
+    result_file = Path(args.save_dir) / "results.csv"
+    df = pd.read_csv(result_file)
+    acc = float(df["qa_acc"].mean()) if "qa_acc" in df and len(df) else 0.0
+    summary_file = Path(args.save_dir) / "summary.txt"
+    summary_file.write_text(f"samples={len(df)}\nqa_acc={acc:.2f}\n", encoding="utf-8")
+    print(f"SMOKE_EVAL samples={len(df)} qa_acc={acc:.2f}")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/models/rekv/model/video_qa/eval/eval_videomme.py b/models/rekv/model/video_qa/eval/eval_videomme.py
new file mode 100644
index 0000000..8bb88ae
--- /dev/null
+++ b/models/rekv/model/video_qa/eval/eval_videomme.py
@@ -0,0 +1,168 @@
+import os
+import pandas as pd
+import argparse
+import matplotlib.pyplot as plt
+import seaborn as sns
+
+
+def calc_average_metric(results, save_dir, metric, vmin=None, vmax=None):
+    if isinstance(results, list):
+        average_metric = sum([item[metric] for item in results]) / len(results)
+        print(f'#Samples: {len(results)}')
+        print(f'Average {metric}: {average_metric:.2f}')
+
+    elif isinstance(results, dict):
+        average_recall = {}
+        for key, value in results.items():
+            recalls = [item[metric] for item in value]
+            if len(value) > 0:
+                average_recall[key] = (sum(recalls) / len(recalls))
+            else:
+                average_recall[key] = None
+
+        df = pd.DataFrame.from_dict(average_recall, orient='index')
+        df.index = pd.MultiIndex.from_tuples(df.index, names=['retrieve_size', 'chunk_size'])
+        df = df.reset_index()
+        df.columns = ['retrieve_size', 'chunk_size', 'value']
+        heatmap_data = df.pivot(index='chunk_size', columns='retrieve_size', values='value')
+        plt.figure(figsize=(10, 8))
+        ax = sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdPu", cbar_kws={'label': 'Value'}, 
+                        xticklabels=True, yticklabels=True, vmin=vmin, vmax=vmax)
+        ax.invert_yaxis()
+        plt.title(f'Heatmap of Average {metric.capitalize()}')
+        plt.xlabel('Retrieve Size')
+        plt.ylabel('Chunk Size')
+        plt.tight_layout()
+        plt.savefig(os.path.join(save_dir, f'{metric}.png'))
+        plt.close()
+
+        print(f'#Samples: {len(results[list(results.keys())[0]])}')
+        print(average_recall)
+        os.system(f"imgcat {os.path.join(save_dir, f'{metric}.png')}")
+    else:
+        raise ValueError(f"Invalid record type: {type(results)}")
+
+    print(f'save_dir: {save_dir}')
+
+
+def evaluate_results(df):
+    """Evaluate the results including overall and duration-specific accuracy"""
+    print("="*50)
+    print("EVALUATION RESULTS")
+    print("="*50)
+    
+    # Overall accuracy
+    total_samples = len(df)
+    total_correct = sum(df['qa_acc'] == 100.0)
+    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
+    
+    print(f"Total samples: {total_samples}")
+    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
+    print()
+    
+    # Accuracy by duration
+    durations = ['short', 'medium', 'long']
+    duration_results = {}
+    
+    for duration in durations:
+        duration_df = df[df['duration'] == duration]
+        if len(duration_df) > 0:
+            duration_correct = sum(duration_df['qa_acc'] == 100.0)
+            duration_accuracy = (duration_correct / len(duration_df)) * 100
+            duration_results[duration] = {
+                'accuracy': duration_accuracy,
+                'correct': duration_correct,
+                'total': len(duration_df)
+            }
+            print(f"{duration.capitalize()} Duration Accuracy: {duration_accuracy:.2f}% ({duration_correct}/{len(duration_df)})")
+        else:
+            duration_results[duration] = {
+                'accuracy': 0,
+                'correct': 0,
+                'total': 0
+            }
+            print(f"{duration.capitalize()} Duration Accuracy: 0.00% (0/0)")
+    
+    print()
+    
+    # Summary table
+    print("Summary by Duration:")
+    print("-" * 60)
+    print(f"{'Duration':<12} {'Accuracy':<12} {'Correct':<10} {'Total':<10}")
+    print("-" * 60)
+    for duration in durations:
+        acc = duration_results[duration]['accuracy']
+        correct = duration_results[duration]['correct']
+        total = duration_results[duration]['total']
+        print(f"{duration.capitalize():<12} {acc:>8.2f}%    {correct:<10} {total:<10}")
+    print("-" * 60)
+    print(f"{'Overall':<12} {overall_accuracy:>8.2f}%    {total_correct:<10} {total_samples:<10}")
+    
+    # Additional statistics
+    print("\nAdditional Statistics:")
+    print(f"Average QA Accuracy (0-100 scale): {df['qa_acc'].mean():.2f}")
+    print(f"Std Deviation: {df['qa_acc'].std():.2f}")
+    print(f"Min Accuracy: {df['qa_acc'].min():.2f}")
+    print(f"Max Accuracy: {df['qa_acc'].max():.2f}")
+    
+    # Error analysis
+    if 'pred_answer' in df.columns and 'correct_choice' in df.columns:
+        n_errors = 0
+        for _, row in df.iterrows():
+            if str(row['pred_answer'])[0] not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
+                n_errors += 1
+        print(f"\nPrediction format errors: {n_errors} ({n_errors/len(df)*100:.2f}%)")
+    
+    # Distribution of durations
+    print(f"\nDistribution of durations:")
+    duration_counts = df['duration'].value_counts()
+    for duration in durations:
+        count = duration_counts.get(duration, 0)
+        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
+        print(f"  {duration.capitalize()}: {count} samples ({percentage:.2f}%)")
+    
+    return overall_accuracy, duration_results
+
+
+parser = argparse.ArgumentParser()
+parser.add_argument('--save_dir', type=str)
+parser.add_argument('--results_path', type=str, default=None)
+parser.add_argument('--debug', action='store_true')
+args = parser.parse_args()
+
+if args.results_path is not None:
+    df = pd.read_csv(args.results_path)
+    args.save_dir = os.path.dirname(args.results_path)
+else:
+    df = pd.read_csv(os.path.join(args.save_dir, 'results.csv'))
+
+# Evaluate the results
+overall_acc, duration_acc = evaluate_results(df)
+
+if 'retrieve_size' in df.columns:
+    results = {}
+    for _, row in df.iterrows():
+        key = (row['retrieve_size'], row['chunk_size'])
+        value = {col: row[col] for col in df.columns if col not in ['retrieve_size', 'chunk_size']}
+        if key not in results:
+            results[key] = []
+        results[key].append(value)
+else:
+    results = df.to_dict(orient='records')
+
+if 'recall' in df.columns:
+    metrics = ['recall', 'precision', 'f1', 'qa_acc', 'acc_at_gqa']
+else:
+    metrics = ['qa_acc']
+
+for metric in metrics:
+    calc_average_metric(results, args.save_dir, metric)
+
+if 'pred_choice' in df.columns:
+    n_errors = 0
+    for _, row in df.iterrows():
+        if row['pred_answer'][0] not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
+            n_errors += 1
+            if args.debug:
+                print(f'Video: {row["video_id"]}, Question: {row["question"]}, GT: {row["correct_choice"]}, Pred: {row["pred_answer"]}')
+    print(f'%Errors: {n_errors / len(df) * 100:.2f}')
\ No newline at end of file
diff --git a/models/rekv/model/video_qa/eval/evaluate.py b/models/rekv/model/video_qa/eval/evaluate.py
new file mode 100644
index 0000000..e69de29
diff --git a/models/rekv/model/video_qa/rekv_offline_refactored.py b/models/rekv/model/video_qa/rekv_offline_refactored.py
new file mode 100644
index 0000000..15e1c79
--- /dev/null
+++ b/models/rekv/model/video_qa/rekv_offline_refactored.py
@@ -0,0 +1,77 @@
+"""重构的离线推理 - 简洁优雅的实现"""
+import torch
+from logzero import logger
+from .base_refactored import BaseVQA
+import torch.distributed as dist
+
+
+class ReKVOfflineVQA(BaseVQA):
+    """离线视频问答 - 所有函数<15行"""
+    
+    def answer_single(self, qa_pair, video_id):
+        """回答单个问题"""
+        if 'choices' in qa_pair:
+            return self._multiple_choice_qa(qa_pair, video_id)
+        return self._open_qa(qa_pair, video_id)
+    
+    def _open_qa(self, qa_pair, video_id):
+        """开放式问答"""
+        question = qa_pair['question']
+        prompt = self.format_openqa_prompt(question)
+        pred = self.model.question_answering(
+            {"question": question, "prompt": prompt},
+            max_new_tokens=1024
+        )
+        return self._format_open_result(pred, qa_pair, video_id)
+    
+    def _multiple_choice_qa(self, qa_pair, video_id):
+        """多选题问答"""
+        question = qa_pair['question']
+        choices = qa_pair['choices']
+        prompt = self.format_mcqa_prompt(question, choices)
+        
+        pred = self.model.question_answering(
+            {"question": question, "prompt": prompt},
+            max_new_tokens=16
+        )
+        return self._format_mc_result(pred, qa_pair, video_id)
+    
+    def _format_open_result(self, pred, qa_pair, video_id):
+        """格式化开放式问答结果"""
+        return {
+            'video_id': video_id,
+            'question': qa_pair['question'],
+            'answer': qa_pair.get('answer'),
+            'pred_answer': pred.replace('\n', ''),
+        }
+    
+    def _format_mc_result(self, pred, qa_pair, video_id):
+        """格式化多选题结果"""
+        pred_choice = self.extract_choice(pred)
+        correct_choice = self._get_correct_choice(qa_pair)
+        
+        return {
+            'video_id': video_id,
+            'question': qa_pair['question'],
+            'choices': qa_pair['choices'],
+            'answer': qa_pair.get('answer'),
+            'correct_choice': correct_choice,
+            'pred_answer': pred.replace('\n', ''),
+            'pred_choice': pred_choice,
+            'qa_acc': float(pred_choice == correct_choice) * 100,
+        }
+    
+    def _get_correct_choice(self, qa_pair):
+        """获取正确选项"""
+        answer = qa_pair.get('answer')
+        if answer is None:
+            return self.choice_letters[0]
+        
+        choices = qa_pair['choices']
+        try:
+            idx = choices.index(answer)
+            return self.choice_letters[idx]
+        except ValueError:
+            logger.warning(f"Answer not in choices: {answer}")
+            return self.choice_letters[0]
+
diff --git a/model/video_qa/rekv_offline_vqa.py b/model/video_qa/rekv_offline_vqa.py
deleted file mode 100644
index e9a9301..0000000
--- a/model/video_qa/rekv_offline_vqa.py
+++ /dev/null
@@ -1,80 +0,0 @@
-import torch
-from logzero import logger
-
-from video_qa.base import BaseVQA, work
-
-
-class ReKVOfflineVQA(BaseVQA):
-    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None):
-        input_text = {
-            "question": question,
-            "prompt": self.qa_model.get_prompt(question)
-        }
-
-        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens, retrieved_indices=retrieved_indices)
-
-        return {
-            'pred_answer': pred_answer.replace('\n', ''),
-        }
-
-    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None):
-        input_text = self.format_mcqa_prompt(question, candidates)
-        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
-        pred_letter = self.extract_characters_regex(pred_answer)
-        return {
-            'pred_answer': pred_answer.replace('\n', ''),
-            'pred_choice': pred_letter,
-            'acc': float(pred_letter == correct_choice),
-        }
-
-    @torch.inference_mode()
-    def analyze_a_video(self, video_sample):
-        # load and preprocess video frames for QA
-        video_path = video_sample['video_path']
-        video = self.load_video(video_path)
-        if not isinstance(video, torch.Tensor):
-            video_tensor = torch.from_numpy(video)
-        else:
-            video_tensor = video
-
-        self.qa_model.clear_cache()
-        self.qa_model.encode_init_prompt()
-        self.qa_model.encode_video(video_tensor)
-
-        for sample in video_sample['conversations']:
-            logger.debug(f'sample: {sample}')
-            question = sample['question']
-            answer = sample['answer']
-            
-            # QA
-            if 'choices' in sample:  # CloseQA
-                choices = sample['choices']
-                if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
-                    answer = choices[0]
-                correct_choice = self.choice_letters[choices.index(answer)]
-                qa_results = self.video_close_qa(question, choices, correct_choice)
-                self.record[(self.retrieve_size, self.chunk_size)].append({
-                    'video_id': video_sample['video_id'],
-                    'question': question,
-                    'choices': choices,
-                    'answer': answer,
-                    'correct_choice': correct_choice,
-                    'pred_answer': qa_results['pred_answer'],
-                    'pred_choice': qa_results['pred_choice'],
-                    'qa_acc': qa_results['acc'] * 100,
-                })
-            else:  # OpenQA
-                qa_results = self.video_open_qa(question)
-                self.record[(self.retrieve_size, self.chunk_size)].append({
-                    'video_id': video_sample['video_id'],
-                    'question': question,
-                    'answer': answer,
-                    'pred_answer': qa_results['pred_answer'],
-                })
-
-            if 'question_type' in sample:
-                self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['question_type']
-
-
-if __name__ == "__main__":
-    work(ReKVOfflineVQA)
diff --git a/models/rekv/model/video_qa/rekv_stream_refactored.py b/models/rekv/model/video_qa/rekv_stream_refactored.py
new file mode 100644
index 0000000..c4ffc04
--- /dev/null
+++ b/models/rekv/model/video_qa/rekv_stream_refactored.py
@@ -0,0 +1,76 @@
+"""流式视频问答 - 重构版本"""
+import torch
+import numpy as np
+from logzero import logger
+from decord import VideoReader, cpu
+from .base_refactored import BaseVQA
+
+
+class ReKVStreamVQA(BaseVQA):
+    """流式视频问答 - 支持时间窗口的增量编码"""
+    
+    def __call__(self, video_sample):
+        """处理流式视频样本 - 增量编码"""
+        video = self._load_stream_video(video_sample['video_path'])
+        video_tensor = self._to_tensor(video)
+        
+        # 初始化
+        self.model.clear_cache()
+        self.model.encode_init_prompt()
+        
+        video_start_idx = 0
+        video_end_idx = 0
+        
+        # 逐问题增量编码
+        for qa in video_sample['conversations']:
+            # 计算需要编码的时间窗口
+            temporal_window = self._get_temporal_window(qa)
+            
+            # 如果需要编码新帧
+            if temporal_window[-1] > video_end_idx:
+                video_end_idx = temporal_window[-1]
+                new_frames = video_tensor[int(video_start_idx):int(video_end_idx)]
+                self.model.encode_video(new_frames)
+                video_start_idx = video_end_idx
+            
+            # 回答问题
+            result = self.answer_single(qa, video_sample['video_id'])
+            self.results.append(result)
+        
+        return self.results
+    
+    def _load_stream_video(self, video_path):
+        """加载流式视频"""
+        if video_path.endswith('.npy'):
+            video = np.load(video_path)
+            num_frames = len(video)
+            fps_ratio = self.args.sample_fps
+            assert fps_ratio <= 1, "sample_fps should <= 1 for .npy files"
+            frame_idx = np.linspace(0, num_frames-1, int(num_frames*fps_ratio), dtype=int)
+            return video[frame_idx]
+        else:
+            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
+            fps = round(vr.get_avg_fps())
+            frame_idx = [i for i in range(0, len(vr), int(fps / self.args.sample_fps))]
+            return vr.get_batch(frame_idx).asnumpy()
+    
+    def _get_temporal_window(self, qa_pair):
+        """获取时间窗口（转换为帧索引）"""
+        start_time = qa_pair.get('start_time', 0)
+        end_time = qa_pair.get('end_time', float('inf'))
+        start_idx = start_time * self.args.sample_fps
+        end_idx = end_time * self.args.sample_fps
+        return [start_idx, end_idx]
+    
+    def answer_single(self, qa_pair, video_id):
+        """回答单个问题 - 流式仅支持开放式问答"""
+        question = qa_pair['question']
+        prompt = self.format_openqa_prompt(question)
+        
+        pred = self.model.question_answering(
+            {"question": question, "prompt": prompt},
+            max_new_tokens=256
+        )
+        
+        return self._format_open_result(pred, qa_pair, video_id)
+
diff --git a/model/video_qa/rekv_stream_vqa.py b/model/video_qa/rekv_stream_vqa.py
deleted file mode 100644
index cf115a9..0000000
--- a/model/video_qa/rekv_stream_vqa.py
+++ /dev/null
@@ -1,70 +0,0 @@
-import torch
-import numpy as np
-from logzero import logger
-from decord import VideoReader, cpu
-
-from video_qa.base import BaseVQA, work
-
-
-class ReKVStreamVQA(BaseVQA):
-    def load_video(self, video_path):
-        if video_path.endswith('.npy'):  # FPS=1
-            video = np.load(video_path)
-            assert self.sample_fps <= 1
-            num_frames = len(video)
-            frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
-            video = video[frame_idx]
-        else:
-            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
-            fps = round(vr.get_avg_fps())
-            frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
-            video = vr.get_batch(frame_idx).asnumpy()
-        return video
-
-    def video_open_qa(self, question, max_new_tokens=1024):
-        input_text = {
-            "question": question,
-            "prompt": self.qa_model.get_prompt(question)
-        }
-        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens)
-
-        return {
-            'pred_answer': pred_answer.replace('\n', ''),
-        }
-
-    @torch.inference_mode()
-    def analyze_a_video(self, video_sample):
-        video_path = video_sample['video_path']
-        video_start_idx = video_end_idx = 0
-        video = self.load_video(video_path)
-        video_tensor = torch.from_numpy(video)
-
-        self.qa_model.clear_cache()
-        self.qa_model.encode_init_prompt()
-
-        for sample in video_sample['conversations']:
-            logger.debug(f'sample: {sample}')
-            question = sample['question']
-            answer = sample['answer']
-
-            temporal_windows = torch.tensor([sample['start_time'], sample['end_time']]) * self.sample_fps
-            temporal_windows = temporal_windows.tolist()
-
-            # encode video until receiving QA
-            if temporal_windows[-1] > video_end_idx:
-                video_end_idx = temporal_windows[-1]
-                self.qa_model.encode_video(video_tensor[int(video_start_idx):int(video_end_idx)])
-                video_start_idx = video_end_idx
-        
-            # OpenQA
-            qa_results = self.video_open_qa(question, max_new_tokens=256)
-            self.record[(self.retrieve_size, self.chunk_size)].append({
-                'video_id': video_sample['video_id'],
-                'question': question,
-                'answer': answer,
-                'pred_answer': qa_results['pred_answer'],
-            })
- 
-
-if __name__ == "__main__":
-    work(ReKVStreamVQA)
diff --git a/models/rekv/model/video_qa/run_distributed.py b/models/rekv/model/video_qa/run_distributed.py
new file mode 100644
index 0000000..c1e3927
--- /dev/null
+++ b/models/rekv/model/video_qa/run_distributed.py
@@ -0,0 +1,169 @@
+"""使用torch.distributed.run的分布式推理 - 使用PyTorch原生gather收集结果"""
+import os
+import argparse
+import torch
+import torch.distributed as dist
+from pathlib import Path
+from tqdm import tqdm
+from logzero import logger
+import pandas as pd
+
+from stc.config import GlobalConfig
+from .configs import DATASETS
+from .utils import (
+    load_and_split_anno, 
+    load_model, 
+    run_evaluation,
+)
+from .solver_factory import create_solver
+
+
+def main():
+    """主函数"""
+    args = parse_args()
+    
+    ###############################################################################
+    # 初始化分布式环境
+    assert torch.cuda.is_available(), "DDP推理需要至少一个GPU"
+    torch.backends.cuda.matmul.allow_tf32 = getattr(args, 'tf32', False)
+    torch.set_grad_enabled(False)
+    
+    # Setup DDP
+    dist.init_process_group("gloo")
+    rank = dist.get_rank()
+    world_size = dist.get_world_size()
+    device = rank % torch.cuda.device_count()
+    
+    # 设置随机种子
+    seed = getattr(args, 'global_seed', 42) * world_size + rank
+    torch.manual_seed(seed)
+    torch.cuda.set_device(device)
+    
+    if rank == 0:
+        logger.info(f"Starting rank={rank}, seed={seed}, world_size={world_size}")
+    dist.barrier()
+    
+    #########################################################################################
+    # 加载配置和数据
+    GlobalConfig.initialize_from_env()
+    dataset_config = DATASETS[args.dataset]
+    
+    # 按rank分配数据
+    anno = load_and_split_anno(
+        dataset_config.anno_path, 
+        world_size=world_size, 
+        rank=rank
+    )
+    
+    # 加载模型
+    
+    model, processor = load_model(
+        args.model,
+        n_local=args.n_local,
+        device=device,
+        topk=args.retrieve_size,
+        chunk_size=args.retrieve_chunk_size
+    )
+    
+    ######################################################################
+    # 同步所有进程
+    dist.barrier()
+    # 运行推理
+    results = run_inference(model, dataset_config,processor, anno, args, rank, world_size)
+    logger.info(f"[Rank {rank}] Gathering results...")
+    all_results = gather_results(results, rank, world_size)
+    
+    # Rank 0 保存合并后的结果并运行评估
+    if rank == 0:
+        save_merged_results(all_results, args.save_dir)
+        run_evaluation(args.save_dir, dataset_config.eval_script)
+    
+    dist.destroy_process_group()
+
+
+
+def run_inference(model, dataset_config, processor, anno, args, rank, world_size):
+    """运行推理并返回结果"""
+    # 使用solver_factory根据配置创建正确的solver
+    solver_name = dataset_config.solver
+
+    
+    vqa = create_solver(solver_name, model, processor, args)
+    
+    # 使用tqdm显示进度
+    desc = f"Processing [Rank {rank}/{world_size}]"
+    for video_sample in tqdm(anno, desc=desc, disable=rank != 0):
+        vqa(video_sample)
+        
+    if rank == 0:
+        logger.info(f"[Rank {rank}] Using solver: {solver_name}")
+        logger.info(f"[Rank {rank}] Processed {len(vqa.results)} samples")
+    return vqa.results  # 直接返回结果列表，不保存CSV
+
+
+def gather_results(results, rank, world_size):
+
+    # 准备接收容器（仅rank 0需要）
+    gathered_results = [None] * world_size if rank == 0 else None
+    # 使用PyTorch原生的gather_object收集Python对象
+    dist.gather_object(
+        obj=results,              # 当前rank的结果
+        object_gather_list=gathered_results,  # 收集容器（仅rank 0有效）
+        dst=0                     # 收集到rank 0
+    )
+    if rank == 0:
+        # 将所有rank的结果扁平化为一个列表
+        all_results = []
+        for rank_results in gathered_results:
+            all_results.extend(rank_results)
+        logger.info(f"Gathered {len(all_results)} total results from {world_size} ranks")
+        return all_results
+    
+    return None
+
+
+def save_merged_results(results, save_dir):
+
+    save_dir = Path(save_dir)
+    save_dir.mkdir(parents=True, exist_ok=True)
+    
+    # 保存为CSV
+    result_file = save_dir / "results.csv"
+    df = pd.DataFrame(results)
+    df.to_csv(result_file, index=False)
+    logger.info(f"Results saved to: {result_file}")
+    logger.info(f"Total samples: {len(results)}")
+
+
+
+def parse_args():
+    """解析参数"""
+    parser = argparse.ArgumentParser(description="分布式视频问答推理")
+    
+    # 必需参数
+    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
+    parser.add_argument("--save_dir", required=True, help="结果保存目录")
+    parser.add_argument("--sample_dir", default="./samples", help="样本保存目录")
+    
+    # 模型参数
+    parser.add_argument("--model", default="llava_ov_7b")
+    parser.add_argument("--n_local", type=int, default=15000)
+    parser.add_argument("--retrieve_size", type=int, default=64)
+    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
+    
+    # 分布式参数
+    parser.add_argument("--global_seed", type=int, default=42)
+    parser.add_argument("--tf32", action="store_true", help="启用TF32加速")
+    
+    # 数据参数
+    parser.add_argument("--sample_fps", type=float, default=0.5)
+    parser.add_argument("--image_size", type=int, default=256)
+    
+    # 调试参数
+    parser.add_argument("--debug", action="store_true")
+    
+    return parser.parse_args()
+
+
+if __name__ == "__main__":
+    main()
diff --git a/model/video_qa/run_eval.py b/model/video_qa/run_eval.py
deleted file mode 100644
index 7d8875f..0000000
--- a/model/video_qa/run_eval.py
+++ /dev/null
@@ -1,276 +0,0 @@
-import os
-import argparse
-import subprocess
-import multiprocessing
-
-
-def exec(cmd, sub=False, device=None):
-    print(f'exec: {cmd}')
-    if not sub:
-        if isinstance(cmd, list):
-            cmd = ' '.join(cmd)
-        os.system(cmd)
-    else:
-        my_env = os.environ.copy()
-        my_env["CUDA_VISIBLE_DEVICES"] = device
-        subprocess.run(cmd, env=my_env)
-
-
-def eval_mlvu(args):
-    num_chunks = args.num_chunks
-    save_dir = f"results/{args.model}/mlvu/{args.retrieve_size}-{args.sample_fps}"
-    solver = "rekv_offline_vqa"
-    if not args.only_eval:
-        # QA
-        processes = []
-        for idx in range(0, num_chunks):
-            cmd = ["python", f"video_qa/{solver}.py",
-                    "--model", args.model,
-                    "--sample_fps", str(args.sample_fps),
-                    "--n_local", str(args.n_local),
-                    "--retrieve_size", str(args.retrieve_size),
-                    "--save_dir", save_dir,
-                    "--anno_path", "data/mlvu/dev_debug_mc.json",
-                    "--debug", args.debug,
-                    "--num_chunks", str(num_chunks),
-                    "--chunk_idx", str(idx)]
-            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
-            processes.append(p)
-            p.start()
-        for p in processes:
-            p.join()
-        # merge results
-        exec(f"> {save_dir}/results.csv")
-        for idx in range(num_chunks):
-            if idx == 0:
-                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
-            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
-            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
-    # eval
-    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")
-
-def eval_qaego4d(args):
-    num_chunks = args.num_chunks
-    save_dir = f"results/{args.model}/qaego4d/{args.retrieve_size}-{args.sample_fps}"
-    solver = "rekv_offline_vqa"
-    if not args.only_eval:
-        # QA
-        processes = []
-        for idx in range(0, num_chunks):
-            cmd = ["python", f"video_qa/{solver}.py",
-                    "--model", args.model,
-                    "--sample_fps", str(args.sample_fps),
-                    "--n_local", str(args.n_local),
-                    "--retrieve_size", str(args.retrieve_size),
-                    "--save_dir", save_dir,
-                    "--anno_path", "data/qaego4d/test_mc.json",
-                    "--debug", args.debug,
-                    "--num_chunks", str(num_chunks),
-                    "--chunk_idx", str(idx)]
-            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
-            processes.append(p)
-            p.start()
-        for p in processes:
-            p.join()
-        # merge results
-        exec(f"> {save_dir}/results.csv")
-        for idx in range(num_chunks):
-            if idx == 0:
-                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
-            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
-            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
-    # eval
-    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")
-
-def eval_egoschema(args):
-    num_chunks = args.num_chunks
-    save_dir = f"results/{args.model}/egoschema/{args.retrieve_size}-{args.sample_fps}"
-    solver = "rekv_offline_vqa"
-    if not args.only_eval:
-        # QA
-        processes = []
-        for idx in range(0, num_chunks):
-            cmd = ["python", f"video_qa/{solver}.py",
-                    "--model", args.model,
-                    "--sample_fps", str(args.sample_fps),
-                    "--n_local", str(args.n_local),
-                    "--retrieve_size", str(args.retrieve_size),
-                    "--save_dir", save_dir,
-                    "--anno_path", "data/egoschema/full.json",
-                    "--debug", args.debug,
-                    "--num_chunks", str(num_chunks),
-                    "--chunk_idx", str(idx)]
-            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
-            processes.append(p)
-            p.start()
-        for p in processes:
-            p.join()
-        # merge results
-        exec(f"> {save_dir}/results.csv")
-        for idx in range(num_chunks):
-            if idx == 0:
-                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
-            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
-            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
-    # eval
-    exec(f"python video_qa/eval/eval_egoschema.py --save_dir {save_dir}")
-
-def eval_activitynet_qa(args):
-    num_chunks = args.num_chunks
-    save_dir = f"results/{args.model}/activitynet_qa/{args.retrieve_size}-{args.sample_fps}"
-    solver = "rekv_offline_vqa"
-    if not args.only_eval:
-        # QA
-        processes = []
-        for idx in range(0, num_chunks):
-            cmd = ["python", f"video_qa/{solver}.py",
-                    "--model", args.model,
-                    "--sample_fps", str(args.sample_fps),
-                    "--n_local", str(args.n_local),
-                    "--retrieve_size", str(args.retrieve_size),
-                    "--save_dir", save_dir,
-                    "--anno_path", "data/activitynet_qa/test.json",
-                    "--debug", args.debug,
-                    "--num_chunks", str(num_chunks),
-                    "--chunk_idx", str(idx)]
-            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
-            processes.append(p)
-            p.start()
-        for p in processes:
-            p.join()
-        # merge results
-        exec(f"> {save_dir}/results.csv")
-        exec(f"rm -rf {save_dir}/tmp")
-        for idx in range(num_chunks):
-            if idx == 0:
-                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
-            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
-            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
-    # eval
-    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")
-
-def eval_rvs_ego(args):
-    num_chunks = args.num_chunks
-    save_dir = f"results/{args.model}/rvs_ego/{args.retrieve_size}-{args.sample_fps}"
-    solver = "rekv_stream_vqa"
-    if not args.only_eval:
-        # QA
-        processes = []
-        for idx in range(0, num_chunks):
-            cmd = ["python", f"video_qa/{solver}.py",
-                    "--model", args.model,
-                    "--sample_fps", str(args.sample_fps),
-                    "--n_local", str(args.n_local),
-                    "--retrieve_size", str(args.retrieve_size),
-                    "--save_dir", save_dir,
-                    "--anno_path", "data/rvs/ego/ego4d_oe.json",
-                    "--debug", args.debug,
-                    "--num_chunks", str(num_chunks),
-                    "--chunk_idx", str(idx)]
-            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
-            processes.append(p)
-            p.start()
-        for p in processes:
-            p.join()
-        # merge results
-        exec(f"> {save_dir}/results.csv")
-        exec(f"rm -rf {save_dir}/tmp")
-        for idx in range(num_chunks):
-            if idx == 0:
-                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
-            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
-            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
-    # eval
-    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")
-
-def eval_rvs_movie(args):
-    num_chunks = args.num_chunks
-    save_dir = f"results/{args.model}/rvs_movie/{args.retrieve_size}-{args.sample_fps}"
-    solver = "rekv_stream_vqa"
-    if not args.only_eval:
-        # QA
-        processes = []
-        for idx in range(0, num_chunks):
-            cmd = ["python", f"video_qa/{solver}.py",
-                    "--model", args.model,
-                    "--sample_fps", str(args.sample_fps),
-                    "--n_local", str(args.n_local),
-                    "--retrieve_size", str(args.retrieve_size),
-                    "--save_dir", save_dir,
-                    "--anno_path", "data/rvs/movie/movienet_oe.json",
-                    "--debug", args.debug,
-                    "--num_chunks", str(num_chunks),
-                    "--chunk_idx", str(idx)]
-            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
-            processes.append(p)
-            p.start()
-        for p in processes:
-            p.join()
-        # merge results
-        exec(f"> {save_dir}/results.csv")
-        exec(f"rm -rf {save_dir}/tmp")
-        for idx in range(num_chunks):
-            if idx == 0:
-                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
-            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
-            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
-    # eval
-    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")
-
-def eval_cgbench(args):
-    num_chunks = args.num_chunks
-    save_dir = f"results/{args.model}/cgbench/{args.retrieve_size}-{args.sample_fps}"
-    solver = "rekv_offline_vqa"
-    if not args.only_eval:
-        # QA
-        processes = []
-        for idx in range(0, num_chunks):
-            cmd = ["python", f"video_qa/{solver}.py",
-                    "--model", args.model,
-                    "--sample_fps", str(args.sample_fps),
-                    "--n_local", str(args.n_local),
-                    "--retrieve_size", str(args.retrieve_size),
-                    "--save_dir", save_dir,
-                    "--anno_path", "data/cgbench/full_mc.json",
-                    "--debug", args.debug,
-                    "--num_chunks", str(num_chunks),
-                    "--chunk_idx", str(idx)]
-            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
-            processes.append(p)
-            p.start()
-        for p in processes:
-            p.join()
-        # merge results
-        exec(f"> {save_dir}/results.csv")
-        for idx in range(num_chunks):
-            if idx == 0:
-                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
-            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
-            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
-    # eval
-    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")
-
-
-if __name__ == "__main__":
-    parser = argparse.ArgumentParser()
-    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b', 'video_llava_7b', 'longva_7b'])
-    parser.add_argument("--dataset", type=str, default=None, choices=['mlvu', 'qaego4d', 'egoschema', 'activitynet_qa', 'rvs_ego', 'rvs_movie', 'cgbench'])
-    parser.add_argument("--num_chunks", type=int, default=1)
-    parser.add_argument("--only_eval", action="store_true")
-    parser.add_argument("--sample_fps", type=float, default=1)
-    parser.add_argument("--n_local", type=int, default=15000)
-    parser.add_argument("--retrieve_size", type=int, default=64)
-    parser.add_argument("--debug", type=str, default='false')
-    args = parser.parse_args()
-    func_dic = {
-        'mlvu': eval_mlvu,
-        'qaego4d': eval_qaego4d,
-        'egoschema': eval_egoschema,
-        'activitynet_qa': eval_activitynet_qa,
-        'rvs_ego': eval_rvs_ego,
-        'rvs_movie': eval_rvs_movie,
-        'cgbench': eval_cgbench,
-    }
-    if args.dataset in func_dic:
-        print(f'Execute {args.dataset} evaluation')
-        func_dic[args.dataset](args)
diff --git a/models/rekv/model/video_qa/solver_factory.py b/models/rekv/model/video_qa/solver_factory.py
new file mode 100644
index 0000000..c728537
--- /dev/null
+++ b/models/rekv/model/video_qa/solver_factory.py
@@ -0,0 +1,26 @@
+
+from logzero import logger
+
+
+def create_solver(solver_name, model, processor, args):
+
+    # 延迟导入避免循环依赖
+    from .rekv_offline_refactored import ReKVOfflineVQA
+    from .rekv_stream_refactored import ReKVStreamVQA
+    from .videomme_refactored import VideoMMEReKVOfflineVQA
+    
+    # Solver映射表
+    SOLVER_MAP = {
+        'rekv_offline_vqa': ReKVOfflineVQA,
+        'videomme_rekv_offline_vqa': VideoMMEReKVOfflineVQA,
+        'rekv_stream_vqa': ReKVStreamVQA,
+    }
+    
+    if solver_name not in SOLVER_MAP:
+        logger.warning(f"Unknown solver: {solver_name}, falling back to rekv_offline_vqa")
+        solver_name = 'rekv_offline_vqa'
+    
+    solver_class = SOLVER_MAP[solver_name]
+    
+    return solver_class(model, processor, args)
+
diff --git a/models/rekv/model/video_qa/utils/__init__.py b/models/rekv/model/video_qa/utils/__init__.py
new file mode 100644
index 0000000..1443eb1
--- /dev/null
+++ b/models/rekv/model/video_qa/utils/__init__.py
@@ -0,0 +1,14 @@
+"""工具模块 - 提供数据加载、模型加载和结果合并功能"""
+from .data_utils import load_and_split_anno, chunk_video
+from .model_utils import load_model, get_device
+from .merge_utils import run_evaluation
+
+__all__ = [
+    'load_and_split_anno',
+    'chunk_video',
+    'load_model',
+    'get_device',
+    
+    'run_evaluation',
+]
+
diff --git a/models/rekv/model/video_qa/utils/data_utils.py b/models/rekv/model/video_qa/utils/data_utils.py
new file mode 100644
index 0000000..7999dba
--- /dev/null
+++ b/models/rekv/model/video_qa/utils/data_utils.py
@@ -0,0 +1,44 @@
+"""数据加载和处理工具"""
+import json
+import math
+import numpy as np
+import torch
+from decord import VideoReader, cpu
+from logzero import logger
+
+
+def load_and_split_anno(anno_path, world_size, rank):
+    """加载并分割标注数据 - PyTorch DistributedSampler推理风格
+    
+    使用间隔索引方式(strided indexing)分配数据，这是torch.utils.data.distributed.DistributedSampler
+    在推理模式下的标准做法。每个rank获取 [rank, rank+world_size, rank+2*world_size, ...]
+    
+    优点:
+    - 数据自动打散，负载更均衡
+    - 无数据重复，适合推理任务
+    - 符合PyTorch生态标准
+    
+    Args:
+        anno_path: 标注文件路径
+        world_size: 总进程数
+        rank: 当前进程编号
+    
+    Returns:
+        分配给当前进程的标注数据列表
+    """
+    with open(anno_path, 'r') as f:
+        anno = json.load(f)
+    
+    # 使用间隔索引: 每个rank取 [rank::world_size]
+    # 例如: rank0取[0,3,6...], rank1取[1,4,7...], rank2取[2,5,8...]
+    return anno[rank::world_size]
+
+
+
+
+
+def chunk_video(video, chunk_size):
+    """将视频分块"""
+    num_frames = video.shape[0]
+    for i in range(0, num_frames, chunk_size):
+        yield video[i:i + chunk_size]
\ No newline at end of file
diff --git a/models/rekv/model/video_qa/utils/merge_utils.py b/models/rekv/model/video_qa/utils/merge_utils.py
new file mode 100644
index 0000000..4300292
--- /dev/null
+++ b/models/rekv/model/video_qa/utils/merge_utils.py
@@ -0,0 +1,19 @@
+"""
+结果合并和评估工具
+
+⚠️ 警告: merge_results() 函数已弃用！
+推荐使用 PyTorch 原生的 torch.distributed.gather_object 方法
+详见: docs/distributed.md
+"""
+import os
+import subprocess
+import warnings
+from pathlib import Path
+from logzero import logger
+
+def run_evaluation(save_dir, eval_script):
+    """运行评估脚本"""
+    logger.info(f"Running evaluation: {eval_script}")
+    cmd = f"python {eval_script} --save_dir {save_dir}"
+    subprocess.run(cmd, shell=True, check=True)
+
diff --git a/models/rekv/model/video_qa/utils/model_utils.py b/models/rekv/model/video_qa/utils/model_utils.py
new file mode 100644
index 0000000..7627529
--- /dev/null
+++ b/models/rekv/model/video_qa/utils/model_utils.py
@@ -0,0 +1,89 @@
+"""模型加载和管理工具"""
+import os
+from pathlib import Path
+
+import torch
+from logzero import logger
+from model import llava_onevision_rekv, video_llava_rekv, longva_rekv
+
+
+PROJECT_ROOT = Path(__file__).resolve().parents[3]
+MODEL_ZOO = PROJECT_ROOT / "model_zoo"
+HF_HOME = Path(os.environ.get("HF_HOME", "/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face"))
+
+
+def _latest_snapshot(repo_cache_name: str) -> str | None:
+    snapshots = HF_HOME / "hub" / repo_cache_name / "snapshots"
+    if not snapshots.exists():
+        return None
+    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
+    if not candidates:
+        return None
+    return str(max(candidates, key=lambda p: p.stat().st_mtime))
+
+
+def _model_path(env_name: str, local_name: str, repo_cache_name: str, repo_id: str) -> str:
+    env_path = os.environ.get(env_name)
+    if env_path:
+        return env_path
+    local_path = MODEL_ZOO / local_name
+    if local_path.exists():
+        return str(local_path)
+    snapshot = _latest_snapshot(repo_cache_name)
+    if snapshot:
+        return snapshot
+    return repo_id
+
+
+# 模型配置映射
+MODEL_REGISTRY = {
+    'llava_ov_0.5b': {
+        'load_func': llava_onevision_rekv.load_model,
+        'model_path': _model_path(
+            'REKV_LLAVA_OV_05B_PATH',
+            'llava-onevision-qwen2-0.5b-ov-hf',
+            'models--llava-hf--llava-onevision-qwen2-0.5b-ov-hf',
+            'llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
+        ),
+    },
+    'llava_ov_7b': {
+        'load_func': llava_onevision_rekv.load_model,
+        'model_path': _model_path(
+            'REKV_LLAVA_OV_7B_PATH',
+            'llava-onevision-qwen2-7b-ov-hf',
+            'models--llava-hf--llava-onevision-qwen2-7b-ov-hf',
+            'llava-hf/llava-onevision-qwen2-7b-ov-hf',
+        ),
+    },
+    'video_llava_7b': {
+        'load_func': video_llava_rekv.load_model,
+        'model_path': '/mnt/data2/huggingface/hub/models--LanguageBind--Video-LLaVA-7B-hf/snapshots/4cf9d8cfc76a54f46a4cb43be5368b46b7f0d736',
+    },
+    'longva_7b': {
+        'load_func': longva_rekv.load_model,
+        'model_path': '/data/wangyiyu-20250922/LongVA-7B',
+    },
+}
+
+
+def load_model(model_name, device, n_local=15000, topk=64, chunk_size=1):
+    """加载视频问答模型"""
+    if model_name not in MODEL_REGISTRY:
+        raise ValueError(f"Unknown model: {model_name}")
+    config = MODEL_REGISTRY[model_name]
+    model, processor = config['load_func'](
+        model_path=config['model_path'],
+        device=device,  
+        n_local=n_local,
+        topk=topk,
+        chunk_size=chunk_size,
+    )
+    
+    return model, processor
+
+
+def get_device():
+    """获取可用设备"""
+    if torch.cuda.is_available():
+        return torch.device('cuda')
+    return torch.device('cpu')
diff --git a/models/rekv/model/video_qa/videomme_refactored.py b/models/rekv/model/video_qa/videomme_refactored.py
new file mode 100644
index 0000000..b53c4b4
--- /dev/null
+++ b/models/rekv/model/video_qa/videomme_refactored.py
@@ -0,0 +1,74 @@
+"""VideoMME专用solver - 支持时间和内存统计"""
+import torch
+from logzero import logger
+from .rekv_offline_refactored import ReKVOfflineVQA
+
+
+class VideoMMEReKVOfflineVQA(ReKVOfflineVQA):
+    """VideoMME数据集专用 - 增加时间/内存统计"""
+    
+    def __init__(self, model, processor, args):
+        super().__init__(model, processor, args)
+        # 初始化统计变量
+        self.acc_time = 0.0
+        self.max_mem = 0.0
+    
+    def encode_video(self, video):
+        """编码视频 - 增加性能统计"""
+        self.model.clear_cache()
+        self.model.encode_init_prompt()
+        
+        # 开始性能监控
+        torch.cuda.reset_peak_memory_stats()
+        torch.cuda.synchronize()
+        
+        gpu_start_event = torch.cuda.Event(enable_timing=True)
+        gpu_end_event = torch.cuda.Event(enable_timing=True)
+        gpu_start_event.record()
+        
+        # 执行编码
+        self.model.encode_video(video)
+        
+        # 结束性能监控
+        gpu_end_event.record()
+        torch.cuda.synchronize()
+        
+        # 记录统计信息
+        gpu_time = gpu_start_event.elapsed_time(gpu_end_event) / 1000.0
+        gen_max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
+        
+        self.acc_time += gpu_time
+        self.max_mem = max(gen_max_mem, self.max_mem)
+        
+        logger.debug(f"Video encoding: {gpu_time:.2f}s, mem: {gen_max_mem:.1f}MB")
+        logger.debug(f"Accumulated time: {self.acc_time:.2f}s, max mem: {self.max_mem:.1f}MB")
+    
+    def answer_questions(self, video_sample):
+        """批量回答问题 - 重置统计"""
+        # 每个视频重置统计
+        self.acc_time = 0.0
+        self.max_mem = 0.0
+        
+        # 保存视频级别的信息（如duration）
+        self.current_video_info = {
+            'duration': video_sample.get('duration'),
+        }
+        
+        return super().answer_questions(video_sample)
+    
+    def _format_mc_result(self, pred, qa_pair, video_id):
+        """格式化多选题结果 - 添加duration字段和正确选项处理"""
+        result = super()._format_mc_result(pred, qa_pair, video_id)
+        
+        # VideoMME特殊处理：answer直接是选项字母（如'A','B'），不需要转换
+        answer = qa_pair.get('answer')
+        if answer and answer in self.choice_letters:
+            result['correct_choice'] = answer
+            result['qa_acc'] = float(result['pred_choice'] == answer) * 100
+        
+        # 添加视频duration字段
+        if hasattr(self, 'current_video_info') and self.current_video_info.get('duration'):
+            result['duration'] = self.current_video_info['duration']
+        
+        return result
+
```
