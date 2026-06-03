# StreamForest Integration Changes

This document records the local reproducibility changes made for
`models/StreamForest`. The goal is to make the freshly cloned upstream
StreamForest code runnable inside the STC workspace without editing absolute
paths by hand.

## What Changed And Why

| Area | Files | Why | What changed |
| --- | --- | --- | --- |
| Runtime path entrypoint | `scripts/env/streamforest_env.sh` | Upstream scripts had hardcoded local paths and no single source of truth for data/checkpoint/output locations. | Added derived `STREAMFOREST_ROOT`, `HF_HOME`, `STREAMFOREST_DATA_ROOT`, `STREAMFOREST_OUTPUT_DIR`, checkpoint defaults, `PYTHONPATH`, and automatic discovery of the official `MCG-NJU/StreamForest-Annodata` HF snapshot. |
| Dataset loading | `lmms_eval/api/task.py` | Upstream expected `anno/eval` local datasets or HF loading, while our annotations are local JSON files/snapshots. | Added local JSON/dataset loading from `STREAMFOREST_ANNO_ROOT`, explicit broken-symlink errors, and support for both parent `.../eval` and direct benchmark roots such as `.../eval/OVOBench`. |
| Video path resolution | `lmms_eval/tasks/_task_utils/streamforest_paths.py`, task `utils.py` files under `ovobench`, `streamingbench`, `videomme`, `mlvu_mc`, `mvbench`, `odvbench`, `ovbench_full` | Upstream task code mixed empty cache paths with remote `s3://` paths, so local evaluation could not reliably find videos. | Added a shared resolver that checks environment overrides and known local layouts under `$HF_HOME`, then falls back to the original remote-style path. |
| Eval launch | `scripts/eval/run_eval.sh`, `scripts/eval/online/*.sh`, `scripts/eval/others/eval_internvl2-8B.sh` | Upstream launch scripts assumed Slurm and hardcoded project/checkpoint/output paths. | Made tasks, model, checkpoint, output dir, max frames, GPU count, Slurm usage, partition, and Python executable configurable through environment variables. Direct `accelerate` launch is now the default; Slurm is opt-in with `STREAMFOREST_USE_SLURM=1`. |
| Smoke test | `scripts/eval/run_smoke.sh` | Repro needs a cheap one-command sanity check before full benchmark runs. | Added a one-sample OVO-Bench smoke entry with defaults `TASKS=ovobench_backward_tracking`, `LIMIT=1`, `NUM_GPUS=1`, `MAX_FRAMES=8`. |
| Environment setup | `scripts/setup/create_streamforest_env.sh` | The container has a verified reusable venv and poor network reproducibility. | Added a setup helper that reuses `/apdcephfs_tj5/share_303570626/yiyuwang/envs/lmms-streamforest-py312-tf446` by default and only creates/installs when explicitly requested. |
| Optional imports | `llava/__init__.py`, `lmms_eval/api/metrics.py` | Eval failed on training-only or optional packages such as `apex`, `sacrebleu`, and `sklearn`. | Deferred training imports unless `LLAVA_IMPORT_TRAINING=1`; made optional metric dependencies lazy. |
| Training launch scripts | `scripts/train/stage*/*.sh` | Training scripts also inherited hardcoded roots. | Source the shared env file and use configurable checkpoint/output roots while preserving the original Slurm-oriented behavior. |
| Reproduction docs | `docs/streamforest/REPRODUCE.md`, `docs/streamforest/CHANGES.md` | Others need to know how to run the patched clone and what was changed. | Added a reproducibility guide and this change log. These docs live outside upstream StreamForest, so they are not included in the upstream diff below. |

## Current Verified Layout

The default runtime paths now resolve to:

```bash
HF_HOME=/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face
STREAMFOREST_DATA_ROOT=$HF_HOME
STREAMFOREST_CKPT_PATH=$HF_HOME/StreamForest-Qwen2-7B
STREAMFOREST_ANNO_ROOT=$HF_HOME/hub/datasets--MCG-NJU--StreamForest-Annodata/snapshots/ca769c269aee198d9ee6735d85a8c39a699e5833/eval
```

OVO-Bench smoke was verified in the Taiji container with:

```bash
cd /apdcephfs_tj5/share_303570626/yiyuwang/work_space/STC_new/models/StreamForest
source /apdcephfs_tj5/share_303570626/yiyuwang/envs/lmms-streamforest-py312-tf446/bin/activate
STREAMFOREST_OUTPUT_DIR=/tmp/streamforest-run-smoke-default \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval/run_smoke.sh
```

It loaded the official snapshot JSON, resolved
`$HF_HOME/OVO-Bench/chunked_videos/0.mp4`, loaded
`$HF_HOME/StreamForest-Qwen2-7B`, and completed one generate request.

## Git Diff Against Upstream StreamForest

Generated from `git -C models/StreamForest diff` plus new-file diffs for
untracked integration files.

```diff
diff --git a/llava/__init__.py b/llava/__init__.py
index 86da520..f1904e0 100644
--- a/llava/__init__.py
+++ b/llava/__init__.py
@@ -1,2 +1,6 @@
+import os
+
 from .model import LlavaQwenForCausalLM
-from .train.train import LazySupervisedDataset, DataCollatorForSupervisedDataset
\ No newline at end of file
+
+if os.environ.get("LLAVA_IMPORT_TRAINING", "0") == "1":
+    from .train.train import LazySupervisedDataset, DataCollatorForSupervisedDataset
diff --git a/lmms_eval/api/metrics.py b/lmms_eval/api/metrics.py
index 3f2d33e..7d1f2a4 100644
--- a/lmms_eval/api/metrics.py
+++ b/lmms_eval/api/metrics.py
@@ -2,8 +2,6 @@
 from collections.abc import Iterable
 
 import numpy as np
-import sacrebleu
-import sklearn.metrics
 import random
 import evaluate
 import torch
@@ -12,6 +10,22 @@
 from loguru import logger as eval_logger
 
 
+def _require_sacrebleu():
+    try:
+        import sacrebleu
+    except ImportError as exc:
+        raise ImportError("sacrebleu is required only for BLEU/chrF/TER metrics.") from exc
+    return sacrebleu
+
+
+def _require_sklearn_metrics():
+    try:
+        import sklearn.metrics
+    except ImportError as exc:
+        raise ImportError("scikit-learn is required only for F1/Matthews metrics.") from exc
+    return sklearn.metrics
+
+
 # Register Aggregations First
 @register_aggregation("bypass")
 def bypass_agg(arr):
@@ -49,21 +63,23 @@ def bits_per_byte(items):
 
 @register_aggregation("f1")
 def f1_score(items):
+    sklearn_metrics = _require_sklearn_metrics()
     unzipped_list = list(zip(*items))
     golds = unzipped_list[0]
     preds = unzipped_list[1]
-    fscore = sklearn.metrics.f1_score(golds, preds)
+    fscore = sklearn_metrics.f1_score(golds, preds)
 
     return np.max(fscore)
 
 
 @register_aggregation("matthews_corrcoef")
 def matthews_corrcoef(items):
+    sklearn_metrics = _require_sklearn_metrics()
     unzipped_list = list(zip(*items))
     golds = unzipped_list[0]
     preds = unzipped_list[1]
     # print(preds)
-    return sklearn.metrics.matthews_corrcoef(golds, preds)
+    return sklearn_metrics.matthews_corrcoef(golds, preds)
 
 
 @register_aggregation("bleu")
@@ -81,6 +97,7 @@ def bleu(items):
     refs = list(zip(*items))[0]
     preds = list(zip(*items))[1]
     refs, preds = _sacreformat(refs, preds)
+    sacrebleu = _require_sacrebleu()
     return sacrebleu.corpus_bleu(preds, refs).score
 
 
@@ -96,6 +113,7 @@ def chrf(items):
     refs = list(zip(*items))[0]
     preds = list(zip(*items))[1]
     refs, preds = _sacreformat(refs, preds)
+    sacrebleu = _require_sacrebleu()
     return sacrebleu.corpus_chrf(preds, refs).score
 
 
@@ -112,6 +130,7 @@ def ter(items):
     refs = list(zip(*items))[0]
     preds = list(zip(*items))[1]
     refs, preds = _sacreformat(refs, preds)
+    sacrebleu = _require_sacrebleu()
     return sacrebleu.corpus_ter(preds, refs).score
 
 
diff --git a/lmms_eval/api/task.py b/lmms_eval/api/task.py
index b908e7e..c673519 100644
--- a/lmms_eval/api/task.py
+++ b/lmms_eval/api/task.py
@@ -12,6 +12,7 @@
 from collections.abc import Callable
 from dataclasses import dataclass, field, asdict
 from glob import glob
+from pathlib import Path
 from typing import Any, List, Union
 
 import datasets
@@ -712,6 +713,50 @@ def _prepare_metric_and_aggregation(self):
                     eval_logger.warning(f"[Task: {self._config.task}] metric {metric_name} is defined, but higher_is_better is not. " f"using default " f"higher_is_better={is_higher_better(metric_name)}")
                     self._higher_is_better[metric_name] = is_higher_better(metric_name)
 
+    def _try_load_from_disk(self):
+        if not self.DATASET_PATH:
+            return None
+
+        root = Path(os.environ.get("STREAMFOREST_ROOT", os.getcwd()))
+        anno_root = Path(os.environ.get("STREAMFOREST_ANNO_ROOT", root / "anno" / "eval")).expanduser()
+        dataset_paths = []
+        raw_path = Path(self.DATASET_PATH).expanduser()
+
+        raw_parts = raw_path.parts
+        if not raw_path.is_absolute() and len(raw_parts) >= 3 and raw_parts[0] == "anno" and raw_parts[1] == "eval":
+            anno_tail = raw_parts[2:]
+            if anno_tail and anno_root.name == anno_tail[0]:
+                dataset_paths.append(anno_root.joinpath(*anno_tail[1:]))
+            else:
+                dataset_paths.append(anno_root.joinpath(*anno_tail))
+
+        dataset_paths.append(raw_path)
+        if not raw_path.is_absolute():
+            dataset_paths.append(root / raw_path)
+
+        candidates = []
+        for path in dataset_paths:
+            if self.DATASET_NAME:
+                json_path = path / "json" / f"{self.DATASET_NAME}.json"
+                if json_path.is_symlink() and not json_path.exists():
+                    raise FileNotFoundError(f"Broken dataset symlink: {json_path} -> {os.readlink(json_path)}")
+                if json_path.exists():
+                    split = self.config.test_split or self.config.validation_split or self.config.training_split or "train"
+                    eval_logger.info(f"Loading local JSON dataset: {json_path}")
+                    return datasets.load_dataset("json", data_files={split: str(json_path)})
+                candidates.append(path / self.DATASET_NAME)
+            candidates.append(path)
+
+        for candidate in candidates:
+            if (candidate / "dataset_dict.json").exists() or (candidate / "state.json").exists():
+                eval_logger.info(f"Loading local dataset from disk: {candidate}")
+                dataset = datasets.load_from_disk(str(candidate))
+                if isinstance(dataset, datasets.Dataset):
+                    split = self.config.test_split or self.config.validation_split or self.config.training_split or "train"
+                    dataset = datasets.DatasetDict({split: dataset})
+                return dataset
+        return None
+
     @retry(stop=(stop_after_attempt(5) | stop_after_delay(60)), wait=wait_fixed(2))
     def download(self, dataset_kwargs=None) -> None:
         # If the dataset is a video dataset,
@@ -720,7 +765,8 @@ def download(self, dataset_kwargs=None) -> None:
         download_config.max_retries = dataset_kwargs.get("max_retries", 10) if dataset_kwargs is not None else 10
         download_config.num_proc = dataset_kwargs.get("num_proc", 8) if dataset_kwargs is not None else 8
         download_config.local_files_only = dataset_kwargs.get("local_files_only", True) if dataset_kwargs is not None else True # NOTE 默认用本地
-        if dataset_kwargs is not None: # NOTE lxh
+        self.dataset = self._try_load_from_disk()
+        if self.dataset is None and dataset_kwargs is not None: # NOTE lxh
             if "From_YouTube" in dataset_kwargs:
                 raise NotImplementedError("I don't want it!")
                 def _download_from_youtube(path):
@@ -881,14 +927,15 @@ def concat_tar_parts(tar_parts, output_tar):
         #     **dataset_kwargs if dataset_kwargs is not None else {},
         # )
 
-        self.dataset = datasets.load_dataset(
-            path=self.DATASET_PATH,
-            name=self.DATASET_NAME,
-            # local_files_only=True
-            # download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
-            # download_config=download_config,
-            # **dataset_kwargs if dataset_kwargs is not None else {},
-        )
+        if self.dataset is None:
+            self.dataset = datasets.load_dataset(
+                path=self.DATASET_PATH,
+                name=self.DATASET_NAME,
+                # local_files_only=True
+                # download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
+                # download_config=download_config,
+                # **dataset_kwargs if dataset_kwargs is not None else {},
+            )
 
         if self.config.process_docs is not None:
             for split in self.dataset:
diff --git a/lmms_eval/tasks/mlvu_mc/utils.py b/lmms_eval/tasks/mlvu_mc/utils.py
index d8af380..b4cb9c3 100644
--- a/lmms_eval/tasks/mlvu_mc/utils.py
+++ b/lmms_eval/tasks/mlvu_mc/utils.py
@@ -11,6 +11,7 @@
 import PIL
 import numpy as np
 from loguru import logger as eval_logger
+from lmms_eval.tasks._task_utils.streamforest_paths import resolve_benchmark_video
 
 import io
 try:
@@ -45,21 +46,16 @@
 cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
 
 
-def mlvu_mc_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
+def _resolve_video_path(doc, lmms_eval_specific_kwargs=None):
     dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path, candidates = resolve_benchmark_video("mlvu_mc", dataset_folder, doc["video"], cache_name)
+    if not os.path.exists(video_path) and "s3://" not in video_path:
+        eval_logger.error(f"Video path: {video_path} does not exist. Checked candidates: {candidates[:8]}")
+    return video_path
+
+
+def mlvu_mc_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     if "start" in doc:
         start, end = doc['start'], doc['end']
@@ -73,20 +69,7 @@ def mlvu_mc_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
     
 
 def mlvu_mc_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
-    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     # frame_image_list = read_frame(video_path)
     if "start" in doc:
diff --git a/lmms_eval/tasks/mvbench/utils.py b/lmms_eval/tasks/mvbench/utils.py
index d5c61a2..f5f647a 100644
--- a/lmms_eval/tasks/mvbench/utils.py
+++ b/lmms_eval/tasks/mvbench/utils.py
@@ -11,6 +11,7 @@
 import PIL
 import numpy as np
 from loguru import logger as eval_logger
+from lmms_eval.tasks._task_utils.streamforest_paths import resolve_benchmark_video
 
 import io
 try:
@@ -58,21 +59,16 @@
 cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
 
 
-def mvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
+def _resolve_video_path(doc, lmms_eval_specific_kwargs=None):
     dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path, candidates = resolve_benchmark_video("mvbench", dataset_folder, doc["video"], cache_name)
+    if not os.path.exists(video_path) and "s3://" not in video_path:
+        eval_logger.error(f"Video path: {video_path} does not exist. Checked candidates: {candidates[:8]}")
+    return video_path
+
+
+def mvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     if "start" in doc:
         start, end = doc['start'], doc['end']
@@ -103,20 +99,7 @@ def mvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
     
 
 def mvbench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
-    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     # frame_image_list = read_frame(video_path)
     if "start" in doc:
diff --git a/lmms_eval/tasks/odvbench/utils.py b/lmms_eval/tasks/odvbench/utils.py
index 8bd9093..05d4784 100644
--- a/lmms_eval/tasks/odvbench/utils.py
+++ b/lmms_eval/tasks/odvbench/utils.py
@@ -11,6 +11,7 @@
 import PIL
 import numpy as np
 from loguru import logger as eval_logger
+from lmms_eval.tasks._task_utils.streamforest_paths import resolve_benchmark_video
 
 import io
 try:
@@ -41,21 +42,16 @@
 cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
 
 
-def odvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
+def _resolve_video_path(doc, lmms_eval_specific_kwargs=None):
     dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path, candidates = resolve_benchmark_video("odvbench", dataset_folder, doc["video"], cache_name)
+    if not os.path.exists(video_path) and "s3://" not in video_path:
+        eval_logger.error(f"Video path: {video_path} does not exist. Checked candidates: {candidates[:8]}")
+    return video_path
+
+
+def odvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     if "start" in doc:
         start, end = doc['start'], doc['end']
@@ -86,20 +82,7 @@ def odvbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
     
 
 def odvbench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
-    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     # frame_image_list = read_frame(video_path)
     if "start" in doc:
diff --git a/lmms_eval/tasks/ovbench_full/utils.py b/lmms_eval/tasks/ovbench_full/utils.py
index 9ca7ab0..1674c84 100644
--- a/lmms_eval/tasks/ovbench_full/utils.py
+++ b/lmms_eval/tasks/ovbench_full/utils.py
@@ -11,6 +11,7 @@
 import PIL
 import numpy as np
 from loguru import logger as eval_logger
+from lmms_eval.tasks._task_utils.streamforest_paths import resolve_benchmark_video
 
 import io
 try:
@@ -43,21 +44,16 @@
 cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
 
 
-def ovbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
+def _resolve_video_path(doc, lmms_eval_specific_kwargs=None):
     dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path, candidates = resolve_benchmark_video("ovbench", dataset_folder, doc["video"], cache_name)
+    if not os.path.exists(video_path) and "s3://" not in video_path:
+        eval_logger.error(f"Video path: {video_path} does not exist. Checked candidates: {candidates[:8]}")
+    return video_path
+
+
+def ovbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     if "start" in doc:
         start, end = doc['start'], doc['end']
@@ -88,20 +84,7 @@ def ovbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
     
 
 def ovbench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
-    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     # frame_image_list = read_frame(video_path)
     if "start" in doc:
diff --git a/lmms_eval/tasks/ovobench/utils.py b/lmms_eval/tasks/ovobench/utils.py
index 3420ef1..7a3bf48 100644
--- a/lmms_eval/tasks/ovobench/utils.py
+++ b/lmms_eval/tasks/ovobench/utils.py
@@ -11,6 +11,7 @@
 import PIL
 import numpy as np
 from loguru import logger as eval_logger
+from lmms_eval.tasks._task_utils.streamforest_paths import resolve_benchmark_video
 
 import io
 try:
@@ -41,21 +42,16 @@
 cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
 
 
-def ovobench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
+def _resolve_video_path(doc, lmms_eval_specific_kwargs=None):
     dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path, candidates = resolve_benchmark_video("ovobench", dataset_folder, doc["video"], cache_name)
+    if not os.path.exists(video_path) and "s3://" not in video_path:
+        eval_logger.error(f"Video path: {video_path} does not exist. Checked candidates: {candidates[:8]}")
+    return video_path
+
+
+def ovobench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     if "start" in doc:
         start, end = doc['start'], doc['end']
@@ -86,20 +82,7 @@ def ovobench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
     
 
 def ovobench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
-    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     # frame_image_list = read_frame(video_path)
     if "start" in doc:
@@ -338,4 +321,4 @@ def ovobench_aggregate_results(results):
         
     print("\n")
     
-    return 100 * total_correct / total_answered if total_answered > 0 else 0
\ No newline at end of file
+    return 100 * total_correct / total_answered if total_answered > 0 else 0
diff --git a/lmms_eval/tasks/streamingbench/utils.py b/lmms_eval/tasks/streamingbench/utils.py
index cf977c1..5f68af3 100644
--- a/lmms_eval/tasks/streamingbench/utils.py
+++ b/lmms_eval/tasks/streamingbench/utils.py
@@ -11,6 +11,7 @@
 import PIL
 import numpy as np
 from loguru import logger as eval_logger
+from lmms_eval.tasks._task_utils.streamforest_paths import resolve_benchmark_video
 
 import io
 try:
@@ -40,24 +41,16 @@
 cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
 
 
-def streamingbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
-    #print('sub_task: ',lmms_eval_specific_kwargs["sub_task"])
+def _resolve_video_path(doc, lmms_eval_specific_kwargs=None):
     dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    #print('data_folder: ',dataset_folder)
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    #print('all_path: ',video_path)
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path, candidates = resolve_benchmark_video("streamingbench", dataset_folder, doc["video"], cache_name)
+    if not os.path.exists(video_path) and "s3://" not in video_path:
+        eval_logger.error(f"Video path: {video_path} does not exist. Checked candidates: {candidates[:8]}")
+    return video_path
+
+
+def streamingbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     if "start" in doc:
         start, end = doc['start'], doc['end']
@@ -88,20 +81,7 @@ def streamingbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
     
 
 def streamingbench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = ""
-    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
-    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
-        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
-        if os.path.exists(alternative_video_path):
-            video_path = alternative_video_path
-        else:
-            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
-    elif "s3://" not in video_path:
-        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
+    video_path = _resolve_video_path(doc, lmms_eval_specific_kwargs)
 
     # frame_image_list = read_frame(video_path)
     if "start" in doc:
diff --git a/lmms_eval/tasks/videomme/utils.py b/lmms_eval/tasks/videomme/utils.py
index 5b4459c..1d0dcf9 100644
--- a/lmms_eval/tasks/videomme/utils.py
+++ b/lmms_eval/tasks/videomme/utils.py
@@ -14,6 +14,7 @@
 import numpy as np
 
 from loguru import logger as eval_logger
+from lmms_eval.tasks._task_utils.streamforest_paths import existing_video_variant, resolve_videomme_path
 
 VIDEO_TYPE = ["short", "medium", "long"]
 CATEGORIES = ["Knowledge", "Film & Television", "Sports Competition", "Artistic Performance", "Life Record", "Multilingual"]
@@ -175,18 +176,10 @@ def extract_subtitles(video_path, subtitle_path):
 
 
 def videomme_doc_to_visual(doc):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = cache_name
-    video_path = doc["videoID"] + ".mp4"
-    video_path = os.path.join(cache_dir, video_path)
-    if os.path.exists(video_path):
-        video_path = video_path
-    elif os.path.exists(video_path.replace("mp4", "MP4")):
-        video_path = video_path.replace("mp4", "MP4")
-    elif os.path.exists(video_path.replace("mp4", "mkv")):
-        video_path = video_path.replace("mp4", "mkv")
-    elif 's3://' not in video_path:
-        sys.exit(f"video path:{video_path} does not exist, please check")
+    video_path, candidates = resolve_videomme_path(doc["videoID"] + ".mp4", cache_name)
+    video_path = existing_video_variant(video_path)
+    if not os.path.exists(video_path) and "s3://" not in video_path:
+        sys.exit(f"video path:{video_path} does not exist, checked candidates: {candidates[:8]}")
     return [video_path]
 
 
@@ -218,11 +211,11 @@ def videomme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
 
 
 def videomme_doc_to_text_subtitle(doc, lmms_eval_specific_kwargs=None):
-    # cache_dir = os.path.join(base_cache_dir, cache_name)
-    cache_dir = cache_name
-    video_path = doc["videoID"] + ".mp4"
-    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
-    video_path = os.path.join(cache_dir, video_path)
+    subtitle_path, _ = resolve_videomme_path(os.path.join("subtitle", doc["videoID"] + ".srt"), cache_name)
+    video_path, candidates = resolve_videomme_path(doc["videoID"] + ".mp4", cache_name)
+    video_path = existing_video_variant(video_path)
+    if not os.path.exists(video_path) and "s3://" not in video_path:
+        eval_logger.error(f"Video path: {video_path} does not exist. Checked candidates: {candidates[:8]}")
     if os.path.exists(subtitle_path):  # Denote have subtitle
         subtitle = open(subtitle_path).readlines()
     else:
diff --git a/scripts/eval/online/eval_online_template.sh b/scripts/eval/online/eval_online_template.sh
old mode 100644
new mode 100755
index d93ab21..44f4576
--- a/scripts/eval/online/eval_online_template.sh
+++ b/scripts/eval/online/eval_online_template.sh
@@ -1,8 +1,14 @@
-CKPT_PATH=""
-MAX_NUM_FRAMES=""
-TASK=""
-MODEL_NAME=""
-TIME_MSG="short_online"
+#!/usr/bin/env bash
+set -euo pipefail
+
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+source "${SCRIPT_DIR}/../../env/streamforest_env.sh"
+
+CKPT_PATH="${STREAMFOREST_CKPT_PATH}"
+MAX_NUM_FRAMES="${MAX_FRAMES:-2048}"
+TASK="${TASK:-}"
+MODEL_NAME="${MODEL_NAME:-streamforest}"
+TIME_MSG="${TIME_MSG:-short_online}"
 
 while [[ $# -gt 0 ]]; do
   case "$1" in
@@ -10,45 +16,78 @@ while [[ $# -gt 0 ]]; do
     --max_frames) MAX_NUM_FRAMES="$2"; shift 2 ;;
     --task) TASK="$2"; shift 2 ;;
     --model_name) MODEL_NAME="$2"; shift 2 ;;
-    --time_msg) TIME_MSG="$2"; shift 2 ;;  # 新增 time_msg 参数
-    *) echo "未知参数: $1"; exit 1 ;;
+    --time_msg) TIME_MSG="$2"; shift 2 ;;
+    *) echo "Unknown argument: $1"; exit 1 ;;
   esac
 done
 
+if [[ -z "${TASK}" ]]; then
+  echo "Missing --task or TASK env var" >&2
+  exit 1
+fi
 
-
-root_path="/your_local_path_to/StreamForest"
-export PYTHONPATH=$root_path
-export HF_DATASETS_OFFLINE=1
-MASTER_PORT=$((18000 + $RANDOM % 100))
-NUM_GPUS=8
-CONV_TEMPLATE=qwen_2
+MASTER_PORT="${MASTER_PORT:-$((18000 + RANDOM % 1000))}"
+NUM_GPUS="${NUM_GPUS:-1}"
+CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_2}"
 TASK_SUFFIX="${TASK//,/_}"
-mkdir ${CKPT_PATH}/eval
-JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M")
-
-echo "检查点路径: $CKPT_PATH"
-echo "最大帧数: $MAX_NUM_FRAMES"
-echo "任务: $TASK"
-echo "模型名称: $MODEL_NAME"
-echo "提示词类型: $TIME_MSG"
-
-srun -p videop1 \
-    --job-name=${JOB_NAME} \
-    --ntasks=1 \
-    --gres=gpu:${NUM_GPUS} \
-    --ntasks-per-node=1 \
-    --cpus-per-task=16 \
-    --kill-on-bad-exit=1 \
-    accelerate launch --num_processes ${NUM_GPUS} --main_process_port 10078 -m lmms_eval \
-        --model ${MODEL_NAME} \
-        --model_args pretrained=$CKPT_PATH,conv_template=$CONV_TEMPLATE,max_frames_num=$MAX_NUM_FRAMES,time_msg=$TIME_MSG \
-        --tasks $TASK \
-        --batch_size 1 \
-        --log_samples \
-        --log_samples_suffix $TASK_SUFFIX \
-        --output_path ${CKPT_PATH}/eval/response__${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME} \
-        2>&1 | tee ${CKPT_PATH}/eval/log_${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}.log
+JOB_NAME="$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")"
+CKPT_TAG="$(basename "${CKPT_PATH}")"
+RUN_OUTPUT_DIR="${STREAMFOREST_OUTPUT_DIR}/eval/${CKPT_TAG}"
+mkdir -p "${RUN_OUTPUT_DIR}"
+
+MODEL_ARGS="pretrained=${CKPT_PATH},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_NUM_FRAMES},time_msg=${TIME_MSG}"
+OUTPUT_PATH="${RUN_OUTPUT_DIR}/response__${TASK_SUFFIX}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}"
+LOG_PATH="${RUN_OUTPUT_DIR}/log_${TASK_SUFFIX}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}.log"
+
+LIMIT_ARGS=()
+if [[ -n "${LIMIT:-}" ]]; then
+  LIMIT_ARGS=(--limit "${LIMIT}")
+fi
+
+EXTRA_ARGS=()
+if [[ -n "${LMMS_EVAL_EXTRA_ARGS:-}" ]]; then
+  read -r -a EXTRA_ARGS <<< "${LMMS_EVAL_EXTRA_ARGS}"
+fi
+
+LAUNCH_PREFIX=()
+if [[ "${STREAMFOREST_USE_SLURM:-0}" == "1" ]]; then
+  PARTITION="${PARTITION:-videop1}"
+  CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
+  LAUNCH_PREFIX=(
+    srun -p "${PARTITION}"
+    --job-name="${JOB_NAME}"
+    --ntasks=1
+    --gres="gpu:${NUM_GPUS}"
+    --ntasks-per-node=1
+    --cpus-per-task="${CPUS_PER_TASK}"
+    --kill-on-bad-exit=1
+  )
+fi
 
+CMD=(
+  "${PYTHON_EXECUTABLE:-python}"
+  -m accelerate.commands.launch
+  --num_processes "${NUM_GPUS}"
+  --main_process_port "${MASTER_PORT}"
+  -m lmms_eval
+  --model "${MODEL_NAME}"
+  --model_args "${MODEL_ARGS}"
+  --tasks "${TASK}"
+  --batch_size "${BATCH_SIZE:-1}"
+  --log_samples
+  --log_samples_suffix "${TASK_SUFFIX}"
+  --output_path "${OUTPUT_PATH}"
+  "${LIMIT_ARGS[@]}"
+  "${EXTRA_ARGS[@]}"
+)
 
+echo "Checkpoint: ${CKPT_PATH}"
+echo "Max frames: ${MAX_NUM_FRAMES}"
+echo "Task: ${TASK}"
+echo "Model: ${MODEL_NAME}"
+echo "Time message: ${TIME_MSG}"
+echo "Python: ${PYTHON_EXECUTABLE:-python}"
+echo "Output: ${OUTPUT_PATH}"
+echo "Log: ${LOG_PATH}"
 
+"${LAUNCH_PREFIX[@]}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
diff --git a/scripts/eval/online/eval_online_template_select_projector.sh b/scripts/eval/online/eval_online_template_select_projector.sh
old mode 100644
new mode 100755
index 630b67c..3122664
--- a/scripts/eval/online/eval_online_template_select_projector.sh
+++ b/scripts/eval/online/eval_online_template_select_projector.sh
@@ -1,51 +1,65 @@
-MAX_NUM_FRAMES="512"
-MODEL_NAME="streamforest"
-TIME_MSG="short_online_v2"
-REPLACE_PROJECTOR="ablation_woSTFW_PEMF"
-CKPT_PATH="/your_local_path_to/StreamForest/ckpt/StreamForest-Qwen2-7B_Siglip_ablation_woPEMF+FSTW"
+#!/usr/bin/env bash
+set -euo pipefail
 
-TASK="ovbench"
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+source "${SCRIPT_DIR}/../../env/streamforest_env.sh"
 
-root_path="/your_local_path_to/StreamForest"
-export PYTHONPATH=$root_path
-export HF_DATASETS_OFFLINE=1
-MASTER_PORT=$((18000 + $RANDOM % 100))
-NUM_GPUS=8
-CONV_TEMPLATE=qwen_2
+MAX_NUM_FRAMES="${MAX_FRAMES:-512}"
+MODEL_NAME="${MODEL_NAME:-streamforest}"
+TIME_MSG="${TIME_MSG:-short_online_v2}"
+REPLACE_PROJECTOR="${REPLACE_PROJECTOR:-ablation_woSTFW_PEMF}"
+CKPT_PATH="${STREAMFOREST_PROJECTOR_CKPT_PATH:-${STREAMFOREST_CKPT_PATH}}"
+TASK="${TASK:-ovbench}"
+CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_2}"
+NUM_GPUS="${NUM_GPUS:-1}"
+MASTER_PORT="${MASTER_PORT:-$((18000 + RANDOM % 1000))}"
 TASK_SUFFIX="${TASK//,/_}"
-mkdir ${CKPT_PATH}/eval
-JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M")
+JOB_NAME="$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")"
+CKPT_TAG="$(basename "${CKPT_PATH}")"
+RESULT_DIR="${STREAMFOREST_OUTPUT_DIR}/eval/${CKPT_TAG}/${MAX_NUM_FRAMES}_${TASK_SUFFIX}_${REPLACE_PROJECTOR}"
+mkdir -p "${RESULT_DIR}"
 
-echo "检查点路径: $CKPT_PATH"
-echo "最大帧数: $MAX_NUM_FRAMES"
-echo "任务: $TASK"
-echo "模型名称: $MODEL_NAME"
-echo "提示词类型: $TIME_MSG"
-echo "记忆类型: $REPLACE_PROJECTOR"
+MODEL_ARGS="pretrained=${CKPT_PATH},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_NUM_FRAMES},time_msg=${TIME_MSG},mm_projector_type=${REPLACE_PROJECTOR}"
+OUTPUT_PATH="${RESULT_DIR}/response__${TASK_SUFFIX}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}"
+LOG_PATH="${RESULT_DIR}/log_${TASK_SUFFIX}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}.log"
 
+LIMIT_ARGS=()
+if [[ -n "${LIMIT:-}" ]]; then
+  LIMIT_ARGS=(--limit "${LIMIT}")
+fi
 
-RESULT_DIR="${CKPT_PATH}/eval/${MAX_NUM_FRAMES}_${TASK}"
-
-if [ ! -d "${RESULT_DIR}" ] && [ -d "${CKPT_PATH}" ]; then
-  mkdir -p ${RESULT_DIR}
-  echo "Created directory: ${RESULT_DIR}"
-else
-    echo "Directory ${RESULT_DIR} already exists or ${CKPT_PATH} not exists."
+LAUNCH_PREFIX=()
+if [[ "${STREAMFOREST_USE_SLURM:-0}" == "1" ]]; then
+  PARTITION="${PARTITION:-videopp1}"
+  CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
+  LAUNCH_PREFIX=(
+    srun -p "${PARTITION}"
+    --job-name="${JOB_NAME}"
+    --ntasks=1
+    --gres="gpu:${NUM_GPUS}"
+    --ntasks-per-node=1
+    --cpus-per-task="${CPUS_PER_TASK}"
+    --kill-on-bad-exit=1
+  )
 fi
 
-srun -p videopp1 \
-    --job-name=${JOB_NAME} \
-    --ntasks=1 \
-    --gres=gpu:8 \
-    --ntasks-per-node=1 \
-    --cpus-per-task=16 \
-    --kill-on-bad-exit=1 \
-    accelerate launch --num_processes ${NUM_GPUS} --main_process_port 10078 -m lmms_eval \
-        --model ${MODEL_NAME} \
-        --model_args pretrained=$CKPT_PATH,conv_template=$CONV_TEMPLATE,max_frames_num=$MAX_NUM_FRAMES,time_msg=$TIME_MSG,mm_projector_type=$REPLACE_PROJECTOR \
-        --tasks $TASK \
-        --batch_size 1 \
-        --log_samples \
-        --log_samples_suffix $TASK_SUFFIX \
-        --output_path ${RESULT_DIR}/response__${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME} \
-        2>&1 | tee ${RESULT_DIR}/log_${TASK}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}.log
+echo "Checkpoint: ${CKPT_PATH}"
+echo "Max frames: ${MAX_NUM_FRAMES}"
+echo "Task: ${TASK}"
+echo "Model: ${MODEL_NAME}"
+echo "Time message: ${TIME_MSG}"
+echo "Projector: ${REPLACE_PROJECTOR}"
+echo "Python: ${PYTHON_EXECUTABLE:-python}"
+echo "Output: ${OUTPUT_PATH}"
+
+"${LAUNCH_PREFIX[@]}" \
+"${PYTHON_EXECUTABLE:-python}" -m accelerate.commands.launch --num_processes "${NUM_GPUS}" --main_process_port "${MASTER_PORT}" -m lmms_eval \
+  --model "${MODEL_NAME}" \
+  --model_args "${MODEL_ARGS}" \
+  --tasks "${TASK}" \
+  --batch_size "${BATCH_SIZE:-1}" \
+  --log_samples \
+  --log_samples_suffix "${TASK_SUFFIX}" \
+  --output_path "${OUTPUT_PATH}" \
+  "${LIMIT_ARGS[@]}" \
+  2>&1 | tee "${LOG_PATH}"
diff --git a/scripts/eval/others/eval_internvl2-8B.sh b/scripts/eval/others/eval_internvl2-8B.sh
old mode 100644
new mode 100755
index 847ab26..3001044
--- a/scripts/eval/others/eval_internvl2-8B.sh
+++ b/scripts/eval/others/eval_internvl2-8B.sh
@@ -1,33 +1,50 @@
-export HF_DATASETS_OFFLINE=1
-MASTER_PORT=$((18000 + $RANDOM % 100))
+#!/usr/bin/env bash
+set -euo pipefail
 
-CKPT_PATH=/your_local_path_to/InternVL2-8B
-MODEL_NAME=internvl2
-CONV_TEMPLATE=internlm
-MAX_NUM_FRAMES=24
-NUM_GPUS=8
-
-TASK=odvbench
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+source "${SCRIPT_DIR}/../../env/streamforest_env.sh"
 
+MASTER_PORT="${MASTER_PORT:-$((18000 + RANDOM % 1000))}"
+CKPT_PATH="${INTERNVL2_CKPT_PATH:-${STREAMFOREST_INTERNVL2_CKPT_PATH:-InternVL2-8B}}"
+MODEL_NAME="${MODEL_NAME:-internvl2}"
+CONV_TEMPLATE="${CONV_TEMPLATE:-internlm}"
+MAX_NUM_FRAMES="${MAX_FRAMES:-24}"
+NUM_GPUS="${NUM_GPUS:-1}"
+TASK="${TASK:-odvbench}"
 TASK_SUFFIX="${TASK//,/_}"
-echo $TASK_SUFFIX
-JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
-LOG_DIR=$CKPT_PATH/eval
-mkdir $LOG_DIR
+JOB_NAME="$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")"
+CKPT_TAG="$(basename "${CKPT_PATH}")"
+LOG_DIR="${STREAMFOREST_OUTPUT_DIR}/eval/${CKPT_TAG}"
+mkdir -p "${LOG_DIR}"
+
+LIMIT_ARGS=()
+if [[ -n "${LIMIT:-}" ]]; then
+  LIMIT_ARGS=(--limit "${LIMIT}")
+fi
+
+LAUNCH_PREFIX=()
+if [[ "${STREAMFOREST_USE_SLURM:-0}" == "1" ]]; then
+  PARTITION="${PARTITION:-video5}"
+  CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
+  LAUNCH_PREFIX=(
+    srun -p "${PARTITION}"
+    --job-name="${JOB_NAME}"
+    --ntasks=1
+    --gres="gpu:${NUM_GPUS}"
+    --ntasks-per-node=1
+    --cpus-per-task="${CPUS_PER_TASK}"
+    --kill-on-bad-exit=1
+  )
+fi
 
-srun -p video5 \
-    --job-name=${JOB_NAME} \
-    --ntasks=1 \
-    --gres=gpu:${NUM_GPUS} \
-    --ntasks-per-node=1 \
-    --cpus-per-task=16 \
-    --kill-on-bad-exit=1 \
-    accelerate launch --num_processes ${NUM_GPUS} --main_process_port 10078 -m lmms_eval \
-        --model ${MODEL_NAME} \
-        --model_args pretrained=$CKPT_PATH,conv_template=$CONV_TEMPLATE,max_frames_num=$MAX_NUM_FRAMES \
-        --tasks $TASK \
-        --batch_size 1 \
-        --log_samples \
-        --log_samples_suffix $TASK_SUFFIX \
-        --output_path ${LOG_DIR}/log_result/${JOB_NAME}_f${MAX_NUM_FRAMES} \
-        2>&1 | tee ${LOG_DIR}/${JOB_NAME}_f${MAX_NUM_FRAMES}.log
+"${LAUNCH_PREFIX[@]}" \
+"${PYTHON_EXECUTABLE:-python}" -m accelerate.commands.launch --num_processes "${NUM_GPUS}" --main_process_port "${MASTER_PORT}" -m lmms_eval \
+  --model "${MODEL_NAME}" \
+  --model_args "pretrained=${CKPT_PATH},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_NUM_FRAMES}" \
+  --tasks "${TASK}" \
+  --batch_size "${BATCH_SIZE:-1}" \
+  --log_samples \
+  --log_samples_suffix "${TASK_SUFFIX}" \
+  --output_path "${LOG_DIR}/log_result/${JOB_NAME}_f${MAX_NUM_FRAMES}" \
+  "${LIMIT_ARGS[@]}" \
+  2>&1 | tee "${LOG_DIR}/${JOB_NAME}_f${MAX_NUM_FRAMES}.log"
diff --git a/scripts/eval/run_eval.sh b/scripts/eval/run_eval.sh
old mode 100644
new mode 100755
index 768aa59..2e97f73
--- a/scripts/eval/run_eval.sh
+++ b/scripts/eval/run_eval.sh
@@ -1,34 +1,46 @@
-STREAMFOREST_ROOT_PATH="/your_local_path_to/StreamForest"
-cd $STREAMFOREST_ROOT_PATH
-export PYTHONPATH=$STREAMFOREST_ROOT_PATH
+#!/usr/bin/env bash
+set -euo pipefail
 
-MAX_FRAMES=2048
-TIME_MSG=short_online_v2
-MODEL_NAME=streamforest
-CKPT_PATH=MCG-NJU/StreamForest-Qwen2-7B             #Our hf_weights or your ckpt path here
-# CKPT_PATH=MCG-NJU/StreamForest-Drive-Qwen2-7B       #Our hf_weights or your ckpt path here
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+source "${SCRIPT_DIR}/../env/streamforest_env.sh"
 
-TASKS=(
-    "odvbench"
-    "streamingbench"
-    "ovbench"
-    "ovobench"
-    "videomme"
-    "mlvu_mc"
-    "mvbench"
-    "perceptiontest_val_mc"
+MAX_FRAMES="${MAX_FRAMES:-2048}"
+TIME_MSG="${TIME_MSG:-short_online_v2}"
+MODEL_NAME="${MODEL_NAME:-streamforest}"
+CKPT_PATH="${STREAMFOREST_CKPT_PATH}"
+
+DEFAULT_TASKS=(
+  "odvbench"
+  "streamingbench"
+  "ovbench"
+  "ovobench"
+  "videomme"
+  "mlvu_mc"
+  "mvbench"
+  "perceptiontest_val_mc"
 )
 
+if [[ -n "${TASKS:-}" ]]; then
+  read -r -a TASK_ARRAY <<< "${TASKS//,/ }"
+else
+  TASK_ARRAY=("${DEFAULT_TASKS[@]}")
+fi
+
+echo "StreamForest root: ${STREAMFOREST_ROOT}"
+echo "HF_HOME: ${HF_HOME}"
+echo "Data root: ${STREAMFOREST_DATA_ROOT}"
+echo "Checkpoint: ${CKPT_PATH}"
+echo "Output root: ${STREAMFOREST_OUTPUT_DIR}"
 
-for TASK in "${TASKS[@]}"; do
-    echo "============================"
-    echo "Running benchmark: $TASK"
-    echo "============================"
+for TASK in "${TASK_ARRAY[@]}"; do
+  echo "============================"
+  echo "Running benchmark: ${TASK}"
+  echo "============================"
 
-    bash scripts/eval/online/eval_online_template.sh \
-        --ckpt_path $CKPT_PATH \
-        --max_frames $MAX_FRAMES \
-        --model_name $MODEL_NAME \
-        --time_msg $TIME_MSG \
-        --task "$TASK"
+  bash scripts/eval/online/eval_online_template.sh \
+    --ckpt_path "${CKPT_PATH}" \
+    --max_frames "${MAX_FRAMES}" \
+    --model_name "${MODEL_NAME}" \
+    --time_msg "${TIME_MSG}" \
+    --task "${TASK}"
 done
diff --git a/scripts/train/stage1-init_connector/s1_siglip_tome64_mlp.sh b/scripts/train/stage1-init_connector/s1_siglip_tome64_mlp.sh
index 0751f89..0e9db20 100644
--- a/scripts/train/stage1-init_connector/s1_siglip_tome64_mlp.sh
+++ b/scripts/train/stage1-init_connector/s1_siglip_tome64_mlp.sh
@@ -1,7 +1,6 @@
 #!/bin/bash
-STREAMFOREST_ROOT_PATH="/your_local_path_to/StreamForest"
-cd $STREAMFOREST_ROOT_PATH
-export PYTHONPATH=$STREAMFOREST_ROOT_PATH
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+source "${SCRIPT_DIR}/../../env/streamforest_env.sh"
 export OMP_NUM_THREADS=1
 export DISABLE_ADDMM_CUDA_LT=1
 export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
@@ -25,10 +24,10 @@ PROMPT_VERSION=plain
 BASE_RUN_NAME=stage1-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${mm_projector_type}-pretrain_${DATA_VERSION_CLEAN}_${PROMPT_VERSION}_$(date +"%Y%m%d_%H%M%S")
 echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
 
-PARTITION='video'
+PARTITION="${PARTITION:-video}"
 JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")
 
-OUTPUT_DIR=ckpt/stage1-init_connector/${BASE_RUN_NAME}
+OUTPUT_DIR="${STREAMFOREST_CKPT_ROOT}/stage1-init_connector/${BASE_RUN_NAME}"
 mkdir -p ${OUTPUT_DIR}/runs
 
 srun -p ${PARTITION} \
@@ -76,4 +75,4 @@ srun -p ${PARTITION} \
     --local_num_frames 1 \
     --sample_type middle \
     --mm_local_num_frames 1 \
-    2>&1 | tee ${OUTPUT_DIR}/runs/${MID_RUN_NAME}.log
\ No newline at end of file
+    2>&1 | tee ${OUTPUT_DIR}/runs/${MID_RUN_NAME}.log
diff --git a/scripts/train/stage2-visual_pretraining/s2_siglip_tome64_mlp.sh b/scripts/train/stage2-visual_pretraining/s2_siglip_tome64_mlp.sh
index 1b403ad..b8ffb4a 100644
--- a/scripts/train/stage2-visual_pretraining/s2_siglip_tome64_mlp.sh
+++ b/scripts/train/stage2-visual_pretraining/s2_siglip_tome64_mlp.sh
@@ -1,7 +1,6 @@
 #!/bin/bash
-STREAMFOREST_ROOT_PATH="/your_local_path_to/StreamForest"
-cd $STREAMFOREST_ROOT_PATH
-export PYTHONPATH=$STREAMFOREST_ROOT_PATH
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+source "${SCRIPT_DIR}/../../env/streamforest_env.sh"
 export OMP_NUM_THREADS=1
 export DISABLE_ADDMM_CUDA_LT=1
 export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
@@ -19,7 +18,7 @@ VISION_MODEL_VERSION_CLEAN=$(basename "$VISION_MODEL_VERSION")
 LLM_VERSION="Qwen/Qwen2-7B-Instruct"
 LLM_VERSION_CLEAN=Qwen2-7B-Instruct
 
-PROJECTOR_PATH="ckpt/stage1-init_connector/path_of_your_stage1_ckpt_here/mm_projector.bin"  #your stage1 ckpt
+PROJECTOR_PATH="${PROJECTOR_PATH:-${STREAMFOREST_CKPT_ROOT}/stage1-init_connector/path_of_your_stage1_ckpt_here/mm_projector.bin}"  # your stage1 ckpt
 
 mm_projector_type=tome64_mlp
 
@@ -28,10 +27,10 @@ PROMPT_VERSION="qwen_2"
 MID_RUN_NAME=stage2-${mm_projector_type}_${DATA_VERSION_CLEAN}_$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")
 echo "MID_RUN_NAME: ${MID_RUN_NAME}"
 
-PARTITION='video'
+PARTITION="${PARTITION:-video}"
 JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")
 
-OUTPUT_DIR=ckpt/stage2-visual_pretraining/${MID_RUN_NAME}
+OUTPUT_DIR="${STREAMFOREST_CKPT_ROOT}/stage2-visual_pretraining/${MID_RUN_NAME}"
 mkdir -p ${OUTPUT_DIR}/runs
 
 srun -p ${PARTITION} \
@@ -92,4 +91,4 @@ srun -p ${PARTITION} \
     --mm_num_compress_query_type pooling \
     --mm_close_init True \
     --mm_local_num_frames 2 \
-    2>&1 | tee ${OUTPUT_DIR}/runs/${MID_RUN_NAME}.log
\ No newline at end of file
+    2>&1 | tee ${OUTPUT_DIR}/runs/${MID_RUN_NAME}.log
diff --git a/scripts/train/stage3-video_sft/s3_siglip_tome16_mlp.sh b/scripts/train/stage3-video_sft/s3_siglip_tome16_mlp.sh
index f7552bc..4cab2d3 100644
--- a/scripts/train/stage3-video_sft/s3_siglip_tome16_mlp.sh
+++ b/scripts/train/stage3-video_sft/s3_siglip_tome16_mlp.sh
@@ -1,7 +1,6 @@
 #!/bin/bash
-STREAMFOREST_ROOT_PATH="/your_local_path_to/StreamForest"
-cd $STREAMFOREST_ROOT_PATH
-export PYTHONPATH=$STREAMFOREST_ROOT_PATH
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+source "${SCRIPT_DIR}/../../env/streamforest_env.sh"
 export OMP_NUM_THREADS=1
 export DISABLE_ADDMM_CUDA_LT=1
 export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
@@ -19,7 +18,7 @@ DATA_VERSION_CLEAN=$(basename "$DATA_VERSION" .yaml)
 VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
 VISION_MODEL_VERSION_CLEAN=$(basename "$VISION_MODEL_VERSION")
 
-LLM_VERSION="ckpt/stage2-visual_pretraining/path_of_your_stage2_ckpt_here"  #your stage2 ckpt
+LLM_VERSION="${LLM_VERSION:-${STREAMFOREST_CKPT_ROOT}/stage2-visual_pretraining/path_of_your_stage2_ckpt_here}"  # your stage2 ckpt
 LLM_VERSION_CLEAN=$(basename "$LLM_VERSION")
 
 mm_projector_type=tome16_mlp
@@ -30,10 +29,10 @@ MID_RUN_NAME=stage3-${mm_projector_type}_${DATA_VERSION_CLEAN}_$(basename "$0" .
 echo "MID_RUN_NAME: ${MID_RUN_NAME}"
 
 
-PARTITION='video5'
+PARTITION="${PARTITION:-video5}"
 JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")
 
-OUTPUT_DIR=ckpt/stage3-video_sft/${MID_RUN_NAME}
+OUTPUT_DIR="${STREAMFOREST_CKPT_ROOT}/stage3-video_sft/${MID_RUN_NAME}"
 mkdir -p ${OUTPUT_DIR}/runs
 
 srun -p ${PARTITION} \
diff --git a/scripts/train/stage4-online_ft/s4_siglip_online_dynamic_tree_memory.sh b/scripts/train/stage4-online_ft/s4_siglip_online_dynamic_tree_memory.sh
index 4d4c61e..e262342 100644
--- a/scripts/train/stage4-online_ft/s4_siglip_online_dynamic_tree_memory.sh
+++ b/scripts/train/stage4-online_ft/s4_siglip_online_dynamic_tree_memory.sh
@@ -1,7 +1,6 @@
 #!/bin/bash
-STREAMFOREST_ROOT_PATH="/your_local_path_to/StreamForest"
-cd $STREAMFOREST_ROOT_PATH
-export PYTHONPATH=$STREAMFOREST_ROOT_PATH
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+source "${SCRIPT_DIR}/../../env/streamforest_env.sh"
 export OMP_NUM_THREADS=1
 export DISABLE_ADDMM_CUDA_LT=1
 export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
@@ -16,7 +15,7 @@ mkdir -p $TRITON_CACHE_DIR
 VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
 VISION_MODEL_VERSION_CLEAN=$(basename "$VISION_MODEL_VERSION")
 
-LLM_VERSION="MCG-NJU/StreamForest-Pretrain-Qwen2-7B"    #Ours hf_weight or your stage3 ckpt
+LLM_VERSION="${LLM_VERSION:-MCG-NJU/StreamForest-Pretrain-Qwen2-7B}"    # Ours hf_weight or your stage3 ckpt
 LLM_VERSION_CLEAN=$(basename "$LLM_VERSION")
 
 DATA_VERSION="anno/data_list/stage4_online_sft.yaml"    #Download from https://huggingface.co/datasets/MCG-NJU/StreamForest-Annodata/tree/main/data_list
@@ -32,10 +31,10 @@ MID_RUN_NAME=stage4-${mm_projector_type}_${DATA_VERSION_CLEAN}_$(basename "$0" .
 echo "MID_RUN_NAME: ${MID_RUN_NAME}"
 
 
-PARTITION='video5'
+PARTITION="${PARTITION:-video5}"
 JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")
 
-OUTPUT_DIR=./ckpt/stage4-postft-qwen-siglip/${MID_RUN_NAME}
+OUTPUT_DIR="${STREAMFOREST_CKPT_ROOT}/stage4-postft-qwen-siglip/${MID_RUN_NAME}"
 mkdir -p ${OUTPUT_DIR}/runs
 
 srun -p ${PARTITION} \
diff --git a/scripts/train/stage5-drive_ft/s5_siglip_online_tree_memory_drive.sh b/scripts/train/stage5-drive_ft/s5_siglip_online_tree_memory_drive.sh
index 74cdfc0..790c7ff 100644
--- a/scripts/train/stage5-drive_ft/s5_siglip_online_tree_memory_drive.sh
+++ b/scripts/train/stage5-drive_ft/s5_siglip_online_tree_memory_drive.sh
@@ -1,7 +1,6 @@
 #!/bin/bash
-STREAMFOREST_ROOT_PATH="/your_local_path_to/StreamForest"
-cd $STREAMFOREST_ROOT_PATH
-export PYTHONPATH=$STREAMFOREST_ROOT_PATH
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+source "${SCRIPT_DIR}/../../env/streamforest_env.sh"
 export OMP_NUM_THREADS=1
 export DISABLE_ADDMM_CUDA_LT=1
 export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
@@ -16,7 +15,7 @@ mkdir -p $TRITON_CACHE_DIR
 VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
 VISION_MODEL_VERSION_CLEAN=$(basename "$VISION_MODEL_VERSION")
 
-LLM_VERSION="MCG-NJU/StreamForest-Qwen2-7B"     #Ours hf_weight or your stage4 ckpt
+LLM_VERSION="${LLM_VERSION:-${STREAMFOREST_CKPT_PATH}}"     # Ours hf_weight or your stage4 ckpt
 LLM_VERSION_CLEAN=$(basename "$LLM_VERSION")
 
 DATA_VERSION="anno/data_list/stage5_drive_sft.yaml"     #Download from https://huggingface.co/datasets/MCG-NJU/StreamForest-Annodata/tree/main/data_list
@@ -32,10 +31,10 @@ MID_RUN_NAME=stage5-${mm_projector_type}_${DATA_VERSION_CLEAN}_$(basename "$0" .
 echo "MID_RUN_NAME: ${MID_RUN_NAME}"
 
 
-PARTITION='video5'
+PARTITION="${PARTITION:-video5}"
 JOB_NAME=$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")
 
-OUTPUT_DIR=ckpt/stage5-driveft-qwen-siglip/${MID_RUN_NAME}
+OUTPUT_DIR="${STREAMFOREST_CKPT_ROOT}/stage5-driveft-qwen-siglip/${MID_RUN_NAME}"
 mkdir -p ${OUTPUT_DIR}/runs
 
 srun -p ${PARTITION} \

diff --git a/lmms_eval/tasks/_task_utils/streamforest_paths.py b/lmms_eval/tasks/_task_utils/streamforest_paths.py
new file mode 100644
index 0000000..492a4b5
--- /dev/null
+++ b/lmms_eval/tasks/_task_utils/streamforest_paths.py
@@ -0,0 +1,151 @@
+import os
+from pathlib import Path
+from typing import Iterable, List, Optional, Tuple
+
+
+DEFAULT_HF_HOME = "/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face"
+
+DEFAULT_TASK_ROOTS = {
+    "odvbench": ("ODVBench", "OMQA-DS-ALL", "anno/eval/ODVBench"),
+    "ovbench": ("OVBench", "ovbench", "anno/eval/OVBench"),
+    "ovbench_full": ("OVBench", "ovbench", "anno/eval/OVBench"),
+    "ovobench": ("OVO-Bench/chunked_videos", "OVO-Bench", "OVOBench/video/chunked_videos"),
+    "streamingbench": (
+        "StreamingBench/Real-Time Visual Understanding",
+        "StreamingBench/StreamingBench",
+        "StreamingBench",
+    ),
+    "mlvu": ("mlvu", "MLVU/video", "MVLU/MLVU/video"),
+    "mlvu_mc": ("mlvu", "MLVU/video", "MVLU/MLVU/video"),
+    "mvbench": ("MVBench", "mvbench", "anno/eval/MVBench"),
+    "videomme": ("Video-MME/videos", "videomme", "Video-MME/videomme"),
+}
+
+
+def hf_home() -> str:
+    return os.path.expanduser(os.environ.get("HF_HOME", DEFAULT_HF_HOME))
+
+
+def data_root() -> str:
+    return os.path.expanduser(os.environ.get("STREAMFOREST_DATA_ROOT", hf_home()))
+
+
+def _env_key(task_name: str, suffix: str) -> str:
+    normalized = "".join(ch if ch.isalnum() else "_" for ch in task_name.upper())
+    return f"STREAMFOREST_{normalized}_{suffix}"
+
+
+def _dedupe(paths: Iterable[str]) -> List[str]:
+    seen = set()
+    result = []
+    for path in paths:
+        if not path:
+            continue
+        expanded = os.path.expanduser(path)
+        if expanded not in seen:
+            result.append(expanded)
+            seen.add(expanded)
+    return result
+
+
+def _under_data_root(path: str) -> str:
+    if os.path.isabs(path):
+        return path
+    return os.path.join(data_root(), path)
+
+
+def _strip_storage_prefix(path: str) -> str:
+    if "s3://" in path:
+        return path.split("s3://", 1)[1]
+    return path
+
+
+def _remote_tails(path: str) -> List[str]:
+    stripped = _strip_storage_prefix(path).strip("/")
+    if not stripped:
+        return []
+    parts = stripped.split("/")
+    tails = [stripped]
+    if len(parts) > 1:
+        tails.append("/".join(parts[1:]))
+    if len(parts) > 2:
+        tails.append("/".join(parts[2:]))
+    tails.append(parts[-1])
+    return _dedupe(tails)
+
+
+def task_roots(task_name: str, cache_name: Optional[str] = None) -> List[str]:
+    roots = []
+    for suffix in ("ROOT", "VIDEO_ROOT", "DATA_ROOT"):
+        env_value = os.environ.get(_env_key(task_name, suffix))
+        if env_value:
+            roots.append(env_value)
+
+    if cache_name:
+        roots.append(cache_name if os.path.isabs(cache_name) else _under_data_root(cache_name))
+
+    for default in DEFAULT_TASK_ROOTS.get(task_name, ()):
+        roots.append(_under_data_root(default))
+
+    roots.append(data_root())
+    return _dedupe(roots)
+
+
+def resolve_benchmark_video(
+    task_name: str,
+    dataset_folder: str,
+    video_name: str,
+    cache_name: Optional[str] = None,
+) -> Tuple[str, List[str]]:
+    if os.path.isabs(video_name) and os.path.exists(video_name):
+        return video_name, [video_name]
+
+    candidates = []
+    dataset_is_remote = "s3://" in dataset_folder
+    dataset_is_abs = os.path.isabs(dataset_folder)
+
+    if dataset_folder and not dataset_is_remote:
+        base = dataset_folder if dataset_is_abs else _under_data_root(dataset_folder)
+        candidates.extend([os.path.join(base, video_name), os.path.join(base, os.path.basename(video_name))])
+
+    tails = _remote_tails(dataset_folder)
+    for root in task_roots(task_name, cache_name):
+        candidates.append(os.path.join(root, video_name))
+        candidates.append(os.path.join(root, os.path.basename(video_name)))
+        for tail in tails:
+            candidates.append(os.path.join(root, tail, video_name))
+            candidates.append(os.path.join(root, tail, os.path.basename(video_name)))
+
+    candidates = _dedupe(candidates)
+    for candidate in candidates:
+        if os.path.exists(candidate):
+            return candidate, candidates
+
+    fallback = os.path.join(dataset_folder, video_name) if dataset_folder else video_name
+    return fallback, candidates
+
+
+def resolve_videomme_path(name: str, cache_name: Optional[str] = None) -> Tuple[str, List[str]]:
+    candidates = []
+    for root in task_roots("videomme", cache_name):
+        candidates.append(os.path.join(root, name))
+        candidates.append(os.path.join(root, "data", name))
+
+    candidates = _dedupe(candidates)
+    for candidate in candidates:
+        if os.path.exists(candidate):
+            return candidate, candidates
+    return candidates[0] if candidates else name, candidates
+
+
+def existing_video_variant(path: str) -> str:
+    variants = [path]
+    suffix = Path(path).suffix
+    if suffix:
+        stem = path[: -len(suffix)]
+        variants.extend([stem + suffix.upper(), stem + ".mkv", stem + ".MKV"])
+
+    for variant in _dedupe(variants):
+        if os.path.exists(variant):
+            return variant
+    return path

diff --git a/scripts/env/streamforest_env.sh b/scripts/env/streamforest_env.sh
new file mode 100755
index 0000000..4f9eb53
--- /dev/null
+++ b/scripts/env/streamforest_env.sh
@@ -0,0 +1,63 @@
+#!/usr/bin/env bash
+
+STREAMFOREST_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+export STREAMFOREST_ROOT="${STREAMFOREST_ROOT:-$(cd "${STREAMFOREST_ENV_DIR}/../.." && pwd)}"
+export STREAMFOREST_PROJECT_ROOT="${STREAMFOREST_PROJECT_ROOT:-$(cd "${STREAMFOREST_ROOT}/../.." && pwd)}"
+
+export HF_HOME="${HF_HOME:-/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face}"
+export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
+export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
+
+export STREAMFOREST_DATA_ROOT="${STREAMFOREST_DATA_ROOT:-${HF_HOME}}"
+export STREAMFOREST_OUTPUT_DIR="${STREAMFOREST_OUTPUT_DIR:-${STREAMFOREST_ROOT}/results}"
+export STREAMFOREST_CKPT_ROOT="${STREAMFOREST_CKPT_ROOT:-${STREAMFOREST_ROOT}/ckpt}"
+if [[ -z "${STREAMFOREST_ANNO_ROOT:-}" ]]; then
+  STREAMFOREST_PROJECT_ANNO_ROOT="${STREAMFOREST_PROJECT_ROOT}/benchmarks/streamforest/eval"
+  STREAMFOREST_HF_ANNODATA_ROOT=""
+  STREAMFOREST_HF_ANNODATA_SNAPSHOTS="${HF_HOME}/hub/datasets--MCG-NJU--StreamForest-Annodata/snapshots"
+  if [[ -d "${STREAMFOREST_HF_ANNODATA_SNAPSHOTS}" ]]; then
+    for STREAMFOREST_HF_ANNODATA_CANDIDATE in "${STREAMFOREST_HF_ANNODATA_SNAPSHOTS}"/*/eval; do
+      if [[ -e "${STREAMFOREST_HF_ANNODATA_CANDIDATE}/OVOBench/json/backward_tracking.json" ]]; then
+        STREAMFOREST_HF_ANNODATA_ROOT="${STREAMFOREST_HF_ANNODATA_CANDIDATE}"
+        break
+      fi
+    done
+  fi
+
+  if [[ -e "${STREAMFOREST_PROJECT_ANNO_ROOT}/OVOBench/json/backward_tracking.json" ]]; then
+    export STREAMFOREST_ANNO_ROOT="${STREAMFOREST_PROJECT_ROOT}/benchmarks/streamforest/eval"
+  elif [[ -n "${STREAMFOREST_HF_ANNODATA_ROOT}" ]]; then
+    export STREAMFOREST_ANNO_ROOT="${STREAMFOREST_HF_ANNODATA_ROOT}"
+  elif [[ -d "${STREAMFOREST_PROJECT_ANNO_ROOT}" ]]; then
+    export STREAMFOREST_ANNO_ROOT="${STREAMFOREST_PROJECT_ANNO_ROOT}"
+  else
+    export STREAMFOREST_ANNO_ROOT="${STREAMFOREST_ROOT}/anno/eval"
+  fi
+  unset STREAMFOREST_PROJECT_ANNO_ROOT
+  unset STREAMFOREST_HF_ANNODATA_ROOT
+  unset STREAMFOREST_HF_ANNODATA_SNAPSHOTS
+  unset STREAMFOREST_HF_ANNODATA_CANDIDATE
+fi
+
+if [[ -z "${STREAMFOREST_CKPT_PATH:-}" ]]; then
+  if [[ -d "${HF_HOME}/StreamForest-Qwen2-7B" ]]; then
+    export STREAMFOREST_CKPT_PATH="${HF_HOME}/StreamForest-Qwen2-7B"
+  else
+    export STREAMFOREST_CKPT_PATH="MCG-NJU/StreamForest-Qwen2-7B"
+  fi
+fi
+
+if [[ -z "${STREAMFOREST_DRIVE_CKPT_PATH:-}" ]]; then
+  if [[ -d "${HF_HOME}/StreamForest-Drive-Qwen2-7B" ]]; then
+    export STREAMFOREST_DRIVE_CKPT_PATH="${HF_HOME}/StreamForest-Drive-Qwen2-7B"
+  else
+    export STREAMFOREST_DRIVE_CKPT_PATH="MCG-NJU/StreamForest-Drive-Qwen2-7B"
+  fi
+fi
+
+case ":${PYTHONPATH:-}:" in
+  *":${STREAMFOREST_ROOT}:"*) ;;
+  *) export PYTHONPATH="${STREAMFOREST_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
+esac
+
+cd "${STREAMFOREST_ROOT}"

diff --git a/scripts/eval/run_smoke.sh b/scripts/eval/run_smoke.sh
new file mode 100755
index 0000000..0c4e9fc
--- /dev/null
+++ b/scripts/eval/run_smoke.sh
@@ -0,0 +1,12 @@
+#!/usr/bin/env bash
+set -euo pipefail
+
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+source "${SCRIPT_DIR}/../env/streamforest_env.sh"
+
+export TASKS="${TASKS:-ovobench_backward_tracking}"
+export LIMIT="${LIMIT:-1}"
+export NUM_GPUS="${NUM_GPUS:-1}"
+export MAX_FRAMES="${MAX_FRAMES:-8}"
+
+bash "${SCRIPT_DIR}/run_eval.sh"

diff --git a/scripts/setup/create_streamforest_env.sh b/scripts/setup/create_streamforest_env.sh
new file mode 100755
index 0000000..5562764
--- /dev/null
+++ b/scripts/setup/create_streamforest_env.sh
@@ -0,0 +1,66 @@
+#!/usr/bin/env bash
+set -euo pipefail
+
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+source "${SCRIPT_DIR}/../env/streamforest_env.sh"
+
+ENV_ROOT="${STREAMFOREST_ENV_ROOT:-/apdcephfs_tj5/share_303570626/yiyuwang/envs}"
+ENV_NAME="${STREAMFOREST_ENV_NAME:-streamforest-py310}"
+ENV_PREFIX="${STREAMFOREST_ENV_PREFIX:-${ENV_ROOT}/${ENV_NAME}}"
+EXISTING_ENV_PREFIX="${STREAMFOREST_EXISTING_ENV_PREFIX:-${ENV_ROOT}/lmms-streamforest-py312-tf446}"
+PYTHON_VERSION="${STREAMFOREST_PYTHON_VERSION:-3.10}"
+PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"
+
+mkdir -p "${ENV_ROOT}"
+
+if [[ "${STREAMFOREST_FORCE_CREATE:-0}" != "1" && -x "${EXISTING_ENV_PREFIX}/bin/python" ]]; then
+  ENV_PREFIX="${EXISTING_ENV_PREFIX}"
+  echo "Using existing verified environment: ${ENV_PREFIX}"
+elif [[ -x "${ENV_PREFIX}/bin/python" ]]; then
+  echo "Using existing environment: ${ENV_PREFIX}"
+elif command -v conda >/dev/null 2>&1; then
+  echo "Creating conda environment: ${ENV_PREFIX}"
+  conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
+elif command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
+  echo "Creating venv environment: ${ENV_PREFIX}"
+  "python${PYTHON_VERSION}" -m venv "${ENV_PREFIX}"
+else
+  echo "Neither conda nor python${PYTHON_VERSION} was found. Enter the taiji container and rerun this script." >&2
+  exit 1
+fi
+
+PYTHON="${ENV_PREFIX}/bin/python"
+PIP="${ENV_PREFIX}/bin/pip"
+
+if [[ "${ENV_PREFIX}" != "${EXISTING_ENV_PREFIX}" || "${STREAMFOREST_INSTALL_REQUIREMENTS:-0}" == "1" ]]; then
+  "${PYTHON}" -m pip install --upgrade pip setuptools wheel
+  "${PIP}" install ${PIP_EXTRA_ARGS} -r "${STREAMFOREST_ROOT}/requirements.txt"
+else
+  echo "Skipping pip install for existing environment. Set STREAMFOREST_INSTALL_REQUIREMENTS=1 to reinstall."
+fi
+
+if [[ "${STREAMFOREST_DOWNLOAD_HF:-0}" == "1" ]]; then
+  "${PYTHON}" "${STREAMFOREST_ROOT}/download_hf.py"
+fi
+
+"${PYTHON}" - <<'PY'
+import importlib
+import sys
+
+modules = ["torch", "transformers", "accelerate", "av", "decord", "lmms_eval", "llava"]
+failed = []
+for name in modules:
+    try:
+        mod = importlib.import_module(name)
+        print(f"{name}: ok {getattr(mod, '__version__', '')}")
+    except Exception as exc:
+        failed.append((name, repr(exc)))
+
+if failed:
+    for name, exc in failed:
+        print(f"{name}: FAIL {exc}", file=sys.stderr)
+    sys.exit(1)
+PY
+
+echo "Environment ready: ${ENV_PREFIX}"
+echo "Activate with: source ${ENV_PREFIX}/bin/activate"
```
