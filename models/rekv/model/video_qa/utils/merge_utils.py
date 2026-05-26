"""
结果合并和评估工具

⚠️ 警告: merge_results() 函数已弃用！
推荐使用 PyTorch 原生的 torch.distributed.gather_object 方法
详见: docs/distributed.md
"""
import os
import subprocess
import warnings
from pathlib import Path
from logzero import logger

def run_evaluation(save_dir, eval_script):
    """运行评估脚本"""
    logger.info(f"Running evaluation: {eval_script}")
    cmd = f"python {eval_script} --save_dir {save_dir}"
    subprocess.run(cmd, shell=True, check=True)

