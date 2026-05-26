# Video QA 模块说明

## 📝 概述

这是一个统一的视频问答推理模块，支持多种数据集和推理模式。

## 🏗️ 架构设计

### 核心组件

```
video_qa/
├── base_refactored.py          # 基类：所有solver的通用逻辑
├── rekv_offline_refactored.py  # 离线推理：标准视频问答
├── videomme_refactored.py      # VideoMME专用：带性能统计
├── rekv_stream_refactored.py  # 流式推理：增量编码
├── solver_factory.py           # Solver工厂：根据配置创建实例
├── configs.py                  # 数据集配置：统一管理
└── run_distributed.py          # 分布式推理：多卡并行
```

### 设计模式

1. **工厂模式** (`solver_factory.py`)
   - 根据数据集配置自动选择正确的solver
   - 解耦配置和实现

2. **模板方法** (`base_refactored.py`)
   - 定义通用流程
   - 子类实现特定逻辑

3. **策略模式** (三种solver)
   - 不同数据集使用不同策略
   - 灵活扩展

## 🚀 使用方法

### 1. 配置数据集

在 `configs.py` 中添加或修改数据集配置：

```python
DATASETS = {
    'my_dataset': DatasetConfig(
        name='my_dataset',
        anno_path='data/my_dataset/test.json',
        solver='rekv_offline_vqa',  # 选择solver类型
        eval_script='models/rekv/model/video_qa/eval/eval_my_dataset.py'
    ),
}
```

### 2. 选择Solver类型

支持三种solver：

| Solver | 用途 | 特性 |
|--------|------|------|
| `rekv_offline_vqa` | 标准视频问答 | 支持多选题和开放式问答 |
| `videomme_rekv_offline_vqa` | VideoMME数据集 | 带GPU时间/内存统计 |
| `rekv_stream_vqa` | 流式视频问答 | 增量编码，支持时间窗口 |

### 3. 运行推理

#### 单卡推理

```bash
python -m model.video_qa.run_distributed \
    --dataset egoschema \
    --save_dir results/egoschema \
    --model llava_ov_7b
```

#### 多卡推理

```bash
torchrun --nproc_per_node=4 \
    -m model.video_qa.run_distributed \
    --dataset videomme \
    --save_dir results/videomme \
    --model llava_ov_7b \
    --retrieve_size 64
```

## 📊 Solver详细说明

### ReKVOfflineVQA (标准离线推理)

**适用数据集**: EgoSchema, MLVU, CG-Bench, ActivityNet-QA

**核心功能**:
- 编码整个视频到KV缓存
- 支持多选题和开放式问答
- 自动提取选项字母

**数据格式**:
```json
{
  "video_id": "xxx",
  "video_path": "path/to/video.mp4",
  "conversations": [
    {
      "question": "What happened?",
      "answer": "Something",
      "choices": ["A", "B", "C", "D"]  // 可选
    }
  ]
}
```

### VideoMMEReKVOfflineVQA (VideoMME专用)

**适用数据集**: Video-MME, Video-MME Subset

**特殊功能**:
- ✅ GPU编码时间统计
- ✅ 显存峰值监控
- ✅ 累积时间追踪
- ✅ 支持duration字段

**特殊数据格式**:
```json
{
  "video_id": "xxx",
  "duration": 120.5,  // 视频时长
  "conversations": [
    {
      "question": "What is shown?",
      "answer": "A",  // 直接是选项字母，不是文本
      "choices": ["A", "B", "C", "D"]
    }
  ]
}
```

**输出字段**:
```python
{
    'video_id': 'xxx',
    'question': '...',
    'pred_answer': '...',
    'pred_choice': 'A',
    'qa_acc': 100.0,
    'duration': 120.5  # 额外的duration字段
}
```

### ReKVStreamVQA (流式推理)

**适用数据集**: RVS-Ego, RVS-Movie

**核心特性**:
- 增量编码视频帧
- 支持时间窗口查询
- 内存效率高

**数据格式**:
```json
{
  "video_id": "xxx",
  "video_path": "path/to/video.npy",
  "conversations": [
    {
      "question": "What happened?",
      "answer": "Something",
      "start_time": 10.0,  // 时间窗口开始
      "end_time": 20.0     // 时间窗口结束
    }
  ]
}
```

## 🔧 扩展指南

### 添加新的Solver

1. **创建新的solver类**:

```python
# my_custom_solver.py
from .rekv_offline_refactored import ReKVOfflineVQA

class MyCustomVQA(ReKVOfflineVQA):
    """自定义solver"""
    
    def answer_single(self, qa_pair, video_id):
        # 实现你的逻辑
        pass
```

2. **注册到工厂**:

```python
# solver_factory.py
SOLVER_MAP = {
    'rekv_offline_vqa': ReKVOfflineVQA,
    'videomme_rekv_offline_vqa': VideoMMEReKVOfflineVQA,
    'rekv_stream_vqa': ReKVStreamVQA,
    'my_custom_vqa': MyCustomVQA,  # 添加这行
}
```

3. **配置数据集**:

```python
# configs.py
DATASETS = {
    'my_dataset': DatasetConfig(
        name='my_dataset',
        anno_path='...',
        solver='my_custom_vqa',  # 使用新solver
        eval_script='...'
    ),
}
```

## 📝 最佳实践

### 1. 保持函数简洁
- 每个函数 < 15行
- 单一职责原则
- 清晰的命名

### 2. 使用统一的接口
- 所有solver继承自`BaseVQA`
- 实现`answer_single()`方法
- 返回标准化的字典

### 3. 配置驱动
- 所有数据集配置在`configs.py`
- 通过solver名称选择实现
- 避免硬编码

## 🐛 常见问题

### Q: 如何添加新的数据集？

A: 在 `configs.py` 中添加配置，选择合适的solver即可。

### Q: solver选择错误怎么办？

A: `solver_factory.py` 会自动fallback到`rekv_offline_vqa`，并记录warning日志。

### Q: 如何自定义输出字段？

A: 重写 `_format_mc_result()` 或 `_format_open_result()` 方法。

### Q: 多选题的正确答案如何处理？

A: 
- 标准数据集：answer是文本，自动匹配choices得到字母
- VideoMME：answer直接是字母（A/B/C/D）

## 📊 性能优化

### 内存优化
- 使用流式推理处理长视频
- 设置合适的`retrieve_size`
- 控制`chunk_size`

### 速度优化
- 使用多卡并行（`torchrun`）
- 启用TF32加速（`--tf32`）
- 调整`sample_fps`降低帧数

## 📚 相关文档

- [分布式推理详解](../../docs/distributed.md)
- [数据集准备](../../data/README.md)
- [模型配置](../config.py)
