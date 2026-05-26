"""数据集配置 - 统一管理所有数据集参数"""
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    """数据集配置类"""
    name: str
    anno_path: str
    solver: str
    eval_script: str

# 所有支持的数据集配置
DATASETS = {
    'smoke': DatasetConfig(
        name='smoke',
        anno_path='benchmarks/offline/smoke/smoke_rekv.json',
        solver='rekv_offline_vqa',
        eval_script='models/rekv/model/video_qa/eval/eval_smoke.py'
    ),
    'videomme': DatasetConfig(
        name='videomme',
        anno_path='benchmarks/offline/videomme/random_videomme.json',
        solver='videomme_rekv_offline_vqa',
        eval_script='models/rekv/model/video_qa/eval/videomme_rekv_offline_vqa.py'
    ),
    'videomme_subset': DatasetConfig(
        name='videomme_subset',
        anno_path='benchmarks/offline/videomme/videomme_subset.json',
        solver='videomme_rekv_offline_vqa',
        eval_script='models/rekv/model/video_qa/eval/eval_videomme.py'
    ),
    'mlvu': DatasetConfig(
        name='mlvu',
        anno_path='benchmarks/offline/mlvu/dev_debug_mc.json',
        solver='rekv_offline_vqa',
        eval_script='models/rekv/model/video_qa/eval/eval_multiple_choice.py'
    ),
    'egoschema': DatasetConfig(
        name='egoschema',
        anno_path='benchmarks/offline/egoschema/full.json',
        solver='rekv_offline_vqa',
        eval_script='models/rekv/model/video_qa/eval/eval_egoschema.py'
    ),
    'egoschema_subset': DatasetConfig(
        name='egoschema_subset',
        anno_path='benchmarks/offline/egoschema_subset/egoschema_subset.json',
        solver='videomme_rekv_offline_vqa',
        eval_script='models/rekv/model/video_qa/eval/eval_egoschema_subset.py'
    ),
    'qaego4d': DatasetConfig(
        name='qaego4d',
        anno_path='benchmarks/offline/qaego4d/test_mc.json',
        solver='rekv_offline_vqa',
        eval_script='models/rekv/model/video_qa/eval/eval_multiple_choice.py'
    ),
    'cgbench': DatasetConfig(
        name='cgbench',
        anno_path='benchmarks/offline/cgbench/full_mc.json',
        solver='rekv_offline_vqa',
        eval_script='models/rekv/model/video_qa/eval/eval_multiple_choice.py'
    ),
    'activitynet_qa': DatasetConfig(
        name='activitynet_qa',
        anno_path='benchmarks/offline/activitynet_qa/test.json',
        solver='rekv_offline_vqa',
        eval_script='models/rekv/model/video_qa/eval/eval_open_ended.py'
    ),
    'rvs_ego': DatasetConfig(
        name='rvs_ego',
        anno_path='benchmarks/offline/rvs/ego/ego4d_oe.json',
        solver='rekv_stream_vqa',
        eval_script='models/rekv/model/video_qa/eval/eval_open_ended.py'
    ),
    'rvs_movie': DatasetConfig(
        name='rvs_movie',
        anno_path='benchmarks/offline/rvs/movie/movienet_oe.json',
        solver='rekv_stream_vqa',
        eval_script='models/rekv/model/video_qa/eval/eval_open_ended.py'
    ),
}
