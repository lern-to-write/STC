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
    'videomme': DatasetConfig(
        name='videomme',
        anno_path='data/videomme/random_videomme.json',
        solver='videomme_rekv_offline_vqa',
        eval_script='model/video_qa/eval/videomme_rekv_offline_vqa.py'
    ),
    'videomme_subset': DatasetConfig(
        name='videomme_subset',
        anno_path='data/videomme/videomme_subset.json',
        solver='videomme_rekv_offline_vqa',
        eval_script='model/video_qa/eval/eval_videomme.py'
    ),
    'mlvu': DatasetConfig(
        name='mlvu',
        anno_path='data/mlvu/dev_debug_mc.json',
        solver='rekv_offline_vqa',
        eval_script='model/video_qa/eval/eval_multiple_choice.py'
    ),
    'egoschema': DatasetConfig(
        name='egoschema',
        anno_path='data/egoschema/full.json',
        solver='rekv_offline_vqa',
        eval_script='model/video_qa/eval/eval_egoschema.py'
    ),
    'egoschema_subset': DatasetConfig(
        name='egoschema_subset',
        anno_path='data/egoschema_subset/egoschema_subset.json',
        solver='videomme_rekv_offline_vqa',
        eval_script='model/video_qa/eval/eval_egoschema_subset.py'
    ),
    'qaego4d': DatasetConfig(
        name='qaego4d',
        anno_path='data/qaego4d/test_mc.json',
        solver='rekv_offline_vqa',
        eval_script='model/video_qa/eval/eval_multiple_choice.py'
    ),
    'cgbench': DatasetConfig(
        name='cgbench',
        anno_path='data/cgbench/full_mc.json',
        solver='rekv_offline_vqa',
        eval_script='model/video_qa/eval/eval_multiple_choice.py'
    ),
    'activitynet_qa': DatasetConfig(
        name='activitynet_qa',
        anno_path='data/activitynet_qa/test.json',
        solver='rekv_offline_vqa',
        eval_script='video_qa/eval/eval_open_ended.py'
    ),
    'rvs_ego': DatasetConfig(
        name='rvs_ego',
        anno_path='data/rvs/ego/ego4d_oe.json',
        solver='rekv_stream_vqa',
        eval_script='video_qa/eval/eval_open_ended.py'
    ),
    'rvs_movie': DatasetConfig(
        name='rvs_movie',
        anno_path='data/rvs/movie/movienet_oe.json',
        solver='rekv_stream_vqa',
        eval_script='video_qa/eval/eval_open_ended.py'
    ),
}