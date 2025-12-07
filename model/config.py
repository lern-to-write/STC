"""
全局配置管理器 - 优雅地管理所有实验配置
"""
from dataclasses import dataclass, field
from typing import Optional, Literal

import os
@dataclass
class CacheConfig:
    """缓存相关配置"""
    strategy: Literal['none', 'cacher'] = 'cacher'
    update_token_ratio: float = 0.3
    cache_interval=2
    

        

@dataclass
class ModelConfig:
    """模型相关配置"""
    token_per_frame: int = 49
    prune_strategy: str = 'full_tokens'
    encode_chunk_size: int = 1
        




@dataclass
class GlobalConfig:
    """全局配置单例"""
    cache: CacheConfig = field(default_factory=CacheConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    _instance: Optional['GlobalConfig'] = None
    
    @classmethod
    def get_instance(cls) -> 'GlobalConfig':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def initialize_from_args(cls, args):
        instance = cls.get_instance()

        return instance
    
    def to_dict(self):
        return {
            'cache': {
                'strategy': self.cache.strategy,
                'update_token_ratio': self.cache.update_token_ratio,
                'cache_interval': self.cache.cache_interval,
            },
            'model': {
                'token_per_frame': self.model.token_per_frame,
                'prune_strategy': self.model.prune_strategy,
                'encode_chunk_size': self.model.encode_chunk_size,
                
            }
        }
    
    def __str__(self):
        import json
        return json.dumps(self.to_dict(), indent=2)


# 便捷访问函数
def get_config() -> GlobalConfig:
    return GlobalConfig.get_instance()