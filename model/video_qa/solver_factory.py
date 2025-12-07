
from logzero import logger


def create_solver(solver_name, model, processor, args):

    # 延迟导入避免循环依赖
    from .rekv_offline_refactored import ReKVOfflineVQA
    from .rekv_stream_refactored import ReKVStreamVQA
    from .videomme_refactored import VideoMMEReKVOfflineVQA
    
    # Solver映射表
    SOLVER_MAP = {
        'rekv_offline_vqa': ReKVOfflineVQA,
        'videomme_rekv_offline_vqa': VideoMMEReKVOfflineVQA,
        'rekv_stream_vqa': ReKVStreamVQA,
    }
    
    if solver_name not in SOLVER_MAP:
        logger.warning(f"Unknown solver: {solver_name}, falling back to rekv_offline_vqa")
        solver_name = 'rekv_offline_vqa'
    
    solver_class = SOLVER_MAP[solver_name]
    
    return solver_class(model, processor, args)

