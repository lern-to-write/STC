import math
import torch
from typing import Optional, Tuple

from .dot_production_attention import get_multi_stage_dot_production_attention
from model.attention.aks import *
from model.attention.dpc_knn import *
from model.attention.mae_cosine_sim import *


import torch.nn.functional as F


# Allocate a fixed-size block of GPU memory specifically for storing the KV-Cache of the local_window.
class CudaCache:
    def __init__(self, num_units, unit_size, dtype):
        self.num_units = num_units  # n_block
        self.unit_size = unit_size  # block_size * hidden_dim * 2
        self.dtype = dtype
        self.data = torch.empty((num_units, unit_size), device="cuda", dtype=dtype)
        self.idle_set = set(list(range(num_units)))

    def alloc(self):
        assert len(self.idle_set) > 0
        idx = self.idle_set.pop()
        return self.data[idx], idx

    def delete(self, idx):
        assert idx not in self.idle_set
        self.idle_set.add(idx)


# The KV-Cache management unit supports data transfer between the CPU and GPU.
class MemoryUnit:
    # Initialize the KV-Cache management unit and store it on the CPU.
    def __init__(
        self,
        kv: Tuple[torch.Tensor, torch.Tensor],
        cache: CudaCache,
        load_to_cache: bool = False,
        pin_memory: bool = False,
    ):
        self.cache = cache

        if kv[0].is_cuda:
            cpu_data = tuple(_t.contiguous().to("cpu", non_blocking=True) for _t in kv)
        else:
            cpu_data = tuple(_t.contiguous() for _t in kv)

        if pin_memory:
            cpu_data = tuple(_t.pin_memory() for _t in cpu_data)

        if load_to_cache:
            gpu_data, gpu_data_id = cache.alloc()
            gpu_data = gpu_data.view((2,) + kv[0].shape)
            gpu_data[0].copy_(kv[0], non_blocking=True)
            gpu_data[1].copy_(kv[1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            gpu_data, gpu_data_id = None, None
            event = None

        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        self.event = event

    # Load data from the CPU to the GPU and copy it to 'target' when necessary.
    # target: 2x (n_head, n_token, head_dim), on GPU
    def load(self, target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> bool:
        if self.gpu_data is not None:
            if target is not None:
                target[0].copy_(self.gpu_data[0], non_blocking=True)
                target[1].copy_(self.gpu_data[1], non_blocking=True)
                target_event = torch.cuda.Event()
                target_event.record(torch.cuda.current_stream())
            else:
                target_event = None

            return False, target_event

        gpu_data, gpu_data_id = self.cache.alloc()
        gpu_data = gpu_data.view((2,) + self.cpu_data[0].shape)
        if target is not None:
            target[0].copy_(self.cpu_data[0], non_blocking=True)
            target[1].copy_(self.cpu_data[1], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            gpu_data[0].copy_(target[0], non_blocking=True)
            gpu_data[1].copy_(target[1], non_blocking=True)

        else:
            gpu_data[0].copy_(self.cpu_data[0], non_blocking=True)
            gpu_data[1].copy_(self.cpu_data[1], non_blocking=True)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.event = event
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id

        return True, target_event

    # Get the KV-Cache stored on GPU
    def get(self):
        assert self.gpu_data is not None
        self.event.wait()
        return self.gpu_data

    # Clear the KV-Cache stored on GPU
    def offload(self):
        assert self.gpu_data is not None
        self.event.wait()
        self.gpu_data = None
        self.cache.delete(self.gpu_data_id)
        self.gpu_data_id = None

    def calculate_cpu_memory(self):
        return (
            len(self.cpu_data)
            * self.cpu_data[0].numel()
            * self.cpu_data[0].element_size()
        )


# A dynamically growing vector cache on the GPU, used to store representative vectors of video frames.
class VectorTensor:
    # Initialize an empty cache of size (16, hidden_dim) on the GPU.
    def __init__(self, hidden_size, element_dtype, device):
        init_cached_size = 16
        self.data = torch.empty(
            (init_cached_size, hidden_size), dtype=element_dtype, device=device
        )
        self.length = 0
        self.cache_size = init_cached_size
        self.hidden_size = hidden_size

    # Double the size of the cache.
    def append_cache(self):
        new_cache_size = self.cache_size * 2
        data_shape = self.data.shape
        new_data = torch.empty(
            (new_cache_size,) + data_shape[1:],
            device=self.data.device,
            dtype=self.data.dtype,
        )
        new_data[: self.cache_size, ...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    # Append a frame vector to the cache, and expand the cache if it exceeds the current cache size.
    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.size(1) == self.hidden_size, (
            f"{tensor.size(1)}, {self.hidden_size}"
        )
        assert tensor.is_contiguous()

        append_l = tensor.size(0)

        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length : self.length + append_l, ...].copy_(tensor)

        self.length += append_l

    # Get the cached frame vectors
    def get_data(self):
        return self.data[: self.length, ...]

    # 实验代码
    def get_cosine_similarity(self, tensor: torch.Tensor):
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size, (
            f"{tensor.size(0)}, {self.hidden_size}"
        )
        key = self.data[
            : self.length
        ].float()  # (T, D), convert to fp32 to prevent numerical overflow
        query = tensor[None, :].float()  # (1, D)
        logits = torch.matmul(query, key.T)[0]  # (T,)
        assert logits.dim() == 1 and logits.size(0) == self.length
        return logits

    def get_block_query_relevance(self, tensor: torch.Tensor):
        K = self.data[: self.length].float()
        Q = tensor.float()
        Q_norm = torch.nn.functional.normalize(
            Q, p=2, dim=1
        )  # 形状 [token_number, channel]
        K_norm = torch.nn.functional.normalize(
            K, p=2, dim=1
        )  # 形状 [token_number, channel]
        similarity_matrix = torch.mm(
            Q_norm, K_norm.T
        )  # 形状 [token_number, token_number]
        # max_scores = torch.max(similarity_matrix, dim=0).values  # 形状 [token_number]
        mean_scores = torch.mean(similarity_matrix, dim=0)  # 形状 [token_number]
        # scores=max_scores+mean_scores
        scores = mean_scores
        return scores

    def get_global_block_uniquenes(self):
        frame_features = self.data[: self.length].float()  # (num_frames, feature_dim)
        video_feature_mean = frame_features.mean(dim=0)  # (feature_dim,)
        video_feature_mean_expanded = video_feature_mean.unsqueeze(0).expand_as(
            frame_features
        )  # (num_frames, feature_dim)
        cosine_similarities = F.cosine_similarity(
            frame_features, video_feature_mean_expanded, dim=1
        )  # (num_frames,)
        return cosine_similarities

    def get_global_block_L2norm(self, global_k=None):
        if global_k == None:
            frame_features = self.data[
                : self.length
            ].float()  # (num_blocks, feature_dim)
        else:
            frame_features = global_k.float()
        l2norm = torch.norm(frame_features, p=2, dim=1)  # (num_blocks,)
        return l2norm

    def get_block_bias_by_video_length(self):
        return math.log(float(self.length) + 1)

    def get_block_bias_by_memory_decay(self):
        if self.length == 0:
            return torch.tensor([], dtype=self.data.dtype, device=self.data.device)
        # 指数递增，归一化到 [0, 1]
        idx = torch.arange(self.length, dtype=self.data.dtype, device=self.data.device)
        decay = torch.exp(idx / (self.length - 1 + 1e-8))  # 指数递增
        decay = (decay - decay.min()) / (
            decay.max() - decay.min() + 1e-8
        )  # 归一化到 [0, 1]
        return decay

    def get_block_bias_by_memory_decay_linear(self):
        if self.length == 0:
            return torch.tensor([], dtype=self.data.dtype, device=self.data.device)
        idx = torch.arange(self.length, dtype=self.data.dtype, device=self.data.device)
        decay = 1.0 - idx / (self.length - 1 + 1e-8)
        return decay

    def get_block_bias_by_memory_decay_linear_increase(self):
        if self.length == 0:
            return torch.tensor([], dtype=self.data.dtype, device=self.data.device)
        idx = torch.arange(self.length, dtype=self.data.dtype, device=self.data.device)
        decay = idx / (self.length - 1 + 1e-8)  # 递减：前面小，后面大
        return decay

    def get_block_bias_by_memory_decay_power(self, power=2):
        if self.length == 0:
            return torch.tensor([], dtype=self.data.dtype, device=self.data.device)
        idx = torch.arange(self.length, dtype=self.data.dtype, device=self.data.device)
        decay = (self.length - 1 - idx) ** power
        decay = (decay - decay.min()) / (decay.max() - decay.min() + 1e-8)
        return decay

    def get_block_bias_by_memory_decay_log(self):
        if self.length == 0:
            return torch.tensor([], dtype=self.data.dtype, device=self.data.device)
        idx = torch.arange(self.length, dtype=self.data.dtype, device=self.data.device)
        decay = -torch.log(idx + 1)
        decay = (decay - decay.min()) / (decay.max() - decay.min() + 1e-8)
        return decay
    
    
    def calculate_inter_frame_similarity(self,global_k=None) -> torch.Tensor:

        if global_k == None:
            tensor = self.data[: self.length].float()  # (num_blocks, feature_dim)
        else:
            tensor = global_k.float()
        if tensor.dim() != 2:
            raise ValueError(f"输入张量需要是2维的 [frame_number, channel]，但得到的是 {tensor.dim()}维")

        # 检查是否有足够的帧来进行计算
        if tensor.shape[0] < 2:
            # 如果只有一帧或没有帧，可以返回一个全为1的张量
            return torch.ones(tensor.shape[0])

        # 提取用于计算的向量
        # t1 的形状是 [frame_number - 1, channel]，包含从第0帧到倒数第二帧
        t1 = tensor[:-1]
        # t2 的形状是 [frame_number - 1, channel]，包含从第1帧到最后一帧
        t2 = tensor[1:]

        # 沿着 channel 维度 (dim=1) 计算余弦相似度
        # similarities 的形状是 [frame_number - 1]
        similarities = F.cosine_similarity(t1, t2, dim=1)

        # 第一帧没有前一帧，我们给它一个默认分数 1.0
        # 创建一个张量来存放第一帧的分数
        first_frame_score = torch.tensor([0.0], device=tensor.device)

        # 将第一帧的分数和后面计算出的分数拼接起来
        # 最终得到形状为 [frame_number] 的分数张量
        scores = torch.cat([first_frame_score, similarities])

        return 1-scores

    # def calculate_smoothed_frame_similarity(
    #     self, global_k=None, window_size: int = 5
    # ) -> torch.Tensor:
    #     """
    #     计算每一帧与前一帧的余弦相似度，并通过移动平均进行平滑处理。

    #     Args:
    #         features (torch.Tensor): 输入的特征张量，形状为 [frame_number, channel]。
    #         window_size (int): 用于平滑处理的移动平均窗口大小，必须是正奇数。

    #     Returns:
    #         torch.Tensor: 每帧的平滑分数，形状为 [frame_number]。
    #     """
    #     # --- 参数验证 ---

    #     if window_size <= 0 or window_size % 2 == 0:
    #         raise ValueError("window_size 必须是一个正奇数，以确保移动平均是中心对齐的。")
    #     if global_k == None:
    #         features = self.data[: self.length].float()  # (num_blocks, feature_dim)
    #     else:
    #         features = global_k.float()
    #     frame_number = features.shape[0]

    #     # 如果帧数少于2，无法计算相似度，直接返回一个全为1的分数
    #     if frame_number < 2:
    #         return torch.ones(frame_number, device=features.device, dtype=features.dtype)

    #     # --- 步骤 1: 计算每帧与前一帧的余弦相似度 ---
    #     # F.cosine_similarity 需要两个张量进行比较
    #     # features[:-1] 是第 0 帧到第 n-2 帧
    #     # features[1:] 是第 1 帧到第 n-1 帧
    #     # dim=1 表示沿着 channel 维度计算
    #     consecutive_sims = F.cosine_similarity(features[:-1], features[1:], dim=1)

    #     # 第一帧没有“前一帧”，我们通常将其相似度定义为1.0（与自身完美相似）
    #     # 将其拼接到计算结果的最前面
    #     # 确保新张量与输入张量在同一设备和数据类型
    #     first_frame_score = torch.tensor(
    #         [1.0],
    #         device=features.device,
    #         dtype=features.dtype
    #     )
    #     # 拼接后，raw_scores 的形状为 [frame_number]
    #     raw_scores = torch.cat([first_frame_score, consecutive_sims])

    #     # --- 步骤 2: 使用一维卷积 (Conv1D) 实现移动平均平滑 ---
    #     # 这是在PyTorch中最快、最高效的平滑方法

    #     # 1. 准备卷积核：一个权重全为 1/window_size 的核
    #     kernel = torch.ones(
    #         window_size,
    #         device=features.device,
    #         dtype=features.dtype
    #     ) / window_size

    #     # 2. 调整输入和核的形状以适应 conv1d
    #     # conv1d 需要输入形状为 [batch, in_channels, length]
    #     scores_for_conv = raw_scores.view(1, 1, -1)
    #     kernel_for_conv = kernel.view(1, 1, -1)

    #     # 3. 计算 padding 以保持输出长度不变 ("same" padding)
    #     # 因为窗口是中心对齐的，所以两边各填充 (window_size - 1) / 2
    #     padding = (window_size - 1) // 2

    #     # 4. 执行卷积
    #     smoothed_scores_conv = F.conv1d(scores_for_conv, kernel_for_conv, padding=padding)

    #     # 5. 将输出形状从 [1, 1, frame_number] 恢复为 [frame_number]
    #     final_scores = smoothed_scores_conv.squeeze()

    #     return final_scores




    def dpc_knn_select_tokens(self, k: int = 20, n_to_keep: int = 64) -> torch.Tensor:
        video_tensor = self.data[: self.length].float()
        n_tokens, _ = video_tensor.shape
        device = video_tensor.device

        if n_to_keep > n_tokens:
            raise ValueError(
                f"希望保留的token数量 ({n_to_keep}) 不能超过总token数量 ({n_tokens})。"
            )
        if k >= n_tokens:
            raise ValueError(f"K值 ({k}) 必须小于总token数量 ({n_tokens})。")

        # 1. 计算所有token之间的欧氏距离矩阵
        dist_matrix = torch.cdist(video_tensor, video_tensor, p=2)

        # 2. 计算每个token的局部密度 (ρ)
        # 局部密度由与K个最近邻的平均距离决定，距离越小密度越高。
        # 我们取负数，这样密度越高的点，其ρ值也越大。
        sorted_dists, _ = torch.sort(dist_matrix, dim=1)
        # 提取从第1到第k+1的邻居（第0个是自身，距离为0）
        k_nearest_dists = sorted_dists[:, 1 : k + 1]
        rho = -k_nearest_dists.mean(dim=1)

        # 3. 计算每个token的相对距离 (δ)
        # δ是当前token与所有比它密度更高的token之间的最小距离。
        sorted_rho_indices = torch.argsort(rho, descending=True)
        delta = torch.full((n_tokens,), float("inf"), device=device)

        # 遍历按密度排序后的token
        for i in range(1, n_tokens):
            current_idx = sorted_rho_indices[i]
            # 获取所有密度比当前token更高的token的索引
            higher_density_indices = sorted_rho_indices[:i]
            # 计算当前token与所有更高密度token的距离，并取最小值
            delta[current_idx] = dist_matrix[current_idx, higher_density_indices].min()

        # 对于密度最高的token，其δ定义为它与其他所有token的最大距离
        highest_rho_idx = sorted_rho_indices[0]
        delta[highest_rho_idx] = dist_matrix[highest_rho_idx].max()

        # 4. 计算决策值 (γ = ρ * δ) 并选择top-N的token
        # 为了避免量纲影响，先对ρ和δ进行归一化处理
        rho_norm = (rho - rho.min()) / (rho.max() - rho.min() + 1e-8)
        delta_norm = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)

        gamma = rho_norm * delta_norm

        # 根据gamma值降序排序，获取前n_to_keep个token的索引
        top_indices = torch.argsort(gamma, descending=True)

        return top_indices[:n_to_keep]

    def __len__(self):
        return self.length


GLOBAL_STREAM = None


class ContextManager:
    def __init__(
        self,
        position_embedding,
        n_init,
        n_local,
        block_size,
        max_cached_block,
        topk,
        chunk_size,
        exc_block_size,
        fattn: bool = False,
        async_global_stream: bool = False,
        pin_memory: bool = False,
    ):
        self.length = 0  # number of tokens in the KV-Cache
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local
        self.block_size = block_size
        self.max_cached_block = max_cached_block
        self.exc_block_size = exc_block_size
        assert exc_block_size <= n_local  # no global token in input
        self.topk = topk
        self.chunk_size = chunk_size
        self.Attn, _ = get_multi_stage_dot_production_attention(fattn)
        self.fattn = fattn
        self.initialized = False
        self.load_count = 0
        self.async_global_stream = async_global_stream
        self.pin_memory = pin_memory
        global GLOBAL_STREAM
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()

        self.reset_retrieval()

    def _remove_lru_blocks(
        self, u, num_remove: Optional[int] = None, ignore_blocks=None
    ):
        if num_remove is None:
            num_remove = len(self.cached_blocks[u]) - self.max_cached_block

        if num_remove <= 0:
            return

        lst = list(self.cached_blocks[u].items())
        lst.sort(key=lambda x: x[1])

        removed = 0
        for i in range(len(lst)):
            idx = lst[i][0]
            if ignore_blocks is None or (idx not in ignore_blocks):
                self.global_blocks[u][idx].offload()
                self.cached_blocks[u].pop(idx)
                removed += 1

            if removed >= num_remove:
                return

    # handle GQA, k: (batch_size, n_head_kv, length, dim_head) -> (batch_size, n_head, length, dim_head)
    def _from_group_kv(self, tensor):
        # tensor: (batch_size, n_head_kv, length, dim_head)
        assert tensor.dim() == 4
        assert tensor.size(1) == self.num_heads_kv
        if self.num_heads == self.num_heads_kv:
            return tensor
        _, _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view(
            (self.num_units, self.unit_size_kv, 1, length, dim_head)
        )  # (batch_size, n_head_kv, 1, length, dim_head)
        tensor = tensor.expand(
            (self.num_units, self.unit_size_kv, num_group, length, dim_head)
        ).reshape(
            (self.num_units, self.num_heads, length, dim_head)
        )  # (batch_size, n_head, length, dim_head)
        return tensor

    def get_block_k(self, block_k_tensor):
        # 计算每个 block 的代表性向量
        block_k_mean = block_k_tensor.mean(
            dim=-2, keepdim=False
        )  # (batch_size, num_heads, dim_head)
        block_k_flat = block_k_mean.reshape(
            self.num_units, -1
        )  # (batch_size, num_heads * dim_head)
        block_k_expanded = block_k_flat[
            :, None, :
        ]  # (batch_size, 1, num_heads * dim_head)
        return block_k_expanded

    def init(self, local_q, local_k, local_v, global_q, global_k, global_v):
        """
        Only use the metadata of these parameters, such as shape, dtype, and device.
        """
        assert local_q.dim() == 4
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        for _t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert _t.size(0) == batch_size
            assert _t.size(1) == num_heads or _t.size(1) == num_heads_kv
            assert _t.size(2) == len_q
            assert _t.size(3) == dim_head
            assert _t.is_cuda

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.num_units = batch_size
        self.unit_size = num_heads
        self.unit_size_kv = num_heads_kv

        self.global_blocks = [
            [] for _ in range(self.num_units)
        ]  # context memory's KV-Cache: [ batch_size x [memory_unit] ]
        self.cached_blocks = [
            {} for _ in range(self.num_units)
        ]  # relavency scores of blocks: batch_size x {block_id: block_score}
        self.num_global_block = 0

        # context memory's representative keys: batch_size x (n_blocks, hidden_dim)
        self.block_k = [
            VectorTensor(dim_head * self.unit_size, global_k.dtype, global_k.device)
            for _ in range(self.num_units)
        ]

        ####################################################################################################################################

        self.block_L2norm = [
            VectorTensor(dim_head * self.unit_size, global_k.dtype, global_k.device)
            for _ in range(self.num_units)
        ]

        self.block_uniquenes = [
            VectorTensor(dim_head * self.unit_size, global_k.dtype, global_k.device)
            for _ in range(self.num_units)
        ]

        self.block_bias_by_video_length = [
            VectorTensor(dim_head * self.unit_size, global_k.dtype, global_k.device)
            for _ in range(self.num_units)
        ]

        self.block_bias_by_memory_decay = [
            VectorTensor(dim_head * self.unit_size, global_k.dtype, global_k.device)
            for _ in range(self.num_units)
        ]
        ####################################################################################################################################

        # local KV
        self.local_k = torch.empty(
            (self.num_units, self.unit_size_kv, 0, dim_head),
            dtype=local_k.dtype,
            device=local_k.device,
        )  # (batch_size, n_head_kv, 0, dim_head)
        self.local_v = torch.empty(
            (self.num_units, self.unit_size_kv, 0, dim_head),
            dtype=local_v.dtype,
            device=local_v.device,
        )

        # global KV that are not yet processed into blocks.
        # 2 x (batch_size, n_head_kv, length, dim_head)
        self.global_remainder = (
            torch.empty(
                (self.num_units, self.unit_size_kv, 0, dim_head),
                dtype=global_k.dtype,
                device=global_k.device,
            ),
            torch.empty(
                (self.num_units, self.unit_size_kv, 0, dim_head),
                dtype=global_v.dtype,
                device=global_v.device,
            ),
        )

        # init KV
        self.init_k = torch.empty(
            (self.num_units, self.unit_size_kv, 0, dim_head),
            dtype=global_k.dtype,
            device=global_k.device,
        )
        self.init_v = torch.empty(
            (self.num_units, self.unit_size_kv, 0, dim_head),
            dtype=global_k.dtype,
            device=global_k.device,
        )
        self.init_exc = False
        self.dtype = local_q.dtype
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )

        # buffering global KV during attention computations
        # (2, batch_size, n_head_kv, L, dim_head)
        # L = n_init + n_retrieve
        buffer_len = self.topk * self.block_size + self.n_init
        self.global_buffer = torch.zeros(
            (2, self.num_units, self.unit_size_kv, buffer_len, dim_head),
            dtype=global_k.dtype,
            device=global_k.device,
        )
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0
        self.cuda_cache = CudaCache(
            self.max_cached_block * self.num_units,
            self.unit_size_kv * self.block_size * dim_head * 2,
            local_k.dtype,
        )  # (max_cached_block * batch_size, block_size * D * 2)

        self.initialized = True

    def set_retrieval(self):
        self.to_retrieve = True

    def reset_retrieval(self):
        self.similarity = None
        self.retrieved_block_indices = None
        self.to_retrieve = False

    def set_retrieved_block_indices(self, retrieved_block_indices):
        # retrieved_block_indices (list): batch_size x n_frames
        if isinstance(retrieved_block_indices, torch.Tensor):
            retrieved_block_indices = retrieved_block_indices.cpu().tolist()
        self.retrieved_block_indices = retrieved_block_indices

    def get_retrieved_kv(self, query=None):
        """retrieve context blocks with retrieved_block_indices
        query: (batch_size, num_heads, length, dim_head)
        return [init_k, retrieved_k] and the respective v
        """

        if (
            query is not None
        ):  # retrieve based on the attention score between query and context's representative keys
            # block_topk = self._calc_block_topk(query)
            # block_topk = self.l2norm_calc_block_topk(query)
            # block_topk = self.aks_test_calc_block_topk(query)

            block_topk = self.dpc_knn_calc_block_topk(query)
            # block_topk = self.aks_calc_block_topk(query)

            self.set_retrieved_block_indices(retrieved_block_indices=block_topk)

        assert len(self.retrieved_block_indices) == self.num_units

        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        with torch.cuda.stream(GLOBAL_STREAM):
            if self.init_exc:  # init KV were loaded in global_h_k, context KV were offloaded in global_blocks
                # offload LRU blocks
                for u in range(self.num_units):
                    num_remove = len(self.cached_blocks[u]) - self.max_cached_block
                    for b_idx in self.retrieved_block_indices[u]:
                        if b_idx not in self.cached_blocks[u]:
                            num_remove += 1
                    self._remove_lru_blocks(
                        u, num_remove, self.retrieved_block_indices[u]
                    )

                self.load_count += 1
                for u in range(self.num_units):
                    for b_idx in self.retrieved_block_indices[u]:
                        self.cached_blocks[u][b_idx] = self.load_count

                # no need to load init KV
                init_st = 0
                init_ed = init_st + self.init_k.size(-2)
                ed = init_ed
                assert (
                    self.global_buffer_init_st == init_st
                    or self.global_buffer_init_ed == init_ed
                )

                # load retrieved context KV
                for u in range(self.num_units):
                    # assert len(self.retrieved_block_indices[u]) == block_num
                    assert (
                        self.retrieved_block_indices[u][-1] < self.num_global_block
                    ), f"{self.retrieved_block_indices[u][-1]}, {self.num_global_block}"
                    for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                        # load global_blocks[u][b_idx] onto GPU and make a copy to (global_h_k, global_h_v)
                        st = init_ed + cnt * self.block_size
                        ed = st + self.block_size
                        self.global_blocks[u][b_idx].load(
                            (global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :])
                        )

                        # In [3]: global_h_k[u, :, st:ed, :].shape Out[3]: torch.Size([4, 98, 128])

            else:  # init KV and context are in self.global_remainder
                # load init KV
                init_st = 0
                init_ed = init_st + self.n_init
                global_h_k[:, :, init_st:init_ed] = self.global_remainder[0][
                    :, :, init_st:init_ed
                ]
                global_h_v[:, :, init_st:init_ed] = self.global_remainder[1][
                    :, :, init_st:init_ed
                ]
                ed = init_ed

                # load retrieved context KV
                for u in range(self.num_units):
                    # assert len(self.retrieved_block_indices[u]) == block_num
                    for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                        remainder_st = init_ed + b_idx * self.block_size
                        remainder_ed = remainder_st + self.block_size
                        if remainder_st >= self.global_remainder[0].size(2):
                            break
                        st = init_ed + cnt * self.block_size
                        ed = st + self.block_size
                        global_h_k[u, :, st:ed] = self.global_remainder[0][
                            u, :, remainder_st:remainder_ed
                        ]
                        global_h_v[u, :, st:ed] = self.global_remainder[1][
                            u, :, remainder_st:remainder_ed
                        ]

            global_h_k = global_h_k[:, :, :ed, :]
            global_h_v = global_h_v[:, :, :ed, :]

            # assert global_h_k.size(-2) == global_h_v.size(-2) == self.n_init + block_num * self.block_size

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        assert global_h_k.size(-2) <= self.n_init + self.n_local
        return global_h_k, global_h_v

    def aks_calc_block_topk(self, global_h_q):
        #########################################################################################################################
        global_h_q = global_h_q.mean(
            dim=2, keepdim=False
        )  # (batch_size, num_heads, dim_head)
        assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
        global_h_q = global_h_q.reshape(
            self.num_units, self.dim_head * self.unit_size
        )  # (batch_size, dim_head * num_heads)

        logits = None

        if self.num_global_block <= self.topk:
            if not self.init_exc:  # The local window has not yet been filled, i.e., KV-Cache offloading has not been activated. Retrieval needs to be performed within the local window.
                assert self.global_remainder[0].size(-2) > self.n_init, (
                    f"{self.global_remainder[0].shape}"
                )
                global_k = self.global_remainder[0][
                    :, :, self.n_init :, :
                ]  # (batch_size, n_head_kv, length - n_init, dim_head)
                global_k = self._from_group_kv(
                    global_k
                )  # (batch_size, num_heads, length - n_init, dim_head)

                assert global_k.size(-2) % self.block_size == 0, f"{global_k.shape}"
                block_num = (
                    global_k.size(-2) // self.block_size
                )  # number of frames in local window
                if block_num <= self.topk:
                    ret = [list(range(block_num)) for _ in range(self.num_units)]

                    selected_frames = [
                        list(range(len(self.global_blocks[0])))
                        for _ in range(self.num_units)
                    ]
                else:
                    global_k = global_k.transpose(
                        1, 2
                    )  # (batch_size, length - n_init, num_heads, dim_head)
                    global_k = global_k.reshape(
                        self.num_units,
                        block_num,
                        self.block_size,
                        self.unit_size * self.dim_head,
                    )  # (batch_size, block_num, block_size, dim)
                    global_k = global_k.mean(
                        dim=-2, keepdim=False
                    )  # (batch_size, block_num, dim)
                    block_query_relevance = torch.matmul(
                        global_k, global_h_q[:, :, None]
                    ).squeeze(dim=-1)  # (batch_size, block_num)
                    logits = block_query_relevance

                    ###################################################################################################
                    selected_frames = adaptive_keyframe_sampling(
                        scores=block_query_relevance,
                        max_frames=64,  # 选择32帧
                        t1=0.8,  # 相关性阈值
                        t2=-100,  # 方差阈值（设置为负值使其更容易满足条件）
                        max_depth=5,  # 最大递归深度
                    )

            else:  # The local window is already filled, but the number of input frames is less than 'topk'.
                ret = [
                    list(range(len(self.global_blocks[0])))
                    for _ in range(self.num_units)
                ]
                selected_frames = [
                    list(range(len(self.global_blocks[0])))
                    for _ in range(self.num_units)
                ]

        else:
            ###########################################################################################################################################################################################################
            # 实验代码
            block_query_relevance = torch.stack(
                [
                    self.block_k[u].get_cosine_similarity(global_h_q[u])
                    for u in range(self.num_units)
                ]
            )  # (batch_size, block_num)

            selected_frames = adaptive_keyframe_sampling(
                scores=block_query_relevance,
                max_frames=64,  # 选择32帧
                t1=0.8,  # 相关性阈值
                t2=-100,  # 方差阈值（设置为负值使其更容易满足条件）
                max_depth=5,  # 最大递归深度
            )
            # random_tensor = torch.randn_like(block_query_relevance)

        # In [1]: logits.shape Out[1]: torch.Size([1, 600])        In [4]: len(self.block_k[0]) Out[4]: 600         In [2]: global_h_q[0].shape  Out[2]: torch.Size([3584])

        ############################################################################################################################################################################################################

        if logits is not None:
            self.similarity = logits
            assert self.topk % self.chunk_size == 0
            remainder_size = logits.shape[1] % self.chunk_size
            chunked_logits = (
                logits[:, : logits.shape[1] - remainder_size]
                .reshape(self.num_units, -1, self.chunk_size)
                .mean(dim=-1)
            )  # (batch_size, block_num // chunk_size)
            if remainder_size > 0:
                remainder_logits = logits[:, -remainder_size:].mean(
                    dim=-1, keepdim=True
                )  # (batch_size, 1)
                chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=1)
                ################################################################################################################################################################
            ret = chunked_logits.topk(
                self.topk // self.chunk_size, dim=1, largest=True
            ).indices
            ##########################################################################################################################################################################
            ret = ret.sort(dim=1)[0][:, :, None]  # (batch_size, topk//chunk_size, 1)
            ret = (
                ret * self.chunk_size
                + torch.arange(self.chunk_size, device=ret.device)[None, None, :]
            )  # (batch_size, topk//chunk_size, chunk_size)
            ret = ret.reshape(self.num_units, -1)  # (batch_size, topk)
            ret = ret.cpu().tolist()

            # NOTE: The last chunk might cause an index overflow
            for u in range(self.num_units):
                ret[u] = list(filter(lambda idx: idx < logits.shape[1], ret[u]))

        return selected_frames
    
    
    
    
    def aks_test_calc_block_topk(self, global_h_q):
        #########################################################################################################################
        global_h_q = global_h_q.mean(
            dim=2, keepdim=False
        )  # (batch_size, num_heads, dim_head)
        assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
        global_h_q = global_h_q.reshape(
            self.num_units, self.dim_head * self.unit_size
        )  # (batch_size, dim_head * num_heads)

        logits = None

        if self.num_global_block <= self.topk:
            if not self.init_exc:  # The local window has not yet been filled, i.e., KV-Cache offloading has not been activated. Retrieval needs to be performed within the local window.
                assert self.global_remainder[0].size(-2) > self.n_init, (
                    f"{self.global_remainder[0].shape}"
                )
                global_k = self.global_remainder[0][
                    :, :, self.n_init :, :
                ]  # (batch_size, n_head_kv, length - n_init, dim_head)
                global_k = self._from_group_kv(
                    global_k
                )  # (batch_size, num_heads, length - n_init, dim_head)

                assert global_k.size(-2) % self.block_size == 0, f"{global_k.shape}"
                block_num = (
                    global_k.size(-2) // self.block_size
                )  # number of frames in local window
                if block_num <= self.topk:
                    ret = [list(range(block_num)) for _ in range(self.num_units)]

                    selected_frames = [
                        list(range(len(self.global_blocks[0])))
                        for _ in range(self.num_units)
                    ]
                else:
                    global_k = global_k.transpose(
                        1, 2
                    )  # (batch_size, length - n_init, num_heads, dim_head)
                    global_k = global_k.reshape(
                        self.num_units,
                        block_num,
                        self.block_size,
                        self.unit_size * self.dim_head,
                    )  # (batch_size, block_num, block_size, dim)
                    global_k = global_k.mean(
                        dim=-2, keepdim=False
                    )  # (batch_size, block_num, dim)
                    
                    
                    block_query_relevance = torch.matmul(
                        global_k, global_h_q[:, :, None]
                    ).squeeze(dim=-1)  # (batch_size, block_num)
                    
                    
                    l2_NORM = torch.stack(
                        [
                            self.block_k[u].get_global_block_L2norm(global_k[u])
                            for u in range(self.num_units)
                        ]
                    )
                        # 归一化 l2_NORM
                    epsilon=1e-8
                    l2_min = l2_NORM.min()
                    l2_max = l2_NORM.max()
                    l2_normalized = (l2_NORM - l2_min) / (l2_max - l2_min + epsilon)

                    # 归一化 block_query_relevance
                    bqr_min = block_query_relevance.min()
                    bqr_max = block_query_relevance.max()
                    bqr_normalized = (block_query_relevance - bqr_min) / (bqr_max - bqr_min + epsilon)

                    # 相加并返回结果
                    block_score = l2_normalized + bqr_normalized
                    
                    logits = block_query_relevance

                    ###################################################################################################
                    selected_frames = adaptive_keyframe_sampling(
                        scores=block_score,
                        max_frames=64,  # 选择32帧
                        t1=0.8,  # 相关性阈值
                        t2=-100,  # 方差阈值（设置为负值使其更容易满足条件）
                        max_depth=5,  # 最大递归深度
                    )

            else:  # The local window is already filled, but the number of input frames is less than 'topk'.
                ret = [
                    list(range(len(self.global_blocks[0])))
                    for _ in range(self.num_units)
                ]
                selected_frames = [
                    list(range(len(self.global_blocks[0])))
                    for _ in range(self.num_units)
                ]

        else:
            ###########################################################################################################################################################################################################
            # 实验代码
            block_query_relevance = torch.stack(
                [
                    self.block_k[u].get_cosine_similarity(global_h_q[u])
                    for u in range(self.num_units)
                ]
            )  # (batch_size, block_num)

            l2_NORM = torch.stack(
            [
                self.block_k[u].get_global_block_L2norm()
                for u in range(self.num_units)
            ])
                # 归一化 l2_NORM
            epsilon=1e-8
            l2_min = l2_NORM.min()
            l2_max = l2_NORM.max()
            l2_normalized = (l2_NORM - l2_min) / (l2_max - l2_min + epsilon)

            # 归一化 block_query_relevance
            bqr_min = block_query_relevance.min()
            bqr_max = block_query_relevance.max()
            bqr_normalized = (block_query_relevance - bqr_min) / (bqr_max - bqr_min + epsilon)

            # 相加并返回结果
            block_score = l2_normalized + bqr_normalized


            selected_frames = adaptive_keyframe_sampling(
                scores=block_score,
                max_frames=64,  # 选择32帧
                t1=0.8,  # 相关性阈值
                t2=-100,  # 方差阈值（设置为负值使其更容易满足条件）
                max_depth=5,  # 最大递归深度
            )
            # random_tensor = torch.randn_like(block_query_relevance)
            logits = block_score

        # In [1]: logits.shape Out[1]: torch.Size([1, 600])        In [4]: len(self.block_k[0]) Out[4]: 600         In [2]: global_h_q[0].shape  Out[2]: torch.Size([3584])

        ############################################################################################################################################################################################################

        if logits is not None:
            self.similarity = logits
            assert self.topk % self.chunk_size == 0
            remainder_size = logits.shape[1] % self.chunk_size
            chunked_logits = (
                logits[:, : logits.shape[1] - remainder_size]
                .reshape(self.num_units, -1, self.chunk_size)
                .mean(dim=-1)
            )  # (batch_size, block_num // chunk_size)
            if remainder_size > 0:
                remainder_logits = logits[:, -remainder_size:].mean(
                    dim=-1, keepdim=True
                )  # (batch_size, 1)
                chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=1)
                ################################################################################################################################################################
            ret = chunked_logits.topk(
                self.topk // self.chunk_size, dim=1, largest=True
            ).indices
            ##########################################################################################################################################################################
            ret = ret.sort(dim=1)[0][:, :, None]  # (batch_size, topk//chunk_size, 1)
            ret = (
                ret * self.chunk_size
                + torch.arange(self.chunk_size, device=ret.device)[None, None, :]
            )  # (batch_size, topk//chunk_size, chunk_size)
            ret = ret.reshape(self.num_units, -1)  # (batch_size, topk)
            ret = ret.cpu().tolist()

            # NOTE: The last chunk might cause an index overflow
            for u in range(self.num_units):
                ret[u] = list(filter(lambda idx: idx < logits.shape[1], ret[u]))

        return selected_frames

        # Get the indices of the top-k vectors in self.block_k[u] that have the highest similarity with global_h_q[u].

    # ret: batch_size x topk
    def dpc_knn_calc_block_topk(self, global_h_q):
        global_h_q = global_h_q.mean(
            dim=2, keepdim=False
        )  # (batch_size, num_heads, dim_head)
        assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
        global_h_q = global_h_q.reshape(
            self.num_units, self.dim_head * self.unit_size
        )  # (batch_size, dim_head * num_heads)

        logits = None

        if self.num_global_block <= self.topk:
            if not self.init_exc:  # The local window has not yet been filled, i.e., KV-Cache offloading has not been activated. Retrieval needs to be performed within the local window.
                assert self.global_remainder[0].size(-2) > self.n_init, (
                    f"{self.global_remainder[0].shape}"
                )
                global_k = self.global_remainder[0][
                    :, :, self.n_init :, :
                ]  # (batch_size, n_head_kv, length - n_init, dim_head)
                global_k = self._from_group_kv(
                    global_k
                )  # (batch_size, num_heads, length - n_init, dim_head)

                assert global_k.size(-2) % self.block_size == 0, f"{global_k.shape}"
                block_num = (
                    global_k.size(-2) // self.block_size
                )  # number of frames in local window
                if block_num <= self.topk:
                    ret = [list(range(block_num)) for _ in range(self.num_units)]
                    dpc_knn_selected_indice = [
                        list(range(block_num)) for _ in range(self.num_units)
                    ]

                else:
                    global_k = global_k.transpose(
                        1, 2
                    )  # (batch_size, length - n_init, num_heads, dim_head)
                    global_k = global_k.reshape(
                        self.num_units,
                        block_num,
                        self.block_size,
                        self.unit_size * self.dim_head,
                    )  # (batch_size, block_num, block_size, dim)
                    global_k = global_k.mean(
                        dim=-2, keepdim=False
                    )  # (batch_size, block_num, dim)

                    dpc_knn_selected_indice = (
                        dpc_knn_select_tokens_batched(global_k, k=20, n_to_keep=64)
                        .cpu()
                        .tolist()
                    )

            else:  # The local window is already filled, but the number of input frames is less than 'topk'.
                dpc_knn_selected_indice = [
                    list(range(len(self.global_blocks[0])))
                    for _ in range(self.num_units)
                ]

        else:
            dpc_knn_selected_indice = torch.stack(
                [self.block_k[u].dpc_knn_select_tokens() for u in range(self.num_units)]
            )  # (batch_size, block_num)
            dpc_knn_selected_indice = dpc_knn_selected_indice.cpu().tolist()

        if logits is not None:
            self.similarity = logits
            assert self.topk % self.chunk_size == 0
            remainder_size = logits.shape[1] % self.chunk_size
            chunked_logits = (
                logits[:, : logits.shape[1] - remainder_size]
                .reshape(self.num_units, -1, self.chunk_size)
                .mean(dim=-1)
            )  # (batch_size, block_num // chunk_size)
            if remainder_size > 0:
                remainder_logits = logits[:, -remainder_size:].mean(
                    dim=-1, keepdim=True
                )  # (batch_size, 1)
                chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=1)
                ################################################################################################################################################################
            ret = chunked_logits.topk(
                self.topk // self.chunk_size, dim=1, largest=True
            ).indices
            ret = ret.sort(dim=1)[0][:, :, None]  # (batch_size, topk//chunk_size, 1)
            ret = (
                ret * self.chunk_size
                + torch.arange(self.chunk_size, device=ret.device)[None, None, :]
            )  # (batch_size, topk//chunk_size, chunk_size)
            ret = ret.reshape(self.num_units, -1)  # (batch_size, topk)
            ret = ret.cpu().tolist()

            # NOTE: The last chunk might cause an index overflow
            for u in range(self.num_units):
                ret[u] = list(filter(lambda idx: idx < logits.shape[1], ret[u]))

        return dpc_knn_selected_indice


    def l2norm_calc_block_topk(self, global_h_q):
        #########################################################################################################################
        global_h_q = global_h_q.mean(dim=2, keepdim=False)  # (batch_size, num_heads, dim_head)
        assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
        global_h_q = global_h_q.reshape(self.num_units, self.dim_head * self.unit_size)  # (batch_size, dim_head * num_heads)

        logits = None

        if self.num_global_block <= self.topk:
            if not self.init_exc:  # The local window has not yet been filled, i.e., KV-Cache offloading has not been activated. Retrieval needs to be performed within the local window.
                assert self.global_remainder[0].size(-2) > self.n_init, f'{self.global_remainder[0].shape}'
                global_k = self.global_remainder[0][:, :, self.n_init:, :]  # (batch_size, n_head_kv, length - n_init, dim_head)
                global_k = self._from_group_kv(global_k)  # (batch_size, num_heads, length - n_init, dim_head)

                assert global_k.size(-2) % self.block_size == 0, f'{global_k.shape}'
                block_num = global_k.size(-2) // self.block_size  # number of frames in local window
                if block_num <= self.topk:
                    ret = [list(range(block_num)) for _ in range(self.num_units)]

                else:
                    global_k = global_k.transpose(1, 2)  # (batch_size, length - n_init, num_heads, dim_head)
                    global_k = global_k.reshape(self.num_units, block_num, self.block_size, self.unit_size * self.dim_head)  # (batch_size, block_num, block_size, dim)
                    global_k = global_k.mean(dim=-2, keepdim=False)  # (batch_size, block_num, dim)

                    l2_NORM = torch.stack(
                        [
                            self.block_k[u].get_global_block_L2norm(
                                global_k=global_k[u]
                            )
                            for u in range(self.num_units)
                        ]
                    )  # (batch_size, block_num)
                    logits = l2_NORM

            else:  # The local window is already filled, but the number of input frames is less than 'topk'.
                ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]

        else:
            ###########################################################################################################################################################################################################

            ###########################################################################################################################################################################################################

            # (batch_size, block_num)

            l2_NORM = torch.stack(
                [
                    self.block_k[u].get_global_block_L2norm()
                    for u in range(self.num_units)
                ]
            )  # (batch_size, block_num)
            # block_uniquenes = torch.stack([self.block_k[u].get_global_block_uniquenes() for u in range(self.num_units)])  # (batch_size, block_num)
            logits = l2_NORM

        if logits is not None:
            self.similarity = logits
            assert self.topk % self.chunk_size == 0
            remainder_size = logits.shape[1] % self.chunk_size
            chunked_logits = logits[:, :logits.shape[1]-remainder_size].reshape(self.num_units, -1, self.chunk_size).mean(dim=-1)  # (batch_size, block_num // chunk_size)
            if remainder_size > 0:
                remainder_logits = logits[:, -remainder_size:].mean(dim=-1, keepdim=True)  # (batch_size, 1)
                chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=1)
            ##########################################################################################################################################################################
            ret = chunked_logits.topk(self.topk//self.chunk_size, dim=1,largest=True).indices
            ##########################################################################################################################################################################
            ret = ret.sort(dim=1)[0][:, :, None]  # (batch_size, topk//chunk_size, 1)
            ret = ret * self.chunk_size + torch.arange(self.chunk_size, device=ret.device)[None, None, :]  # (batch_size, topk//chunk_size, chunk_size)
            ret = ret.reshape(self.num_units, -1)  # (batch_size, topk)
            ret = ret.cpu().tolist()

            # NOTE: The last chunk might cause an index overflow
            for u in range(self.num_units):
                ret[u] = list(filter(lambda idx: idx < logits.shape[1], ret[u]))

        return ret

    def _calc_block_topk(self, global_h_q):
        #########################################################################################################################
        global_h_q = global_h_q.mean(
            dim=2, keepdim=False
        )  # (batch_size, num_heads, dim_head)
        assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
        global_h_q = global_h_q.reshape(
            self.num_units, self.dim_head * self.unit_size
        )  # (batch_size, dim_head * num_heads)

        ##########################################################################################################################################

        # batch_size, num_heads, token_number,dim_head=global_h_q.shape
        # x = global_h_q.transpose(1, 2)  # [batch_size, token_number, num_heads, dim_head]
        # global_h_q = x.reshape(batch_size, token_number, -1)

        # # 计算L2范数并筛选前50%

        # norms = torch.norm(global_h_q, p=2, dim=2)  # (batch_size, token_number)
        # k = token_number // 4  # 保留一半数量
        # topk_indices = torch.topk(norms, k, dim=1,largest=True).indices  # (batch_size, k)

        # # 按索引排序后提取特征
        # topk_indices = topk_indices.sort(dim=1).values  # 保持序列顺序
        # global_h_q = torch.gather(
        #     global_h_q,
        #     1,
        #     topk_indices.unsqueeze(-1).expand(-1, -1, global_h_q.shape[-1])
        # )  # (batch_size, k, dim)

        ###################################################################################################################################
        logits = None

        if self.num_global_block <= self.topk:
            if not self.init_exc:  # The local window has not yet been filled, i.e., KV-Cache offloading has not been activated. Retrieval needs to be performed within the local window.
                assert self.global_remainder[0].size(-2) > self.n_init, (
                    f"{self.global_remainder[0].shape}"
                )
                global_k = self.global_remainder[0][
                    :, :, self.n_init :, :
                ]  # (batch_size, n_head_kv, length - n_init, dim_head)
                global_k = self._from_group_kv(
                    global_k
                )  # (batch_size, num_heads, length - n_init, dim_head)

                assert global_k.size(-2) % self.block_size == 0, f"{global_k.shape}"
                block_num = (
                    global_k.size(-2) // self.block_size
                )  # number of frames in local window
                if block_num <= self.topk:
                    ret = [list(range(block_num)) for _ in range(self.num_units)]

                else:
                    global_k = global_k.transpose(
                        1, 2
                    )  # (batch_size, length - n_init, num_heads, dim_head)
                    global_k = global_k.reshape(
                        self.num_units,
                        block_num,
                        self.block_size,
                        self.unit_size * self.dim_head,
                    )  # (batch_size, block_num, block_size, dim)
                    global_k = global_k.mean(
                        dim=-2, keepdim=False
                    )  # (batch_size, block_num, dim)

                    logits = torch.matmul(global_k, global_h_q[:, :, None]).squeeze(dim=-1)  # (batch_size, block_num)

            else:  # The local window is already filled, but the number of input frames is less than 'topk'.
                ret = [
                    list(range(len(self.global_blocks[0])))
                    for _ in range(self.num_units)
                ]

        else:
            ###########################################################################################################################################################################################################

            ###########################################################################################################################################################################################################

            # (batch_size, block_num)

            # block_query_relevance = torch.stack([self.block_k[u].get_block_query_relevance(tensor=global_h_q[u]) for u in range(self.num_units)])  # (batch_size, block_num)
            # block_uniquenes = torch.stack([self.block_k[u].get_global_block_uniquenes() for u in range(self.num_units)])  # (batch_size, block_num)
            mae_cosine_score = torch.stack(
                [
                    self.block_k[u].get_cosine_similarity(tensor=global_h_q[u])
                    for u in range(self.num_units)
                ]
            )  # (batch_size, block_num)
            logits = mae_cosine_score

        if logits is not None:
            self.similarity = logits
            assert self.topk % self.chunk_size == 0
            remainder_size = logits.shape[1] % self.chunk_size
            chunked_logits = (
                logits[:, : logits.shape[1] - remainder_size]
                .reshape(self.num_units, -1, self.chunk_size)
                .mean(dim=-1)
            )  # (batch_size, block_num // chunk_size)
            if remainder_size > 0:
                remainder_logits = logits[:, -remainder_size:].mean(
                    dim=-1, keepdim=True
                )  # (batch_size, 1)
                chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=1)
            ##########################################################################################################################################################################
            ret = chunked_logits.topk(
                self.topk // self.chunk_size, dim=1, largest=True
            ).indices
            ##########################################################################################################################################################################
            ret = ret.sort(dim=1)[0][:, :, None]  # (batch_size, topk//chunk_size, 1)
            ret = (
                ret * self.chunk_size
                + torch.arange(self.chunk_size, device=ret.device)[None, None, :]
            )  # (batch_size, topk//chunk_size, chunk_size)
            ret = ret.reshape(self.num_units, -1)  # (batch_size, topk)
            ret = ret.cpu().tolist()

            # NOTE: The last chunk might cause an index overflow
            for u in range(self.num_units):
                ret[u] = list(filter(lambda idx: idx < logits.shape[1], ret[u]))

        return ret

    # load init KV
    def get_global_hidden_and_mask(self, exc_length):
        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        global_remainder_ed = self._global_remainder_ed + exc_length
        global_remainder_st = self._global_remainder_st
        global_remainder_len = global_remainder_ed - global_remainder_st

        # prepare init KV-Cache until it's full
        if not self.init_exc and global_remainder_len > self.n_local:
            global_k = self.global_remainder[0]
            global_v = self.global_remainder[1]

            # append_init_len = min(
            #     self.n_init - self.init_k.size(-2),
            #     global_remainder_len - self.n_local
            # )

            ####################################################################
            append_init_len = self.n_init - self.init_k.size(-2)

            #############################################################################################
            self.init_k = torch.cat(
                (
                    self.init_k,
                    global_k[
                        :,
                        :,
                        global_remainder_st : global_remainder_st + append_init_len,
                        :,
                    ],
                ),
                dim=-2,
            )
            self.init_v = torch.cat(
                (
                    self.init_v,
                    global_v[
                        :,
                        :,
                        global_remainder_st : global_remainder_st + append_init_len,
                        :,
                    ],
                ),
                dim=-2,
            )
            global_remainder_st += append_init_len
            global_remainder_len -= append_init_len

            if self.init_k.size(-2) == self.n_init:
                self.init_exc = True  # init KV-Cache is full

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

        # load init KV
        init_st = 0
        init_ed = init_st + self.init_k.size(-2)
        if (
            self.global_buffer_init_st != init_st
            or self.global_buffer_init_ed != init_ed
        ):  # init KV haven't been loaded into global_h_kv
            global_h_k[:, :, init_st:init_ed, :].copy_(self.init_k, non_blocking=True)
            global_h_v[:, :, init_st:init_ed, :].copy_(self.init_v, non_blocking=True)

        self.global_buffer_init_st = init_st
        self.global_buffer_init_ed = init_ed

        global_h_k = global_h_k[:, :, :init_ed, :]
        global_h_v = global_h_v[:, :, :init_ed, :]

        return global_h_k, global_h_v

    def _append(
        self,
        local_q,
        local_k,
        local_v,
        global_q,
    ):
        """calculate attention results

        Args:
            local_q (_type_): (batch_size, num_heads, length, dim_head)
            local_k (_type_): (batch_size, num_heads, length, dim_head)
            local_v (_type_): (batch_size, num_heads, length, dim_head)
            global_q (_type_): (batch_size, num_heads, length, dim_head)

        Returns:
            chunk_o: (batch_size, num_heads, length, dim_head)
        """

        # apply RoPE to input QKV
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v

        # input Q attends to input + local KV
        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(
            local_h_q,
            local_h_k,
            local_h_v,
            get_score=False,
            sliding_window=self.n_local,
        )

        # load init KV
        with torch.cuda.stream(GLOBAL_STREAM):
            global_h_q = global_q
            global_h_k, global_h_v = self.get_global_hidden_and_mask(
                exc_length=global_q.size(-2)
            )

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        # input Q attends to init KV
        attn.append(
            global_h_q,
            global_h_k,
            global_h_v,
            end=True,  # the final append operation
            get_score=False,
            sliding_window=None,
            complement_sliding_window=True,
        )

        o, _ = attn.get_result()

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        return o.view((self.batch_size, self.num_heads, -1, self.dim_head))

    def _append_global(self):
        """offload context memory"""

        global_remainder_ed = self._global_remainder_ed
        global_remainder_st = self._global_remainder_st

        global_remainder_len = global_remainder_ed - global_remainder_st

        # offload context KV to CPU
        if self.init_exc:
            assert global_remainder_len % self.block_size == 0, (
                f"global_remainder_len: {global_remainder_len}, block_size: {self.block_size}"
            )
            while global_remainder_len > 0:
                global_remainder_len -= self.block_size

                # Context KV-Cache
                for u in range(self.num_units):
                    self.global_blocks[u].append(
                        (
                            MemoryUnit(
                                (
                                    self.global_remainder[0][
                                        u,
                                        :,
                                        global_remainder_st : global_remainder_st
                                        + self.block_size,
                                        :,
                                    ],
                                    self.global_remainder[1][
                                        u,
                                        :,
                                        global_remainder_st : global_remainder_st
                                        + self.block_size,
                                        :,
                                    ],
                                ),
                                self.cuda_cache,
                                False,
                                self.pin_memory,
                            )
                        )
                    )

                # NOTE: the average of global_remainder is used as the representative vector.
                global_block_k = self.global_remainder[0][
                    :, :, global_remainder_st : global_remainder_st + self.block_size, :
                ]
                global_block_k = self._from_group_kv(
                    global_block_k
                )  # (batch_size, num_heads, length, dim_head)

                global_block_k = self.get_block_k(global_block_k)

                for u in range(self.num_units):
                    self.block_k[u].append(global_block_k[u])

                    ###################################################################################################################################

                self.num_global_block += 1
                global_remainder_st += self.block_size

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

    def append(
        self,
        local_q,
        local_k,
        local_v,
        global_q,
        global_k,
        global_v,
    ):
        # Pre-allocate GPU Memory.
        if not self.initialized:
            self.init(local_q, local_k, local_v, global_q, global_k, global_v)

        input_length = local_q.size(-2)

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        # append local KV
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        kv_length = self.local_k.size(-2)

        # append global remainder
        with torch.cuda.stream(GLOBAL_STREAM):
            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)

            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
            )

        # apply RoPE to global_q
        with torch.cuda.stream(GLOBAL_STREAM):
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
                global_q, self.n_local
            )

        o_list = []
        for st in range(
            0, input_length, self.exc_block_size
        ):  # Process the input tokens in blocks.
            ed = min(st + self.exc_block_size, input_length)

            # calculate attention results
            kv_st = max(kv_length + st - input_length - self.n_local, 0)
            kv_ed = kv_length + ed - input_length
            chunk_o = self._append(
                local_q[:, :, st:ed, :],
                self.local_k[:, :, kv_st:kv_ed, :],
                self.local_v[:, :, kv_st:kv_ed, :],
                global_q[:, :, st:ed, :],
            )
            o_list.append(chunk_o)

            # offload context memory
            with torch.cuda.stream(GLOBAL_STREAM):
                self._append_global()

            if self.async_global_stream:
                torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        self.length += input_length

        # restrict the length of local KV-cache to self.n_local
        if self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local :, :]
            self.local_v = self.local_v[:, :, -self.n_local :, :]

        # update global remainder
        assert self._global_remainder_ed == self.global_remainder[0].size(-2)
        assert (
            not self.init_exc or self._global_remainder_st == self._global_remainder_ed
        ), (
            f"self.init_exc: {self.init_exc}, global_remainder_st: {self._global_remainder_st}, global_remainder_ed: {self._global_remainder_ed}"
        )
        with torch.cuda.stream(GLOBAL_STREAM):
            self.global_remainder = (
                self.global_remainder[0][:, :, self._global_remainder_st :, :],
                self.global_remainder[1][:, :, self._global_remainder_st :, :],
            )

        ret = torch.cat(o_list, dim=-2)

        return ret

    def size(self, *args, **kwargs):
        return self.length

    def calculate_cpu_memory(self):
        memory = 0
        for u in range(self.num_units):
            for block in self.global_blocks[u]:
                memory += block.calculate_cpu_memory()
        return memory
