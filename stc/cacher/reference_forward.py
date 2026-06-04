"""Selective ViT-layer forwards used by STC-Cacher.

These forwards are bound to HuggingFace pre-LN ViT layers (``layer_norm1`` /
``self_attn`` / ``layer_norm2`` / ``mlp``) by ``stc.integrations.hf_vit``.
They never read the global STC config or default cache directly: instead the
``register`` helper attaches the active cache and config to each layer, and
the forwards read them from there.  This keeps the hot path free of
process-wide state and makes per-tower configuration possible.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from stc.cacher.state import STCCache, default_cache
from stc.config import CacheConfig, default_config
from stc.core.selectors import select_dynamic_token_indices
from stc.utils.distributed import is_main_process

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer-shape helpers (HF SigLIP / CLIP pre-LN layout)
# ---------------------------------------------------------------------------


def _attention_module(layer):
    if not hasattr(layer, "self_attn"):
        raise AttributeError("STC-Cacher expects a layer with self_attn")
    return layer.self_attn


def _num_heads(attn) -> int:
    if hasattr(attn, "num_heads"):
        return int(attn.num_heads)
    if hasattr(attn, "num_attention_heads"):
        return int(attn.num_attention_heads)
    raise AttributeError("Cannot infer attention head count")


def _dropout_p(layer) -> float:
    attn = _attention_module(layer)
    dropout = getattr(attn, "dropout", getattr(layer, "dropout", 0.0))
    if isinstance(dropout, torch.nn.Dropout):
        return float(dropout.p)
    return float(dropout)


def _reshape_heads(states: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, seq_len, embed_dim = states.shape
    head_dim = embed_dim // num_heads
    return states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


# ---------------------------------------------------------------------------
# Per-layer cache/config resolution
# ---------------------------------------------------------------------------


def _layer_cache(layer) -> STCCache:
    """Return the cache bound at registration, falling back to the default."""

    cache = getattr(layer, "_stc_cache", None)
    return cache if cache is not None else default_cache()


def _layer_cache_config(layer) -> CacheConfig:
    config = getattr(layer, "_stc_config", None)
    return config if config is not None else default_config().cache


# ---------------------------------------------------------------------------
# SDPA wrapper
# ---------------------------------------------------------------------------


def stc_sdpa_attention(
    layer,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run SDPA and project back through the layer attention output.

    The SigLIP/CLIP layer forwards are pre-LN and non-causal; the wrapper just
    forwards into ``F.scaled_dot_product_attention`` and re-applies the
    layer's ``out_proj``.
    """

    del output_attentions
    attn = _attention_module(layer)
    query_length = query_states.shape[-2]
    batch_size = query_states.shape[0]
    embed_dim = getattr(
        attn, "embed_dim", query_states.shape[1] * query_states.shape[-1]
    )

    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=_dropout_p(layer) if layer.training else 0.0,
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, query_length, embed_dim)
    attn_output = attn.out_proj(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# Reference-frame management
# ---------------------------------------------------------------------------


_REFERENCE_ATTRS = (
    "reference_frame_key",
    "reference_frame_value",
    "reference_frame_attn_out",
    "reference_frame_mlp_out",
)


def _has_reference(layer) -> bool:
    return all(hasattr(layer, name) for name in _REFERENCE_ATTRS)


def _should_refresh_reference(layer) -> bool:
    config = _layer_cache_config(layer)
    if config.strategy == "none":
        return True
    if not _has_reference(layer):
        return True
    cache = _layer_cache(layer)
    chunk_idx = getattr(cache, "chunk_idx", 1)
    return chunk_idx % config.cache_interval == 0


def _store_reference(layer, name: str, tensor: torch.Tensor) -> None:
    """Store a per-frame reference into a persistent in-place buffer.

    The original code did ``layer.<name> = tensor.detach().clone()`` which
    allocates a *new* tensor (new address) on every refresh.  A captured CUDA
    graph bakes in the buffer address it reads, so a fresh address would make
    the graph read stale/freed memory after a refresh.  Writing in place keeps
    the address stable; replay always sees the latest reference.  Behaviour is
    otherwise identical to the clone.
    """
    src = tensor.detach()
    buf = getattr(layer, name, None)
    if (
        buf is None
        or buf.shape != src.shape
        or buf.dtype != src.dtype
        or buf.device != src.device
    ):
        buf = torch.empty_like(src)
        setattr(layer, name, buf)
    buf.copy_(src)


def _full_forward_and_cache(
    layer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    output_attentions: bool,
) -> tuple:
    residual = hidden_states
    hidden_states_ln1 = layer.layer_norm1(hidden_states)
    attn = _attention_module(layer)

    num_heads = _num_heads(attn)

    query_states = attn.q_proj(hidden_states_ln1)
    key_states = attn.k_proj(hidden_states_ln1)
    value_states = attn.v_proj(hidden_states_ln1)

    _store_reference(layer, "reference_frame_key", key_states[-1])
    _store_reference(layer, "reference_frame_value", value_states[-1])

    query_heads = _reshape_heads(query_states, num_heads)
    key_heads = _reshape_heads(key_states, num_heads)
    value_heads = _reshape_heads(value_states, num_heads)
    attn_output, attn_weights = layer.stc_attention(
        query_states=query_heads,
        key_states=key_heads,
        value_states=value_heads,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
    )

    hidden_states = residual + attn_output
    residual = hidden_states
    hidden_states_ln2 = layer.layer_norm2(hidden_states)
    mlp_output = layer.mlp(hidden_states_ln2)
    hidden_states = residual + mlp_output

    _store_reference(layer, "reference_frame_attn_out", attn_output[-1])
    _store_reference(layer, "reference_frame_mlp_out", mlp_output[-1])

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)
    return outputs


def _filled_buffer(layer, name: str, reference: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return a persistent ``(batch, *reference.shape)`` buffer filled from a
    single-frame ``reference`` (shape ``[seq, dim]``).

    Replaces the per-frame ``reference.unsqueeze(0).expand(...).clone()`` so the
    selective path reuses one allocation instead of churning cudaMalloc/cudaFree
    on every layer of every frame.  The returned tensor is bit-identical to the
    clone it replaces; callers mutate it in place (scatter) afterwards.
    """
    want = (batch_size,) + tuple(reference.shape)
    buf = getattr(layer, name, None)
    if (
        buf is None
        or tuple(buf.shape) != want
        or buf.dtype != reference.dtype
        or buf.device != reference.device
    ):
        buf = reference.new_empty(want)
        setattr(layer, name, buf)
    buf.copy_(reference.unsqueeze(0).expand(want))
    return buf


def _selective_key_forward(
    layer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    output_attentions: bool,
) -> tuple:
    cache = _layer_cache(layer)
    config = _layer_cache_config(layer)
    update_ratio = getattr(cache, "update_token_ratio", config.update_token_ratio)

    residual = hidden_states
    hidden_states_ln1 = layer.layer_norm1(hidden_states)
    attn = _attention_module(layer)

    batch_size, seq_len, embed_dim = hidden_states_ln1.shape
    num_heads = _num_heads(attn)
    head_dim = embed_dim // num_heads

    # k_proj 在“选择”和“注意力”里都要用，算一次复用（原实现把同一个 k_proj 算了两遍）。
    key_states_full = attn.k_proj(hidden_states_ln1)

    # share_selection：只让首层（selector）算一次“重算哪些 token”，其余层复用，
    # 省掉每层的 cosine + topk。属于近似（各层共享同一组 token），精度需自行验证。
    share = getattr(config, "share_selection", False)
    shared = getattr(cache, "shared_update_indices", None)
    if share and not getattr(layer, "_stc_is_selector", True) and shared is not None:
        update_indices = shared
    else:
        update_indices = select_dynamic_token_indices(
            key_states_full,
            layer.reference_frame_key,
            update_ratio=update_ratio,
            metric=config.selector_metric,
        )
        if share:
            cache.shared_update_indices = update_indices

    expanded_indices = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
    tokens_to_update = hidden_states_ln1.gather(1, expanded_indices)
    if is_main_process():
        logger.debug(
            "STC-Cacher selected %s/%s tokens for recomputation",
            tokens_to_update.shape[1],
            seq_len,
        )

    query_selected = attn.q_proj(tokens_to_update)
    value_selected = attn.v_proj(tokens_to_update)
    query_selected = _reshape_heads(query_selected, num_heads)
    value_selected = _reshape_heads(value_selected, num_heads)

    # 复用持久 buffer 代替每帧 expand().clone()，避免反复 cudaMalloc/Free。
    value_states = _filled_buffer(
        layer, "_stc_buf_value", layer.reference_frame_value, batch_size
    )
    value_states = value_states.view(batch_size, seq_len, num_heads, head_dim)
    value_states = value_states.transpose(1, 2)

    scatter_indices = update_indices.unsqueeze(1).unsqueeze(-1).expand(
        batch_size, num_heads, update_indices.shape[1], head_dim
    )
    value_states.scatter_(2, scatter_indices, value_selected)

    key_states = _reshape_heads(key_states_full, num_heads)
    attn_output_selected, _ = layer.stc_attention(
        query_states=query_selected,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
    )

    attn_output = _filled_buffer(
        layer, "_stc_buf_attn", layer.reference_frame_attn_out, batch_size
    )
    attn_output.scatter_(1, expanded_indices, attn_output_selected)

    hidden_states = residual + attn_output
    residual = hidden_states
    hidden_states_ln2 = layer.layer_norm2(hidden_states)

    mlp_output = _filled_buffer(
        layer, "_stc_buf_mlp", layer.reference_frame_mlp_out, batch_size
    )
    ln2_tokens = hidden_states_ln2.gather(1, expanded_indices)
    mlp_selected = layer.mlp(ln2_tokens)
    mlp_output.scatter_(1, expanded_indices, mlp_selected)
    hidden_states = residual + mlp_output

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (None,)
    return outputs


# ---------------------------------------------------------------------------
# Public layer forwards
# ---------------------------------------------------------------------------


def forward_with_selective_key_recompute(
    layer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
):
    """Forward for SigLIP-like encoder layers patched by STC-Cacher."""

    if _should_refresh_reference(layer):
        return _full_forward_and_cache(
            layer, hidden_states, attention_mask, output_attentions
        )
    return _selective_key_forward(
        layer, hidden_states, attention_mask, output_attentions
    )


def forward_with_selective_key_recompute_clip(
    layer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
):
    """Forward for CLIP-like layers.

    ``causal_attention_mask`` is accepted to match HuggingFace CLIP's layer
    signature.  Vision CLIP layers are non-causal so STC ignores it.
    """

    del causal_attention_mask
    return forward_with_selective_key_recompute(
        layer,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
    )
