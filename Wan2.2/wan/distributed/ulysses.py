# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist

from ..modules.attention import flash_attention
from .util import all_to_all


def prepare_kv_for_distributed_attention(k, v):
    """Move sequence-sharded K/V into Ulysses head-sharded layout once."""
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")
    k = all_to_all(k, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, scatter_dim=2, gather_dim=1)
    return k, v


def distributed_attention_with_prepared_kv(
        q,
        k,
        v,
        seq_lens,
        window_size=(-1, -1),
        softmax_scale=None,
):
    """Run Ulysses attention with K/V already in head-sharded layout."""
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")

    q = all_to_all(q, scatter_dim=2, gather_dim=1)
    x = flash_attention(
        q,
        k,
        v,
        k_lens=seq_lens,
        softmax_scale=softmax_scale,
        window_size=window_size,
    )
    return all_to_all(x, scatter_dim=1, gather_dim=2)


def distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=(-1, -1),
        softmax_scale=None,
):
    """
    Performs distributed attention based on DeepSpeed Ulysses attention mechanism.
    please refer to https://arxiv.org/pdf/2309.14509

    Args:
        q:           [B, Lq // p, Nq, C1].
        k:           [B, Lk // p, Nk, C1].
        v:           [B, Lk // p, Nk, C2]. Nq must be divisible by Nk.
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
    """
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")

    k, v = prepare_kv_for_distributed_attention(k, v)
    return distributed_attention_with_prepared_kv(
        q,
        k,
        v,
        seq_lens,
        softmax_scale=softmax_scale,
        window_size=window_size,
    )
