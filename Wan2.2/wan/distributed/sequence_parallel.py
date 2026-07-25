# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
import math
from torch.utils.checkpoint import checkpoint

from ..modules.model import sinusoidal_embedding_1d
from .ulysses import (distributed_attention,
                      distributed_attention_with_prepared_kv,
                      prepare_kv_for_distributed_attention)
from .util import gather_forward, get_rank, get_world_size


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # Context Parallel: shard the sequence before building timestep embeddings.
    # The original implementation constructed full-sequence e/e0 on every rank
    # and only then chunked them, which is prohibitively expensive at SR sizes.
    world_size = get_world_size()
    rank = get_rank()
    if world_size > 1:
        if t.dim() != 1:
            t = torch.chunk(t, world_size, dim=1)[rank]
        x = torch.chunk(x, world_size, dim=1)[rank]
    local_seq_len = x.size(1)

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), local_seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, local_seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    use_checkpointing = getattr(self, "use_activation_checkpointing", False)
    for block in self.blocks:
        if use_checkpointing and torch.is_grad_enabled():
            def block_forward(hidden_states, block_context, block=block):
                block_kwargs = {**kwargs, "context": block_context}
                return block(hidden_states, **block_kwargs)

            x = checkpoint(block_forward, x, context, use_reentrant=False)
        else:
            x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    s, n, d = x.shape[1], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        input_b, input_s = x.shape[:2]
        q = self.norm_q(self.q(x)).view(input_b, input_s, n, d)
        k = self.norm_k(self.k(x)).view(input_b, input_s, n, d)
        v = self.v(x).view(input_b, input_s, n, d)
        return q, k, v

    if getattr(self, "block_tiled_attn_enabled", False):
        # Prepare global head-sharded K/V once per block. Each tile then
        # exchanges only Q/output while applying its own rectified RoPE to K.
        x = self.block_tiled_self_attention(
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            hidden_states=x,
            distributed_kv_prepare_fn=prepare_kv_for_distributed_attention,
            distributed_prepared_attention_fn=(
                distributed_attention_with_prepared_kv),
            sequence_rank=get_rank(),
            sequence_world_size=get_world_size())
    else:
        q, k, v = qkv_fn(x)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        x = distributed_attention(
            half(q),
            half(k),
            half(v),
            seq_lens,
            window_size=self.window_size,
            softmax_scale=1 / math.sqrt(d),
        )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
