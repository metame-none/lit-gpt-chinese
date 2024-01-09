"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self

from lit_gpt.config import Config
import torch.nn.functional as F


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        if config.lm_head_type == "linear":
            self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        elif config.lm_head_type == "norm_head":
            self.lm_head = NormHead(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd, config.pad_token_id),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=get_norm_func(config),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value
        if self.config.position_emb_type == "rope":
            if not hasattr(self, "cos"):
                # first call
                cos, sin = self.rope_cache()
                self.register_buffer("cos", cos, persistent=False)
                self.register_buffer("sin", sin, persistent=False)
            # overrides
            elif self.cos.device.type == "meta":
                self.cos, self.sin = self.rope_cache()
            elif value != self.cos.size(0):
                self.cos, self.sin = self.rope_cache(device=self.cos.device)
            # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
            # if the kv cache is expected
        elif self.config.position_emb_type == "alibi":
            if not hasattr(self, "future_mask"):
                self.register_buffer(
                    "future_mask",
                    build_alibi_mask(self.config.n_head, value),
                    persistent=False,
                )
            elif value != self.future_mask.size(0):
                self.future_mask = build_alibi_mask(self.config.n_head, value).to(self.future_mask.device)

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.max_seq_length = self.config.block_size

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        cos, sin = None, None
        if input_pos is not None:  # use the kv cache
            max_pos = torch.max(input_pos)+1
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            if self.config.position_emb_type == "rope":
                if self.cos.device != idx.device:
                    self.cos = self.cos.to(idx.device)
                    self.sin = self.sin.to(idx.device)
                cos = self.cos.index_select(0, input_pos)
                sin = self.sin.index_select(0, input_pos)
                mask = self.mask_cache.index_select(2, input_pos)
            elif self.config.position_emb_type == "alibi":
                mask = self.mask_cache[:, :, :max_pos, :]
        else:
            if self.config.position_emb_type == "rope":
                cos = self.cos[:T]
                sin = self.sin[:T]
            mask = None
            max_pos = T
        
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if self.config.position_emb_type == "alibi":
            # TODO(metame): training may be different
            if self.future_mask.device != x.device:
                self.future_mask = self.future_mask.to(x.device)
            alibi_mask = self.future_mask[:self.config.n_head, :max_pos, :max_pos]
            if mask is not None:
                mask = mask[..., :max_pos]
                mask = self.update_attention_mask(x, mask, alibi_mask)
            else:
                mask = alibi_mask

        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)

        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    def update_attention_mask(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, 
                          alibi_mask: torch.Tensor) -> torch.Tensor:
        if len(attention_mask.shape) == 2:
                expanded_mask = attention_mask.to(alibi_mask.dtype)
                expanded_mask = torch.tril(
                    torch.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
                ) * torch.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
        else:
            expanded_mask = attention_mask

        if len(expanded_mask.shape) == 3:
           expanded_mask = expanded_mask.unsqueeze(1)
        bsz = inputs_embeds.size(0)
        src_len, tgt_len = alibi_mask.size()[-2:]
        expanded_mask = (
            expanded_mask.expand(bsz, 1, src_len, tgt_len)
            .to(alibi_mask.dtype)
        )
        inverted_mask = 1.0 - expanded_mask
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(alibi_mask.dtype).min
        )
        attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
        return attention_mask

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.get_default_dtype()
        if self.config.rope_dtype is not None:
            dtype = self.config.rope_dtype
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)

        if "baichuan2" in self.config.name:
            device = "cpu"
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            dtype=dtype,
            # NOTE: baichuan2 init with device=None, convert to device=input.device during computing
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            out_dtype=self.config.rope_out_dtype,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None and self.config.position_emb_type == "rope":
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = get_norm_func(config)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = None if config.shared_attention_norm else get_norm_func(config)
        self.mlp = config.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_1 = self.norm_1(x)
        h = self.attn(n_1, cos, sin, mask, input_pos)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(n_2) + h + x
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            x = h + x
            x = self.mlp(self.norm_2(x)) + x
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        attn_bias = config.add_qkv_bias
        self.attn = nn.Linear(config.n_embd, shape, bias=attn_bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        if self.config.position_emb_type == "rope":
            q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin, self.config.rope_type)
            k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin, self.config.rope_type)
            q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        if self.config.position_emb_type == "alibi" and "baichuan2-13b" in self.config.name:
            if input_pos is not None:
                max_pos = torch.max(input_pos) + 1
                k = k[:, :, :max_pos, :]
                v = v[:, :, :max_pos, :]
            y = self.attention_with_alibi(T, q, k, v, mask)
        else:
            y = self.scaled_dot_product_attention(q, k, v, mask)
        

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)
        if not self.config.add_attention_scale:
            scale = None
        is_causal = mask is None
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=is_causal
        )
        return y.transpose(1, 2)

    def attention_with_alibi(self, seq_len: int,
        query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.config.head_size)

        if attention_mask is not None:
            if seq_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        return attn_output

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None and self.config.position_emb_type == "rope":
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            if self.config.position_emb_type == "rope":
                last_dim = rope_cache_length + self.config.head_size - self.config.rope_n_elem
            elif self.config.position_emb_type == "alibi":
                last_dim = self.config.head_size
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                last_dim,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class ChatGLM2MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.add_bias = config.bias
        self.dense_h_to_4h = nn.Linear(config.n_embd, config.intermediate_size * 2,
                                       bias=self.add_bias)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.n_embd,
                                       bias=self.add_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_h_to_4h(x)
        x = self.activation_func(x)
        out = self.dense_4h_to_h(x)
        return out


class NormHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.first_flag = True

    def forward(self, hidden_states):
        if self.training:
            norm_weight = nn.functional.normalize(self.weight)
            self.first_flag = True
        elif self.first_flag:
            self.first_flag = False
            self.weight.data = nn.functional.normalize(self.weight)
            norm_weight = self.weight
        else:
            norm_weight = self.weight
        return nn.functional.linear(hidden_states, norm_weight)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: Optional[torch.device] = None, base: int = 10000, condense_ratio: int = 1,
    out_dtype: Optional[torch.dtype] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    if out_dtype is None:
        out_dtype = dtype
    if isinstance(out_dtype, str):
        out_dtype = getattr(torch, out_dtype)
    # for chatglm n_elem is 64
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    # NOTE: chatglm convert idx_theta with float()
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # this is to mimic the behaviour of complex32, else we will get different results
    if out_dtype in (torch.float16, torch.bfloat16, torch.int8):
        return (cos.bfloat16(), sin.bfloat16()) if dtype == torch.bfloat16 else (cos.half(), sin.half())
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_type: str = "default") -> torch.Tensor:
    if cos.device != x.device:
        cos = cos.to(x.device)
        sin = sin.to(x.device)
    if rope_type == "default":
        head_size = x.size(-1)
        x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
        x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
        rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
        roped = (x * cos) + (rotated * sin)
    elif rope_type == 'baichuan':
        head_size = x.size(-1)
        x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
        x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
        rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
        # baichuan2 convert to float() to apply rope
        roped = (x.float() * cos) + (rotated.float() * sin)
    elif rope_type == "chatglm":
        # for chatglm it add: @torch.jit.script to apply_rope, make the result different
        B, nh, T, head_size = x.shape
        x = x.reshape(B, nh, T, head_size // 2, 2)  # (B, nh, T, hs/2, 2)
        x1, x2 = x[..., 0], x[..., 1]
        x = torch.cat((x1, x2), dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
        roped = (x * cos) + (rotated * sin)
        roped = roped.reshape(B, nh, T, 2, head_size//2)
        roped = roped.permute(0, 1, 2, 4, 3)
        roped = roped.reshape(B, nh, T, head_size)
    return roped.type_as(x)


def get_norm_func(config: Config) -> nn.Module:
    ln_func = config.norm_class(config.n_embd, eps=config.norm_eps)
    return ln_func


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    
def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
    _future_mask = _future_mask.unsqueeze(0) + alibi
    new_future_mask = _future_mask.to(tensor)
    return new_future_mask[: tensor.shape[0] * attn_heads, :maxpos, :maxpos]


def build_alibi_mask(n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head))
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    slopes = slopes.to(position_point.device)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask