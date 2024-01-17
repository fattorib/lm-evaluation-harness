""" PyTorch StableLM Epoch model. """
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from .kv_cache import KVCache

from dataclasses import dataclass

@dataclass
class StableLMConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    norm_eps: float
    rope_pct: float
    rope_theta: int


STABLE_125M = StableLMConfig(
    vocab_size=50304,
    hidden_size=768,
    intermediate_size=2048,
    max_position_embeddings=1024,
    num_attention_heads=12,
    num_key_value_heads=12,
    num_hidden_layers=12,
    norm_eps=1e-05,
    rope_pct=0.25,
    rope_theta=10000,
)


class EmbeddingNoInit(nn.Embedding):
    def reset_parameters(self):
        pass


class LayerNormNoInit(nn.LayerNorm):
    def reset_parameters(self):
        pass


class LinearNoInit(nn.Linear):
    def reset_parameters(self):
        pass


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: int = 10_000,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

        # Don't do einsum, it converts fp32 to fp16 under AMP
        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        # x: [batch_size, num_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=torch.get_default_dtype()
            )
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLP(nn.Module):
    def __init__(self, config: StableLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = LinearNoInit(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = LinearNoInit(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = LinearNoInit(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Attention(nn.Module):
    def __init__(self, config: StableLMConfig, use_cache: bool = False):
        super().__init__()
        self.config = config
        self.use_cache = use_cache
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = LinearNoInit(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = LinearNoInit(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = LinearNoInit(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.o_proj = LinearNoInit(self.hidden_size, self.hidden_size, bias=False)

        self._init_rope()

    def _init_rope(self):
        self.rotary_ndims = int(self.head_dim * self.config.rope_pct)
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        layer_past: KVCache = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        has_layer_past = layer_past is not None and layer_past.current_kv_size > 0

        # TODO: Leaving a bit of performance on table (1-2%) by not fusing QKV matmul
        query_states = rearrange(
            self.q_proj(hidden_states),
            "b s (nh hd) -> b nh s hd",
            nh=self.num_heads,
            hd=self.head_dim,
        )
        key_states = rearrange(
            self.k_proj(hidden_states),
            "b s (nh hd) -> b nh s hd",
            nh=self.num_heads,
            hd=self.head_dim,
        )
        value_states = rearrange(
            self.v_proj(hidden_states),
            "b s (nh hd) -> b nh s hd",
            nh=self.num_heads,
            hd=self.head_dim,
        )

        query_rot = query_states[..., : self.rotary_ndims]
        query_pass = query_states[..., self.rotary_ndims :]
        key_rot = key_states[..., : self.rotary_ndims]
        key_pass = key_states[..., self.rotary_ndims :]

        kv_seq_len = key_states.shape[-2]
        if has_layer_past:
            kv_seq_len += layer_past.current_kv_size

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, position_ids
        )

        # [batch_size, num_heads, seq_len, head_dim]
        query_states = torch.cat((query_states, query_pass), dim=-1)
        key_states = torch.cat((key_states, key_pass), dim=-1)

        if self.use_cache:
            if has_layer_past:  # kv length is > 0
                # assume bs = 1
                layer_past.update_kv(key_states, value_states)
                key_states, value_states = layer_past.get_valid()

            else:
                layer_past.update_kv(key_states, value_states)
            kv_cache = layer_past

        else:
            kv_cache = None

        if not has_layer_past:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            ):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states, is_causal=True
                )

        else:
            import math

            # TODO: SDPA doesn't work with naive KV Cache
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Final linear projection
        attn_output = self.o_proj(attn_output)

        return attn_output, kv_cache


class DecoderLayer(nn.Module):
    def __init__(self, config: StableLMConfig, use_cache: bool = False):
        super().__init__()
        self.use_cache = use_cache
        self.self_attn = Attention(config, use_cache)
        self.mlp = MLP(config)
        self.input_layernorm = LayerNormNoInit(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = LayerNormNoInit(
            config.hidden_size, eps=config.norm_eps
        )

        self.dropout_1 = nn.Dropout(p=0.1)

        self.dropout_2 = nn.Dropout(p=0.1)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past=None,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, kv_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
        )
        hidden_states = self.dropout_1(hidden_states)

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout_2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, kv_cache


class StableLMEpochModel(nn.Module):
    def __init__(self, config: StableLMConfig, use_cache=False):
        super().__init__()
        self.config = config
        self.use_cache = use_cache
        self.embed_tokens = EmbeddingNoInit(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, use_cache) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LayerNormNoInit(config.hidden_size, eps=config.norm_eps)

        self.lm_head = LinearNoInit(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, LinearNoInit):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, EmbeddingNoInit):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNormNoInit):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: List[KVCache] = None,
    ):
        # Retrieve input_ids and inputs_embeds

        batch_size, seq_length = input_ids.shape

        seq_length_with_past = seq_length

        past_key_values_length = 0

        if self.use_cache:
            if layer_past[0].current_kv_size > 0:
                past_key_values_length = layer_past[0].current_kv_size
                seq_length_with_past += past_key_values_length

        device = input_ids.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if layer_past is None:
            layer_past = [None] * len(self.layers)

        kv_cache_list = []

        inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = None

        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, kv_cache = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_past=layer_past[idx],
            )

            kv_cache_list.append(kv_cache)

        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)

        if self.use_cache:
            return logits, kv_cache_list

        else:
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()

                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                return loss
            return logits
