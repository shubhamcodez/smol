"""A practical ~204M parameter decoder-only language model for a 6GB GPU.

This combines a straightforward nanoGPT-style PyTorch training interface with
selected modern decoder features:

* pre-normalized RMSNorm transformer blocks
* rotary position embeddings (RoPE)
* grouped-query attention (GQA)
* PyTorch scaled-dot-product/Flash Attention when available
* a gated SwiGLU feed-forward network
* tied token embedding and output weights
* memory-efficient KV caching for autoregressive generation
* residual-projection initialization scaled by model depth

The default configuration has 203,821,824 trainable parameters and a 4,096
token context window. It targets full-parameter mixed-precision training on a
6GB RTX 3070 with micro-batch size 1 and gradient checkpointing. A tokenizer,
dataset, training loop, and trained weights are still required.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


KVCache = tuple[torch.Tensor, torch.Tensor]


class CausalLMOutput(NamedTuple):
    logits: Optional[torch.Tensor]
    loss: Optional[torch.Tensor]
    past_key_values: Optional[tuple[KVCache, ...]]


@dataclass
class ModelConfig:
    # 50,304 keeps compatibility with the commonly padded GPT-2 vocabulary.
    # Replacing the tokenizer/vocabulary changes the parameter count.
    vocab_size: int = 50_304
    max_seq_len: int = 4_096

    # Default architecture: 203,821,824 parameters with tied embeddings.
    n_layer: int = 24
    n_embd: int = 768
    n_head: int = 12
    n_kv_head: int = 4
    intermediate_size: int = 2_304

    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("attention head size must be even for RoPE")
        if self.vocab_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("vocab_size and max_seq_len must be positive")


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulating the variance in fp32 is more stable for fp16/bfloat16.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.outer(positions.float(), self.inv_freq.float())
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # x: [batch, heads, sequence, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.kv_groups = config.n_head // config.n_kv_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(
            config.n_embd,
            config.n_head * self.head_dim,
            bias=config.bias,
        )
        self.k_proj = nn.Linear(
            config.n_embd,
            config.n_kv_head * self.head_dim,
            bias=config.bias,
        )
        self.v_proj = nn.Linear(
            config.n_embd,
            config.n_kv_head * self.head_dim,
            bias=config.bias,
        )
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        batch_size, sequence_len, _ = x.shape
        past_len = 0 if past_key_value is None else past_key_value[0].size(2)

        q = self.q_proj(x).view(
            batch_size, sequence_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, sequence_len, self.n_kv_head, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, sequence_len, self.n_kv_head, self.head_dim
        ).transpose(1, 2)

        positions = torch.arange(
            past_len,
            past_len + sequence_len,
            device=x.device,
        )
        cos, sin = self.rope.cos_sin(positions, q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present = (k, v) if use_cache else None

        attention_mask = None
        is_causal = past_len == 0
        if past_len > 0:
            query_positions = past_len + torch.arange(sequence_len, device=x.device)
            key_positions = torch.arange(k.size(2), device=x.device)
            attention_mask = key_positions[None, :] <= query_positions[:, None]
            attention_mask = attention_mask[None, None, :, :]

        dropout_p = self.dropout if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                enable_gqa=self.n_head != self.n_kv_head,
            )
        else:
            # Compatibility path for older PyTorch builds without native GQA.
            expanded_k = k.repeat_interleave(self.kv_groups, dim=1)
            expanded_v = v.repeat_interleave(self.kv_groups, dim=1)
            scores = torch.matmul(q, expanded_k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            if attention_mask is not None:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            else:
                causal_mask = torch.ones(
                    sequence_len,
                    expanded_k.size(2),
                    dtype=torch.bool,
                    device=x.device,
                ).tril()
                scores = scores.masked_fill(
                    ~causal_mask[None, None, :, :], float("-inf")
                )
            weights = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
            weights = F.dropout(weights, p=dropout_p, training=self.training)
            y = torch.matmul(weights, expanded_v)

        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_len, -1)
        return self.resid_dropout(self.o_proj(y)), present


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.up_proj = nn.Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        attention, present = self.attn(
            self.attn_norm(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attention
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


class TransformerLM(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.n_embd
        )
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(self.config) for _ in range(self.config.n_layer)
        )
        self.final_norm = RMSNorm(self.config.n_embd, self.config.norm_eps)
        self.lm_head = nn.Linear(
            self.config.n_embd,
            self.config.vocab_size,
            bias=False,
        )
        self.gradient_checkpointing = False

        self.apply(self._init_weights)
        self._scale_residual_projections()
        if self.config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=std)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[KVCache]] = None,
        use_cache: bool = False,
        last_token_only: bool = False,
        return_logits: bool = True,
        loss_chunk_size: Optional[int] = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        batch_size, sequence_len = input_ids.shape
        del batch_size  # The value is documented by the shape check above.
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(2)
        if past_len + sequence_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {past_len + sequence_len} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one KV cache per layer")
        if self.gradient_checkpointing and self.training and use_cache:
            raise ValueError("KV caching is incompatible with training-time checkpointing")

        x = self.embedding_dropout(self.token_embedding(input_ids))
        presents: list[KVCache] = []

        for layer_index, block in enumerate(self.blocks):
            layer_past = (
                None if past_key_values is None else past_key_values[layer_index]
            )
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda hidden, module=block: module(
                        hidden,
                        past_key_value=None,
                        use_cache=False,
                    )[0],
                    x,
                    use_reentrant=False,
                )
                present = None
            else:
                x, present = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
            if present is not None:
                presents.append(present)

        x = self.final_norm(x)
        if last_token_only and targets is None:
            x = x[:, -1:, :]

        loss = None
        logits: Optional[torch.Tensor] = None
        if targets is not None and loss_chunk_size is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if loss_chunk_size <= 0:
                raise ValueError("loss_chunk_size must be positive")

            # Checkpointing the vocabulary projection prevents an entire
            # [batch, 4096, vocab] logits tensor from staying live for backward.
            # This is important on a 6GB GPU with the 50,304-token vocabulary.
            chunk_losses = []
            for start in range(0, x.size(1), loss_chunk_size):
                end = min(start + loss_chunk_size, x.size(1))
                hidden_chunk = x[:, start:end, :]
                target_chunk = targets[:, start:end]

                def compute_chunk_loss(
                    hidden: torch.Tensor,
                    labels: torch.Tensor,
                ) -> torch.Tensor:
                    chunk_logits = self.lm_head(hidden)
                    return F.cross_entropy(
                        chunk_logits.reshape(-1, chunk_logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-1,
                        reduction="sum",
                    )

                if self.training and torch.is_grad_enabled():
                    chunk_loss = checkpoint(
                        compute_chunk_loss,
                        hidden_chunk,
                        target_chunk,
                        use_reentrant=False,
                    )
                else:
                    chunk_loss = compute_chunk_loss(hidden_chunk, target_chunk)
                chunk_losses.append(chunk_loss)

            valid_tokens = (targets != -1).sum().clamp_min(1)
            loss = torch.stack(chunk_losses).sum() / valid_tokens
            if return_logits:
                logits = self.lm_head(x)
        else:
            logits = self.lm_head(x)
            if targets is not None:
                if targets.shape != input_ids.shape:
                    raise ValueError("targets must have the same shape as input_ids")
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-1,
                )

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(presents) if use_cache else None,
        )

    def configure_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cuda",
    ) -> torch.optim.AdamW:
        decay = []
        no_decay = []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        fused_args = {"fused": True} if fused_available and device_type == "cuda" else {}
        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=betas,
            **fused_args,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if input_ids.size(1) + max_new_tokens > self.config.max_seq_len:
            raise ValueError("prompt plus generated tokens exceed max_seq_len")

        was_training = self.training
        self.eval()
        try:
            output = self(input_ids, use_cache=True, last_token_only=True)
            cache = output.past_key_values
            generated = input_ids

            for step in range(max_new_tokens):
                if output.logits is None:
                    raise RuntimeError("generation requires logits")
                logits = output.logits[:, -1, :] / temperature
                if top_k is not None:
                    threshold = torch.topk(
                        logits, min(top_k, logits.size(-1))
                    ).values[:, [-1]]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))

                probabilities = F.softmax(logits.float(), dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                if step + 1 < max_new_tokens:
                    output = self(
                        next_token,
                        past_key_values=cache,
                        use_cache=True,
                        last_token_only=True,
                    )
                    cache = output.past_key_values

            return generated
        finally:
            self.train(was_training)


def default_parameter_count() -> int:
    """Return the exact count analytically without allocating the model."""
    config = ModelConfig()
    head_dim = config.n_embd // config.n_head
    embedding = config.vocab_size * config.n_embd
    attention = (
        config.n_embd * config.n_head * head_dim
        + 2 * config.n_embd * config.n_kv_head * head_dim
        + config.n_embd * config.n_embd
    )
    mlp = 3 * config.n_embd * config.intermediate_size
    norms_per_layer = 2 * config.n_embd
    transformer = config.n_layer * (attention + mlp + norms_per_layer)
    final_norm = config.n_embd
    output = 0 if config.tie_embeddings else embedding
    return embedding + transformer + final_norm + output


# Backward-compatible names for code written against the earlier 1B draft.
Hybrid1BConfig = ModelConfig
Hybrid1B = TransformerLM


if __name__ == "__main__":
    print(f"Default parameter count: {default_parameter_count():,}")
