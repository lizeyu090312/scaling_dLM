"""ELF transformer model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modules.layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder,
    DEFAULT_KERNEL_INIT, DEFAULT_BIAS_INIT, NORMAL_INIT_002, ZERO_INIT,
    _make_linear,
)


class ELFBlock(nn.Module):
    """ELF Transformer block."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x: torch.Tensor, rope_fn: Optional[nn.Module] = None,
                attention_mask: Optional[torch.Tensor] = None,
                deterministic: bool = True) -> torch.Tensor:
        x_normed = self.norm1(x)
        attn_out = self.attn(x_normed, rope_fn, attention_mask=attention_mask,
                             deterministic=deterministic)
        x = x + attn_out

        x_normed = self.norm2(x)
        mlp_out = self.mlp(x_normed, deterministic=deterministic)
        x = x + mlp_out
        return x


class REPAProjector(nn.Module):
    """MLP projector used only for auxiliary REPA alignment."""

    def __init__(self, hidden_size: int, projector_dim: int, target_dim: int, num_layers: int = 3):
        super().__init__()
        if num_layers < 1:
            raise ValueError("REPA projector num_layers must be at least 1")
        dims = [hidden_size]
        if num_layers > 1:
            dims.extend([projector_dim] * (num_layers - 1))
        dims.append(target_dim)

        layers = []
        for idx in range(num_layers):
            layers.append(_make_linear(dims[idx], dims[idx + 1], bias=True))
            if idx < num_layers - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class REPATransformerAdapter(nn.Module):
    """Masked transformer adapter used only for auxiliary REPA alignment."""

    def __init__(
        self,
        hidden_size: int,
        target_dim: int,
        num_heads: int,
        depth: int,
        max_length: int,
        mlp_ratio: float = 4.0,
        has_reg_token: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("REPA adapter depth must be at least 1")

        self.mask_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        NORMAL_INIT_002(self.mask_token)
        self.rope = TextRotaryEmbeddingFast(
            dim=hidden_size // num_heads,
            pt_seq_len=max_length,
            num_empty_token=int(has_reg_token),
        )
        self.blocks = nn.ModuleList([
            ELFBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = RMSNorm(hidden_size)
        self.output = _make_linear(hidden_size, target_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        input_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_mask.shape != x.shape[:2]:
            raise ValueError(
                f"REPA input mask must have shape {tuple(x.shape[:2])}, "
                f"got {tuple(input_mask.shape)}"
            )
        batch, length = x.shape[:2]
        valid_attention_shapes = {(batch, length), (batch, length, length)}
        if tuple(attention_mask.shape) not in valid_attention_shapes:
            raise ValueError(
                f"REPA attention mask must have shape {(batch, length)} or "
                f"{(batch, length, length)}, "
                f"got {tuple(attention_mask.shape)}"
            )

        x = torch.where(
            input_mask.to(torch.bool).unsqueeze(-1),
            self.mask_token.to(x.dtype),
            x,
        )
        for block in self.blocks:
            x = block(x, rope_fn=self.rope, attention_mask=attention_mask)
        return self.output(self.norm(x))


class ELF(nn.Module):
    """Text ELF Transformer."""

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        vocab_size: int = 0,
        gradient_checkpointing: bool = False,
        repa_target_dim: Optional[int] = None,
        repa_projector_dim: int = 2048,
        repa_projector_layers: int = 3,
        repa_projector_type: str = "mlp",
        repa_adapter_layers: int = 1,
        reg_target_dim: Optional[int] = None,
        reg_projection_topology: str = "separate",
        olt_enabled: bool = False,
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.bottleneck_dim = bottleneck_dim
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.repa_target_dim = repa_target_dim
        self.repa_projector_layers = repa_projector_layers
        self.repa_projector_type = repa_projector_type
        self.repa_adapter_layers = repa_adapter_layers
        self.reg_target_dim = reg_target_dim
        self.reg_projection_topology = reg_projection_topology
        self.olt_enabled = bool(olt_enabled)

        if reg_projection_topology not in {"separate", "shared_text"}:
            raise ValueError(
                "reg_projection_topology must be 'separate' or 'shared_text'"
            )
        if repa_projector_type not in {"mlp", "transformer"}:
            raise ValueError("repa_projector_type must be 'mlp' or 'transformer'")
        if int(repa_adapter_layers) < 1:
            raise ValueError("repa_adapter_layers must be positive")

        if self.olt_enabled and reg_target_dim is not None:
            raise ValueError("OLT and REG cannot both be enabled")

        self.repa_projector = None
        if repa_target_dim is not None:
            if repa_projector_type == "mlp":
                self.repa_projector = REPAProjector(
                    hidden_size, repa_projector_dim, repa_target_dim,
                    num_layers=repa_projector_layers,
                )
            else:
                self.repa_projector = REPATransformerAdapter(
                    hidden_size=hidden_size,
                    target_dim=repa_target_dim,
                    num_heads=num_heads,
                    depth=repa_adapter_layers,
                    max_length=max_length,
                    mlp_ratio=mlp_ratio,
                    has_reg_token=reg_target_dim is not None,
                )

        self.reg_input_proj = None
        self.reg_self_cond_proj = None
        self.reg_output_proj = None
        if reg_target_dim is not None:
            if int(reg_target_dim) <= 0:
                raise ValueError("reg_target_dim must be positive")
            if reg_projection_topology == "separate":
                self.reg_input_proj = _make_linear(reg_target_dim, hidden_size, bias=True)
                self.reg_self_cond_proj = _make_linear(2 * hidden_size, hidden_size, bias=True)
            else:
                # Preserve the separate topology's RNG position so all common
                # parameters have identical initialization in a topology ablation.
                with torch.random.fork_rng(devices=[]):
                    self.reg_input_proj = _make_linear(
                        reg_target_dim, text_encoder_dim, bias=False,
                    )
                    self.reg_output_proj = _make_linear(
                        text_encoder_dim, reg_target_dim, bias=False,
                    )
                _make_linear(reg_target_dim, hidden_size, bias=True)
                _make_linear(2 * hidden_size, hidden_size, bias=True)

        self.olt_token = None
        if self.olt_enabled:
            self.olt_token = nn.Parameter(torch.empty(1, 1, hidden_size))
            NORMAL_INIT_002(self.olt_token)

        # Self-conditioning input projection (only used when input is [z, x_pred]).
        self.self_cond_proj = _make_linear(2 * text_encoder_dim, text_encoder_dim, bias=True)

        # Text bottleneck projection.
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        # Time / SC-CFG embedders + learned prefix tokens.
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        NORMAL_INIT_002(self.t_emb_tokens)

        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(torch.empty(1, num_self_cond_cfg_tokens, hidden_size))
            NORMAL_INIT_002(self.self_cond_cfg_tokens)

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))
            NORMAL_INIT_002(self.mode_tokens)

        head_dim = hidden_size // num_heads
        prefix_total = num_model_mode_tokens + num_time_tokens
        if num_self_cond_cfg_tokens > 0:
            prefix_total += num_self_cond_cfg_tokens
        if reg_target_dim is not None or self.olt_enabled:
            prefix_total += 1
        self.feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=max_length, num_empty_token=prefix_total,
        )

        self.blocks = nn.ModuleList()
        q1, q3 = depth // 4, depth // 4 * 3
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            self.blocks.append(ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if in_drop_range else 0.0,
                proj_drop=proj_drop if in_drop_range else 0.0,
            ))

        # Final flow-matching output head.
        separate_reg_dim = (
            reg_target_dim if reg_projection_topology == "separate" else None
        )
        self.final_layer = FinalLayer(
            hidden_size, patch_size=1, out_channels=text_encoder_dim,
            reg_out_channels=separate_reg_dim,
        )
        if reg_target_dim is not None and reg_projection_topology == "shared_text":
            _make_linear(
                hidden_size, reg_target_dim, bias=True,
                kernel_init=ZERO_INIT, bias_init=ZERO_INIT,
            )

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab.
        bn = text_encoder_dim
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, bn))
        self.proj_bias = nn.Parameter(torch.empty(bn))
        self.unembed_kernel = nn.Parameter(torch.empty(bn, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        DEFAULT_KERNEL_INIT(self.proj_kernel)
        DEFAULT_BIAS_INIT(self.proj_bias)
        DEFAULT_KERNEL_INIT(self.unembed_kernel)
        DEFAULT_BIAS_INIT(self.unembed_bias)

    def build_context(self, t: torch.Tensor,
                      self_cond_cfg_scale: Optional[torch.Tensor] = None) -> list:
        B = t.shape[0]
        prefix_tokens = []

        time_emb = self.t_embedder(t)  # (B, hidden)
        prefix_tokens.append(
            self.t_emb_tokens.expand(B, -1, -1) + time_emb.unsqueeze(1)
        )

        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            prefix_tokens.append(
                self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb.unsqueeze(1)
            )
        return prefix_tokens

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
        return_repa_hidden: bool = False,
        return_repa_hidden_only: bool = False,
        repa_depth: Optional[int] = None,
        reg_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Text `x` and optional heterogeneous REG state. `attention_mask` covers text only."""
        needs_repa_hidden = return_repa_hidden or return_repa_hidden_only
        if needs_repa_hidden:
            if repa_depth is None:
                raise ValueError("repa_depth must be provided when returning REPA hidden")
            if repa_depth < 1 or repa_depth > self.depth:
                raise ValueError(f"repa_depth must be in [1, {self.depth}], got {repa_depth}")

        B, text_length = x.shape[:2]
        has_reg = self.reg_target_dim is not None
        has_aux_token = has_reg or self.olt_enabled
        if has_reg != (reg_x is not None):
            raise ValueError("reg_x must be provided exactly when REG is enabled")

        # Self-conditioning: input is [z, x_pred] when 2x encoder dim
        with torch.amp.autocast('cuda', enabled=False):
            has_self_cond = x.shape[-1] == 2 * self.text_encoder_dim
            if has_self_cond:
                x = self.self_cond_proj(x.float())
            elif x.shape[-1] != self.text_encoder_dim:
                raise ValueError(
                    f"Text input dimension must be {self.text_encoder_dim} or {2 * self.text_encoder_dim}, "
                    f"got {x.shape[-1]}"
                )

            if has_reg:
                expected_reg_dim = self.reg_target_dim * (2 if has_self_cond else 1)
                if reg_x.shape[:2] != (B, 1) or reg_x.shape[-1] != expected_reg_dim:
                    raise ValueError(
                        f"REG input must have shape ({B}, 1, {expected_reg_dim}), got {tuple(reg_x.shape)}"
                    )
                if has_self_cond:
                    reg_current, reg_previous = reg_x.float().chunk(2, dim=-1)
                    reg_projected = torch.cat([
                        self.reg_input_proj(reg_current),
                        self.reg_input_proj(reg_previous),
                    ], dim=-1)
                    if self.reg_projection_topology == "separate":
                        reg_projected = self.reg_self_cond_proj(reg_projected)
                    else:
                        reg_projected = self.text_proj(
                            self.self_cond_proj(reg_projected)
                        )
                else:
                    reg_projected = self.reg_input_proj(reg_x.float())
                    if self.reg_projection_topology == "shared_text":
                        reg_projected = self.text_proj(reg_projected)

            x = self.text_proj(x.float())
            if has_reg:
                x = torch.cat([reg_projected, x], dim=1)
            elif self.olt_enabled:
                x = torch.cat([self.olt_token.expand(B, -1, -1), x], dim=1)
            context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)

        if has_aux_token and attention_mask is not None:
            if attention_mask.shape != (B, text_length):
                raise ValueError(
                    f"Text attention mask must have shape {(B, text_length)}, got {tuple(attention_mask.shape)}"
                )
            aux_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([aux_mask, attention_mask], dim=1)

        # Prepend learnable model-mode tokens (gated by decoder_step_active).
        # decoder_step_active may be None / Python bool / (B,) tensor — the last
        # form supports per-example branching at training time.
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1)
            if decoder_step_active is None:
                active_gate = 0.0
            elif isinstance(decoder_step_active, torch.Tensor) and decoder_step_active.dim() > 0:
                active_gate = decoder_step_active.to(mode_tokens.dtype).view(-1, 1, 1)
            else:
                active_gate = float(decoder_step_active)
            mode_tokens = mode_tokens * active_gate
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones((B, self.num_model_mode_tokens),
                                       dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        prefix_len = 0
        if context_prefix_tokens:
            prefix_tokens = torch.cat(context_prefix_tokens, dim=1)
            prefix_len = prefix_tokens.shape[1]
            x = torch.cat([prefix_tokens, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones((B, prefix_len),
                                         dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        use_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        repa_hidden = None
        for block_idx, block in enumerate(self.blocks, start=1):
            if use_checkpoint:
                def _block_forward(hidden: torch.Tensor, block: ELFBlock = block) -> torch.Tensor:
                    return block(hidden, rope_fn=self.feat_rope, attention_mask=attention_mask,
                                 deterministic=deterministic)

                x = checkpoint(_block_forward, x, use_reentrant=False)
            else:
                x = block(x, rope_fn=self.feat_rope, attention_mask=attention_mask,
                          deterministic=deterministic)
            if needs_repa_hidden and block_idx == repa_depth:
                repa_start = prefix_len + model_mode_offset + int(self.olt_enabled)
                repa_hidden = x[:, repa_start:]
                if return_repa_hidden_only:
                    return repa_hidden

        x = x[:, prefix_len + model_mode_offset:]

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab
        with torch.amp.autocast('cuda', enabled=False):
            decoder_logits = None
            text_hidden = x[:, 1:] if has_aux_token else x
            if decoder_step_active is not None:
                x_f32 = text_hidden.float()
                hidden = F.gelu(x_f32 @ self.proj_kernel + self.proj_bias, approximate="tanh")
                decoder_logits = hidden @ self.unembed_kernel + self.unembed_bias
            reg_output = None
            if has_reg:
                if self.reg_projection_topology == "separate":
                    output, reg_output = self.final_layer(x.float())
                else:
                    shared_output = self.final_layer(x.float())
                    output = shared_output[:, 1:]
                    reg_output = self.reg_output_proj(shared_output[:, :1])
            else:
                output = self.final_layer(text_hidden.float())
        if return_repa_hidden:
            if has_reg:
                return output, decoder_logits, reg_output, repa_hidden
            return output, decoder_logits, repa_hidden
        if has_reg:
            return output, decoder_logits, reg_output
        return output, decoder_logits


# Model factory functions
def ELF_B(**kwargs): return ELF(depth=12, hidden_size=768,  num_heads=12, **kwargs)
def ELF_M(**kwargs): return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)
def ELF_L(**kwargs): return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)

ELF_MODEL_HIDDEN_SIZES = {
    'ELF-B': 768, 'ELF-M': 1056, 'ELF-L': 1280,
}

ELF_models = {
    'ELF-B': ELF_B, 'ELF-M': ELF_M, 'ELF-L': ELF_L,
}
