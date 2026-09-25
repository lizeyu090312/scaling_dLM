import torch
import numpy as np

from modules.text_encoder import feature_standardize
from utils.teacher_utils import validate_input_ids_compatibility


@torch.no_grad()
def encode_text(
    input_ids,
    attention_mask,
    encoder,
    latent_mean,
    latent_std,
    use_bf16=True,
    return_raw=False,
):
    """Encoder pass from text to latent with normalization."""
    validate_input_ids_compatibility(
        input_ids,
        getattr(encoder, "incompatible_dataset_token_ids", ()),
        getattr(encoder, "token_consumer_name", "clean encoder"),
    )
    applies_own_normalization = bool(
        getattr(getattr(encoder, "config", None), "applies_own_normalization", False)
    )
    if applies_own_normalization and (latent_mean != 0.0 or latent_std != 1.0):
        raise ValueError(f"Qwen3 per-vector standardization requires latent_mean=0.0 and latent_std=1.0; got latent_mean={latent_mean!r}, latent_std={latent_std!r}")
    autocast_enabled = bool(use_bf16) and input_ids.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
        if return_raw and applies_own_normalization:
            config = encoder.config
            raw = encoder.encode_views(
                input_ids,
                attention_mask,
                {"default": (config.d_model, config.encoder_layer)},
                deterministic=True,
                normalize=False,
            )["default"]
            latents = feature_standardize(raw)
        else:
            latents = encoder(
                input_ids=input_ids, attention_mask=attention_mask, deterministic=True,
            )
            raw = latents if return_raw else None
    if not applies_own_normalization:
        latents = (latents - latent_mean) / latent_std
    return (latents, raw) if return_raw else latents


def apply_label_drop_to_encoder_attention_mask(encoder_attention_mask, cond_seq_mask, label_drop_mask):
    drop = label_drop_mask.to(dtype=encoder_attention_mask.dtype).reshape(-1, 1, 1)
    block_mask = (1 - cond_seq_mask).unsqueeze(-1) * cond_seq_mask.unsqueeze(1)
    return encoder_attention_mask * (1 - drop * block_mask)


def build_self_attn_cond_masks(is_cond, is_valid, xp=np):
    """Build self-attention conditioning masks from cond/valid token flags."""
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :]) |
        (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(xp.float32)
    attention_mask = is_valid.astype(xp.float32)
    cond_seq_mask = is_cond.astype(xp.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask
