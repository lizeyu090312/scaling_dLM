"""Small helpers for optional REPA auxiliary alignment."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.text_encoder import encoder_family_from_model_type


REPA_ADAPTER_ATTENTION_MODES = {"full", "causal", "directional"}
_TEACHER_FAMILY_UNSET = object()


def validate_repa_shift(config, teacher_family=_TEACHER_FAMILY_UNSET) -> int:
    shift = getattr(config, "repa_shift", 0)
    if isinstance(shift, bool) or not isinstance(shift, int) or shift < 0:
        raise ValueError("repa_shift must be a non-negative integer")
    if shift == 0:
        return shift
    if not bool(getattr(config, "repa_enabled", False)):
        raise ValueError("repa_shift must be 0 when REPA is disabled")
    if float(getattr(config, "repa_prompt_loss_weight", 0.0)) != 0.0:
        raise ValueError("positive repa_shift requires repa_prompt_loss_weight=0")
    if float(getattr(config, "label_drop_prob", 0.0)) != 0.0:
        raise ValueError("positive repa_shift requires label_drop_prob=0")
    if teacher_family is not _TEACHER_FAMILY_UNSET and teacher_family != "qwen3":
        raise ValueError("positive repa_shift requires a Qwen3 REPA teacher")
    return shift


def _canonical_layer(layer) -> int:
    return -1 if layer is None else int(layer)


def _selected_dim(raw_dim: int, dim, field_name: str = "repa_teacher_dim") -> int:
    if dim is None:
        return int(raw_dim)
    dim = int(dim)
    if dim <= 0 or dim > int(raw_dim):
        raise ValueError(f"{field_name} must be in [1, {raw_dim}], got {dim}")
    return dim


def truncate_qwen3_model_for_layers(model, layers) -> Optional[int]:
    """Stop a dense Qwen3 model at the deepest required explicit hidden state."""
    layers = [_canonical_layer(layer) for layer in layers]
    if not layers:
        return None

    config = getattr(model, "config", None)
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        raise ValueError("Qwen3 model config has no valid num_hidden_layers")
    invalid = [layer for layer in layers if layer != -1 and not 0 <= layer <= num_layers]
    if invalid:
        raise ValueError(
            f"Qwen3 layers must be -1 or in [0, {num_layers}], got {invalid}"
        )
    if -1 in layers:
        return None

    deepest = max(layers)
    if deepest == num_layers:
        return None
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError(
            "Intermediate-layer early exit requires a dense Qwen3 model, "
            f"got {getattr(config, 'model_type', None)!r}"
        )
    missing = [name for name in ("layers", "norm") if not hasattr(model, name)]
    if missing:
        raise ValueError(
            "Intermediate-layer early exit requires a dense Qwen3 model; "
            f"missing {', '.join(missing)}"
        )

    existing = getattr(model, "_qwen3_truncated_layer", None)
    if existing is not None:
        if int(existing) != deepest:
            raise ValueError(
                f"Qwen3 model is already truncated at layer {existing}, "
                f"cannot truncate to {deepest}"
            )
        return deepest
    if len(model.layers) != num_layers:
        raise ValueError("Qwen3 model layers do not match config.num_hidden_layers")

    model.layers = nn.ModuleList(list(model.layers[:deepest]))
    model.norm = nn.Identity()
    model._qwen3_truncated_layer = deepest
    return deepest


def same_repa_teacher_model(config, encoder_config) -> bool:
    return getattr(config, "repa_teacher_model_name", None) == getattr(encoder_config, "model_name", None)


def same_repa_teacher(config, encoder_config=None) -> bool:
    if encoder_config is None:
        return getattr(config, "repa_teacher_model_name", None) == getattr(config, "encoder_model_name", None)
    if not same_repa_teacher_model(config, encoder_config):
        return False
    if getattr(config, "repa_teacher_input_format", "raw") != "raw":
        return False
    if getattr(encoder_config, "family", None) == "t5" and getattr(config, "repa_teacher_layer", None) is not None:
        return False
    teacher_layer = _canonical_layer(getattr(config, "repa_teacher_layer", None))
    encoder_layer = _canonical_layer(getattr(encoder_config, "encoder_layer", None))
    teacher_dim = _selected_dim(
        int(getattr(encoder_config, "raw_d_model", getattr(encoder_config, "d_model"))),
        getattr(config, "repa_teacher_dim", None),
    )
    return teacher_layer == encoder_layer and teacher_dim == int(encoder_config.d_model)


def repa_target_dim(config, encoder_config) -> int:
    if same_repa_teacher(config, encoder_config):
        return int(encoder_config.d_model)
    if same_repa_teacher_model(config, encoder_config):
        return _selected_dim(
            int(getattr(encoder_config, "raw_d_model", getattr(encoder_config, "d_model"))),
            getattr(config, "repa_teacher_dim", None),
        )

    model_name = getattr(config, "repa_teacher_model_name")
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_name)
    raw_dim = int(getattr(hf_config, "hidden_size", getattr(hf_config, "d_model", 0)) or 0)
    if raw_dim <= 0:
        raise ValueError(f"Could not determine d_model for REPA teacher {model_name!r}")
    return _selected_dim(raw_dim, getattr(config, "repa_teacher_dim", None))


def reg_target_dim(config, encoder_config) -> int:
    if same_repa_teacher_model(config, encoder_config):
        raw_dim = int(getattr(encoder_config, "raw_d_model", getattr(encoder_config, "d_model")))
    else:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(getattr(config, "repa_teacher_model_name"))
        raw_dim = int(getattr(hf_config, "hidden_size", getattr(hf_config, "d_model", 0)) or 0)
        if raw_dim <= 0:
            raise ValueError(
                f"Could not determine hidden size for REG teacher {config.repa_teacher_model_name!r}"
            )
    return _selected_dim(
        raw_dim, getattr(config, "reg_teacher_dim", None), field_name="reg_teacher_dim"
    )


def validate_external_repa_teacher(repa_teacher, repa_teacher_config, encoder_config) -> None:
    teacher_family = getattr(repa_teacher_config, "family", None)
    if teacher_family is None:
        teacher_family = encoder_family_from_model_type(getattr(repa_teacher.model.config, "model_type", ""))
    encoder_family = getattr(encoder_config, "family", None)
    if encoder_family is None:
        encoder_family = encoder_family_from_model_type(getattr(encoder_config, "model_type", ""))
    if teacher_family != encoder_family or teacher_family not in {"t5", "qwen3"}:
        raise ValueError(
            "REPA teacher family and clean encoder family must both be T5 or both be Qwen3 "
            f"({teacher_family!r} vs {encoder_family!r})"
        )


def validate_repa_token_ids(input_ids: torch.Tensor, repa_teacher) -> None:
    embeddings = repa_teacher.model.get_input_embeddings()
    embedding_rows = int(getattr(embeddings, "num_embeddings", 0) or 0)
    if embedding_rows <= 0:
        raise ValueError("REPA teacher has no valid input embedding table")
    if input_ids.numel() and int(input_ids.min().item()) < 0:
        raise ValueError("Batch input_ids contain negative ids")
    if input_ids.numel() and int(input_ids.max().item()) >= embedding_rows:
        raise ValueError(
            f"Batch input_ids contain ids outside REPA teacher embedding rows={embedding_rows}"
        )


def repa_is_active(end_fraction: float, total_optimizer_steps: int, optimizer_step: int) -> bool:
    if end_fraction < 0:
        raise ValueError("repa_end_fraction must be non-negative")
    if end_fraction >= 1.0:
        return True
    return optimizer_step < math.ceil(float(end_fraction) * float(total_optimizer_steps))


def sample_repa_input_mask(eligible_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Independently sample masked REPA adapter inputs from eligible positions."""
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("repa_mask_ratio must be in [0, 1]")
    eligible_mask = eligible_mask.to(torch.bool)
    if mask_ratio == 0.0:
        return torch.zeros_like(eligible_mask)
    if mask_ratio == 1.0:
        return eligible_mask
    return eligible_mask & (
        torch.rand(eligible_mask.shape, device=eligible_mask.device) < mask_ratio
    )


def build_repa_adapter_attention_mask(
    attention_mask: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    mode: str,
    has_reg_token: bool,
) -> torch.Tensor:
    """Build valid-token or structured attention for the REPA adapter."""
    if mode not in REPA_ADAPTER_ATTENTION_MODES:
        raise ValueError(
            f"repa_adapter_attention_mode must be one of "
            f"{sorted(REPA_ADAPTER_ATTENTION_MODES)}, got {mode!r}"
        )
    if attention_mask.dim() != 2:
        raise ValueError(
            f"REPA valid-token mask must be 2D, got {tuple(attention_mask.shape)}"
        )

    if mode == "full":
        if not has_reg_token:
            return attention_mask
        reg_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return torch.cat([reg_mask, attention_mask], dim=1)

    batch, length = attention_mask.shape
    if encoder_attention_mask.shape != (batch, length, length):
        raise ValueError(
            f"REPA encoder attention mask must have shape {(batch, length, length)}, "
            f"got {tuple(encoder_attention_mask.shape)}"
        )

    valid = attention_mask.to(torch.bool)
    text_mask = encoder_attention_mask.to(torch.bool)
    text_mask = text_mask & valid.unsqueeze(2) & valid.unsqueeze(1)
    if mode == "causal":
        causal = torch.ones(
            (length, length), dtype=torch.bool, device=attention_mask.device,
        ).tril()
        text_mask = text_mask & causal

    if not has_reg_token:
        return text_mask

    mask = torch.zeros(
        (batch, length + 1, length + 1),
        dtype=torch.bool,
        device=attention_mask.device,
    )
    mask[:, 1:, 1:] = text_mask
    mask[:, 0, 0] = True
    mask[:, 0, 1:] = valid
    mask[:, 1:, 0] = valid
    return mask


def build_repa_masks(
    attention_mask: torch.Tensor, cond_seq_mask: torch.Tensor,
    decoder_step_active: torch.Tensor, label_drop_mask: torch.Tensor,
    align_decoder_rows: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cond_seq_mask.dim() == 3:
        cond_seq_mask = cond_seq_mask.squeeze(-1)
    valid = attention_mask.to(torch.float32)
    condition = cond_seq_mask.to(torch.float32)
    prompt_mask = valid * condition
    prompt_mask = prompt_mask * (~label_drop_mask.to(torch.bool)).to(valid.dtype).view(-1, 1)
    response_mask = valid * (1.0 - condition)
    if not align_decoder_rows:
        active = 1.0 - decoder_step_active.to(valid.dtype).view(-1, 1)
        prompt_mask = prompt_mask * active
        response_mask = response_mask * active
    return prompt_mask, response_mask


def repa_cosine_loss(projected: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    projected = F.normalize(projected.float(), dim=-1)
    target = F.normalize(target.detach().float(), dim=-1)
    distance = (1.0 - (projected * target).sum(dim=-1)).clamp_min(0.0)
    return _masked_repa_loss(distance, mask, projected)


def repa_smooth_l1_loss(projected: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    projected = projected.float()
    distance = F.smooth_l1_loss(projected, target.detach().float(), beta=0.05, reduction="none").mean(dim=-1)
    return _masked_repa_loss(distance, mask, projected)


def repa_alignment_loss(
    projected: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if loss_type == "cos_sim":
        return repa_cosine_loss(projected, target, mask)
    if loss_type == "smooth_l1":
        return repa_smooth_l1_loss(projected, target, mask)
    raise ValueError(f"Unknown repa_loss_type={loss_type!r}")


def _masked_repa_loss(
    distance: torch.Tensor, mask: torch.Tensor, projected: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = mask.to(distance.dtype)
    valid = mask.sum()
    valid_frac = valid / max(mask.numel(), 1)
    loss = (distance * mask).sum() / torch.clamp(valid, min=1.0)
    loss = torch.where(valid > 0, loss, projected.sum() * 0.0)
    return loss, valid_frac.detach()
