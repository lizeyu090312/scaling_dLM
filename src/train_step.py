"""One mini-batch forward/backward for the ELF diffusion language model.

Each example in the batch independently picks the decoder (CE) or denoiser
(L2) branch via a Bernoulli draw at `decoder_prob`. A single forward consumes
a mixed input (decoder_z for decoder rows, denoiser_z for denoiser rows) and
both heads run; the CE / L2 losses are then masked to their respective rows
and combined with a single denominator. Self-conditioning + CFG guidance is
applied on the denoiser branch only.
"""

import contextlib
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.text_encoder import encoder_family_from_model_type, feature_standardize
from utils.train_utils import TrainState, ema_decay_for_update, ema_decay_key, ema_update, unwrap_model
from utils.encoder_utils import apply_label_drop_to_encoder_attention_mask, encode_text
from utils.teacher_utils import build_auxiliary_content_mask
from utils.repa_utils import (
    REPA_ADAPTER_ATTENTION_MODES, build_repa_adapter_attention_mask,
    build_repa_masks, repa_alignment_loss, repa_is_active,
    same_repa_teacher, sample_repa_input_mask, validate_repa_token_ids,
    validate_repa_shift,
)
from utils.sampling_utils import (
    sample_cfg_scale, add_noise, sample_timesteps,
    net_out_to_v_x, restore_cond,
)


def _trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _reg_loss_terms(
    per_row_loss: torch.Tensor,
    row_mask: torch.Tensor,
    decoder_prob: float,
    graph_tensor: torch.Tensor,
):
    valid = row_mask.sum()
    loss_sum = (per_row_loss * row_mask).sum()
    expected_rows = per_row_loss.numel() * (1.0 - float(decoder_prob))
    objective = loss_sum / expected_rows if expected_rows > 0 else graph_tensor.sum() * 0.0
    conditional = torch.where(
        valid > 0,
        loss_sum / torch.clamp(valid, min=1.0),
        graph_tensor.sum() * 0.0,
    )
    return objective, conditional, valid / max(per_row_loss.numel(), 1)


def train_step(
    state: TrainState,
    encoder: Optional[nn.Module],
    batch: Dict[str, torch.Tensor],
    config,
    total_optimizer_steps: int,
    repa_teacher: Optional[nn.Module] = None,
    teacher_target_provider=None,
    repa_special_token_ids=(),
    backward_loss_component: Optional[str] = None,
) -> Tuple[TrainState, Dict[str, float]]:
    """Perform a single training step."""
    if backward_loss_component not in {None, "dlm", "reg", "repa"}:
        raise ValueError(
            "backward_loss_component must be one of None, 'dlm', 'reg', or 'repa'"
        )
    device = next(state.model.parameters()).device
    dtype = next(state.model.parameters()).dtype
    use_bf16 = bool(getattr(config, "use_bf16", True)) and device.type == "cuda"
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    latent_mean, latent_std = config.latent_mean, config.latent_std
    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale
    repa_enabled = bool(getattr(config, "repa_enabled", False))
    reg_enabled = bool(getattr(config, "reg_enabled", False))
    reg_loss_weight = float(getattr(config, "reg_loss_weight", 0.03))
    if reg_loss_weight < 0:
        raise ValueError("reg_loss_weight must be non-negative")
    repa_prompt_loss_weight = float(getattr(config, "repa_prompt_loss_weight", 0.0))
    repa_response_loss_weight = float(getattr(config, "repa_response_loss_weight", 1.0))
    repa_shift = validate_repa_shift(config)
    repa_projector_type = getattr(config, "repa_projector_type", "mlp")
    repa_adapter_attention_mode = getattr(config, "repa_adapter_attention_mode", "full")
    repa_mask_ratio = float(getattr(config, "repa_mask_ratio", 0.0))
    if repa_prompt_loss_weight < 0 or repa_response_loss_weight < 0:
        raise ValueError("REPA prompt and response loss weights must be non-negative")
    if repa_projector_type not in {"mlp", "transformer"}:
        raise ValueError("repa_projector_type must be 'mlp' or 'transformer'")
    if repa_adapter_attention_mode not in REPA_ADAPTER_ATTENTION_MODES:
        raise ValueError(
            f"repa_adapter_attention_mode must be one of "
            f"{sorted(REPA_ADAPTER_ATTENTION_MODES)}"
        )
    if repa_projector_type == "mlp" and repa_mask_ratio != 0.0:
        raise ValueError("repa_mask_ratio must be 0 for the MLP REPA projector")
    if repa_projector_type == "mlp" and repa_adapter_attention_mode != "full":
        raise ValueError("non-full REPA adapter attention requires the transformer REPA projector")
    repa_t_min = float(getattr(config, "repa_t_min", 0.0))
    repa_t_max = float(getattr(config, "repa_t_max", 1.0))
    if not (0.0 <= repa_t_min <= repa_t_max <= 1.0):
        raise ValueError(
            "repa_t_min and repa_t_max must satisfy 0 <= repa_t_min <= repa_t_max <= 1"
        )
    accum_steps = max(config.grad_accum_steps, 1)
    optimizer_step = state.step // accum_steps
    repa_active = repa_enabled and repa_is_active(config.repa_end_fraction, total_optimizer_steps, optimizer_step)

    gen = state.dropout_generator

    # encoder_attention_mask: cond sees cond, x sees all
    input_ids = batch["input_ids"].to(device, non_blocking=True).long()
    encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
    cond_seq_mask = batch["cond_seq_mask"].to(device, dtype=torch.float32, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
    label_drop_mask = batch.get(
        "label_drop_mask", torch.zeros((input_ids.shape[0],), dtype=torch.bool),
    )
    has_label_drop = bool(torch.any(label_drop_mask))
    label_drop_mask = label_drop_mask.to(device, non_blocking=True)
    cached_latents = "cached_text_latents" in batch
    if cached_latents and config.label_drop_prob > 0:
        raise ValueError("Cached latents require label_drop_prob=0")
    if repa_shift > 0:
        if cached_latents:
            # The compressed-latent cache writer only supports Qwen3.
            teacher_family = "qwen3"
        elif teacher_target_provider is not None:
            teacher_family = getattr(teacher_target_provider, "family", None)
        else:
            teacher = repa_teacher if repa_teacher is not None else encoder
            teacher_config = getattr(teacher, "config", None)
            teacher_family = getattr(teacher_config, "family", None)
            if teacher_family is None:
                model_config = getattr(getattr(teacher, "model", None), "config", None)
                teacher_family = encoder_family_from_model_type(getattr(model_config, "model_type", ""))
        validate_repa_shift(config, teacher_family)

    # Label drop before encoding: prevent target tokens from attending to
    # condition tokens so x0 is truly unconditional for dropped samples.
    if config.label_drop_prob > 0:
        encoder_attention_mask = apply_label_drop_to_encoder_attention_mask(
            encoder_attention_mask, cond_seq_mask, label_drop_mask,
        )

    repa_reuses_encoder_pass = (
        not cached_latents
        and repa_active
        and repa_teacher is None
        and (
            getattr(config, "repa_teacher_model_name", None) is None
            or same_repa_teacher(config, getattr(encoder, "config", None))
        )
    )
    if cached_latents:
        # Compressed codes stand in for raw Qwen states, so clean ELF inputs
        # keep the same per-vector standardization as the live encoder path.
        x0 = batch["cached_text_latents"].to(
            device, dtype=dtype, non_blocking=True,
        )
        x0 = feature_standardize(x0)
        raw_x0 = None
    else:
        encoded_text = encode_text(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            encoder=encoder,
            latent_mean=latent_mean,
            latent_std=latent_std,
            use_bf16=use_bf16,
            return_raw=repa_reuses_encoder_pass,
        )
        if repa_reuses_encoder_pass:
            x0, raw_x0 = encoded_text
            raw_x0 = raw_x0.to(dtype)
        else:
            x0 = encoded_text
            raw_x0 = None
    x0 = x0.to(dtype)

    model = state.model

    batch_size, seq_length = x0.shape[0], x0.shape[1]
    reg_x0 = None
    repa_target = None
    teacher_views = {}
    if cached_latents and reg_enabled:
        if "cached_reg_latent" not in batch:
            raise ValueError("REG training requires cached_reg_latent")
        reg_x0 = feature_standardize(
            batch["cached_reg_latent"].to(
                device, dtype=dtype, non_blocking=True,
            ),
        ).detach()
    if not cached_latents and reg_enabled:
        teacher_views["reg"] = (
            getattr(config, "reg_teacher_dim", None),
            getattr(config, "reg_teacher_layer", None),
        )
    if (
        not cached_latents
        and repa_active
        and (reg_enabled or (not repa_reuses_encoder_pass and repa_teacher is None))
    ):
        teacher_views["repa"] = (
            getattr(config, "repa_teacher_dim", None),
            getattr(config, "repa_teacher_layer", None),
        )

    teacher_targets = None
    if teacher_views:
        if teacher_target_provider is None:
            raise ValueError("REG or formatted/multi-view REPA requires teacher_target_provider")
        primary_label_drop_mask = None if reg_enabled or not has_label_drop else label_drop_mask
        teacher_targets = teacher_target_provider.encode(
            input_ids,
            attention_mask,
            cond_seq_mask,
            primary_label_drop_mask,
            teacher_views,
            include_reg=reg_enabled,
            use_bf16=use_bf16,
        )
        if reg_enabled:
            reg_x0 = teacher_targets.reg["reg"].to(dtype)
        if (
            reg_enabled and repa_active and not repa_reuses_encoder_pass
            and has_label_drop
        ):
            dropped = torch.where(label_drop_mask)[0]
            dropped_targets = teacher_target_provider.encode(
                input_ids[dropped],
                attention_mask[dropped],
                cond_seq_mask[dropped],
                torch.ones_like(label_drop_mask[dropped]),
                {"repa": teacher_views["repa"]},
                include_reg=False,
                use_bf16=use_bf16,
            )
            repa_text = teacher_targets.text["repa"].clone()
            repa_text[dropped] = dropped_targets.text["repa"]
            teacher_targets.text["repa"] = repa_text

    if repa_active:
        if cached_latents:
            if "cached_repa_latents" not in batch:
                raise ValueError("REPA training requires cached_repa_latents")
            # Token-level REPA targets are raw in the live teacher path.
            repa_text_target = batch["cached_repa_latents"].to(
                device, dtype=dtype, non_blocking=True,
            ).detach()
        elif repa_reuses_encoder_pass:
            repa_text_target = raw_x0.detach()
        elif teacher_targets is not None:
            repa_text_target = teacher_targets.text["repa"].to(dtype).detach()
        elif repa_teacher is not None:
            validate_repa_token_ids(input_ids, repa_teacher)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                if hasattr(repa_teacher, "encode_views"):
                    repa_text_target = repa_teacher.encode_views(
                        input_ids,
                        encoder_attention_mask,
                        {"repa": (
                            getattr(config, "repa_teacher_dim", None),
                            getattr(config, "repa_teacher_layer", None),
                        )},
                        deterministic=True,
                        normalize=False,
                    )["repa"]
                else:
                    repa_text_target = repa_teacher(
                        input_ids=input_ids,
                        attention_mask=encoder_attention_mask,
                        deterministic=True,
                    )
                repa_text_target = repa_text_target.to(dtype).detach()
        else:
            repa_text_target = x0.detach()

        if repa_text_target.shape[:2] != x0.shape[:2]:
            raise ValueError(
                f"REPA teacher output must match ELF text grid "
                f"({tuple(repa_text_target.shape[:2])} vs {tuple(x0.shape[:2])})"
            )
        if reg_enabled:
            if cached_latents:
                if "cached_repa_reg_latent" not in batch:
                    raise ValueError("Cached REPA+REG training requires a REPA prefix target")
                repa_reg_target = feature_standardize(
                    batch["cached_repa_reg_latent"].to(
                        device, dtype=dtype, non_blocking=True,
                    ),
                ).detach()
            else:
                repa_reg_target_source = getattr(config, "repa_reg_target_source", "repa")
                repa_reg_target = teacher_targets.reg[repa_reg_target_source].to(dtype).detach()
            if repa_reg_target.shape[-1] != repa_text_target.shape[-1]:
                raise ValueError(
                    "repa_reg_target_source='reg' requires reg_teacher_dim "
                    "to match repa_teacher_dim"
                )
            repa_target = torch.cat([repa_reg_target, repa_text_target], dim=1)
        else:
            repa_target = repa_text_target

    t = sample_timesteps(
        batch_size,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
        device=device, dtype=dtype,
    )
    noise = torch.randn(x0.shape, dtype=dtype, device=device)
    reg_noise = torch.randn(reg_x0.shape, dtype=dtype, device=device) if reg_enabled else None

    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = torch.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - cond_seq_mask)
    model_attention_mask = attention_mask if bool(getattr(config, "use_model_attention_mask", False)) else None
    cond_seq_mask = cond_seq_mask.unsqueeze(-1)  # (B, S, 1)

    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)
    denoiser_reg_z = add_noise(reg_x0, reg_noise, t, config) if reg_enabled else None

    drop = label_drop_mask.unsqueeze(1)  # (B, 1)
    if config.label_drop_prob > 0:
        denoiser_z = torch.where(drop.unsqueeze(-1) & (cond_seq_mask > 0), torch.zeros_like(denoiser_z), denoiser_z)
        x0 = torch.where(drop.unsqueeze(-1) & (cond_seq_mask > 0), torch.zeros_like(x0), x0)

    decoder_targets = input_ids  # (B, S)

    # Per-example branching: each example independently picks decoder (CE) vs.
    # denoiser (L2) instead of one scalar bernoulli per step. Smooths training
    decoder_step_active = torch.bernoulli(
        torch.full((batch_size,), decoder_prob, dtype=torch.float32),
        generator=gen,
    ).to(device=device, dtype=dtype)  # (B,) — 1.0 = decoder mode, 0.0 = denoiser
    decoder_mask_B11 = decoder_step_active.view(-1, 1, 1)
    decoder_mask_B1 = decoder_step_active.view(-1, 1)

    # Decoder-branch input: logit-normal-noised latent (decoder_z) at t=1
    decoder_z_vals = (
        torch.randn((batch_size * seq_length,), dtype=dtype, device=device)
        * config.decoder_p_std + config.decoder_p_mean
    )
    decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = torch.randn(x0.shape, dtype=dtype, device=device) * decoder_noise_scale
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise
    decoder_reg_z = None
    if reg_enabled:
        decoder_reg_vals = (
            torch.randn((batch_size,), dtype=dtype, device=device)
            * config.decoder_p_std + config.decoder_p_mean
        )
        decoder_reg_lambda = torch.sigmoid(decoder_reg_vals).reshape(batch_size, 1, 1)
        decoder_reg_noise = torch.randn(reg_x0.shape, dtype=dtype, device=device) * decoder_noise_scale
        decoder_reg_z = decoder_reg_lambda * reg_x0 + (1 - decoder_reg_lambda) * decoder_reg_noise

    t_expanded = t.reshape(-1, 1, 1)
    v_target = (x0 - denoiser_z) / torch.clamp(1 - t_expanded, min=t_eps)
    reg_v_target = None
    if reg_enabled:
        reg_v_target = (reg_x0 - denoiser_reg_z) / torch.clamp(1 - t_expanded, min=t_eps)

    if self_cond_prob > 0:
        use_self_cond_mask = (
            (torch.rand((batch_size,), dtype=dtype, device=device) < self_cond_prob)
            .reshape(-1, 1, 1).to(dtype)
        )
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            batch_size,
            cfg_min=config.self_cond_cfg_min, cfg_max=config.self_cond_cfg_max,
            dtype=dtype, device=device,
        )
    else:
        self_cond_cfg_scale = None

    def model_v_x(model_output, z, t_input, reg_z=None):
        if not reg_enabled:
            v, x = net_out_to_v_x(model_output, z, t_input, t_eps)
            return v, x, None, None
        text_output, _, reg_output = model_output
        v, x = net_out_to_v_x(text_output, z, t_input, t_eps)
        reg_v, reg_x = net_out_to_v_x(reg_output, reg_z, t_input, t_eps)
        return v, x, reg_v, reg_x

    def compute_shared_uncond(z, t_input, x_tokens, reg_z=None):
        """Unconditional forward shared by self-cond-init and sc-cfg-uncond."""
        z_uncond = restore_cond(torch.zeros_like(z), x_tokens, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        reg_input_uncond = (
            torch.cat([reg_z, torch.zeros_like(reg_z)], dim=-1)
            if reg_enabled else None
        )
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            reg_kwargs = {"reg_x": reg_input_uncond} if reg_enabled else {}
            net_out_uncond = model(
                z_input_uncond, t_input,
                deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale, attention_mask=model_attention_mask,
                **reg_kwargs,
            )
        return net_out_uncond

    def get_sc_cond_and_uncond(
        z, t_input, cond_mask, x_tokens, shared_net_out_uncond, reg_z=None,
    ):
        if config.self_cond_prob == 0:
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                reg_kwargs = {"reg_x": reg_z} if reg_enabled else {}
                net_out_uncond = model(
                    z, t_input,
                    deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale, attention_mask=model_attention_mask,
                    **reg_kwargs,
                )
            v_uncond, _, reg_v_uncond, _ = model_v_x(
                net_out_uncond, z, t_input, reg_z,
            )
            return v_uncond, v_uncond, reg_v_uncond, reg_v_uncond

        v_uncond, x_uncond, reg_v_uncond, reg_x_uncond = model_v_x(
            shared_net_out_uncond, z, t_input, reg_z,
        )
        x_uncond = restore_cond(x_uncond, x_tokens, cond_mask)

        z_input_cond = torch.cat([z, x_uncond], dim=-1)
        reg_input_cond = (
            torch.cat([reg_z, reg_x_uncond], dim=-1)
            if reg_enabled else None
        )
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            reg_kwargs = {"reg_x": reg_input_cond} if reg_enabled else {}
            net_out_cond = model(
                z_input_cond, t_input,
                deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale, attention_mask=model_attention_mask,
                **reg_kwargs,
            )
        v_cond, _, reg_v_cond, _ = model_v_x(net_out_cond, z, t_input, reg_z)
        return v_cond, v_uncond, reg_v_cond, reg_v_uncond

    def get_sc_guided_v(
        z, t_input, base_v_target, x_tokens, shared_net_out_uncond,
        reg_z=None, base_reg_v_target=None,
    ):
        """v target with self-conditioning guidance."""
        v_cond, v_uncond, reg_v_cond, reg_v_uncond = get_sc_cond_and_uncond(
            z, t_input, cond_mask=cond_seq_mask, x_tokens=x_tokens,
            shared_net_out_uncond=shared_net_out_uncond, reg_z=reg_z,
        )
        sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)
        sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
        sc_guidance = torch.where(use_self_cond_mask.bool(), sc_guidance, torch.zeros_like(sc_guidance))
        v_target_out = (base_v_target + sc_guidance).detach()
        if not reg_enabled:
            return v_target_out, None
        reg_sc_guidance = (1 - 1 / sc_w) * (reg_v_cond - reg_v_uncond)
        reg_sc_guidance = torch.where(
            use_self_cond_mask.bool(), reg_sc_guidance, torch.zeros_like(reg_sc_guidance),
        )
        return v_target_out, (base_reg_v_target + reg_sc_guidance).detach()

    def get_v_target(
        z, t_input, base_v_target, x_tokens, shared_net_out_uncond,
        reg_z=None, base_reg_v_target=None,
    ):
        """Compute final v target with self-conditioning guidance."""
        if config.num_self_cond_cfg_tokens > 0 and config.self_cond_prob > 0:
            return get_sc_guided_v(
                z, t_input, base_v_target=base_v_target, x_tokens=x_tokens,
                shared_net_out_uncond=shared_net_out_uncond,
                reg_z=reg_z, base_reg_v_target=base_reg_v_target,
            )
        return base_v_target, base_reg_v_target

    model.train()

    # Per-example branching: build a mixed input (decoder_z for decoder-mode
    # rows, denoiser_z for denoiser-mode rows). One forward computes both
    # heads; we mask CE / L2 losses to their respective rows. 
    denoiser_t = t
    decoder_t = torch.ones_like(t)
    t_mixed = decoder_step_active * decoder_t + (1.0 - decoder_step_active) * t  # (B,)
    z_mixed = decoder_mask_B11 * decoder_z + (1.0 - decoder_mask_B11) * denoiser_z
    reg_z_mixed = (
        decoder_mask_B11 * decoder_reg_z + (1.0 - decoder_mask_B11) * denoiser_reg_z
        if reg_enabled else None
    )

    # Self-cond shared forward (run on denoiser_z / t — only relevant for
    # denoiser-mode rows; decoder-mode rows zero out the self-cond half below).
    if self_cond_prob > 0 or config.num_self_cond_cfg_tokens > 0:
        shared_net_out_uncond = compute_shared_uncond(
            denoiser_z, denoiser_t, x0, denoiser_reg_z,
        )
    else:
        shared_net_out_uncond = None

    reg_model_input = None
    if config.self_cond_prob > 0:
        _, x_pred_init, _, reg_x_pred_init = model_v_x(
            shared_net_out_uncond, denoiser_z, denoiser_t, denoiser_reg_z,
        )
        x_pred_init = restore_cond(x_pred_init, x0, cond_seq_mask)
        x_pred_cond = x_pred_init * use_self_cond_mask.to(dtype)
        x_pred_cond = restore_cond(x_pred_cond, x0, cond_seq_mask)
        # Zero the self-cond half for decoder-mode rows (matches the old
        # `cat([decoder_z, zeros], -1)` decoder-branch input).
        sc_half = x_pred_cond * (1.0 - decoder_mask_B11)
        model_input = torch.cat([z_mixed, sc_half], dim=-1)
        if reg_enabled:
            reg_sc_half = reg_x_pred_init * use_self_cond_mask.to(dtype)
            reg_sc_half = reg_sc_half * (1.0 - decoder_mask_B11)
            reg_model_input = torch.cat([reg_z_mixed, reg_sc_half], dim=-1)
    else:
        model_input = z_mixed
        reg_model_input = reg_z_mixed

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        reg_kwargs = {"reg_x": reg_model_input} if reg_enabled else {}
        model_out = model(
            model_input, t_mixed,
            deterministic=False,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=decoder_step_active,  # (B,) tensor
            return_repa_hidden=repa_active,
            repa_depth=config.repa_depth if repa_active else None, attention_mask=model_attention_mask,
            **reg_kwargs,
        )
    if repa_active and reg_enabled:
        net_out, decoder_logits, reg_net_out, repa_hidden = model_out
    elif repa_active:
        net_out, decoder_logits, repa_hidden = model_out
        reg_net_out = None
    elif reg_enabled:
        net_out, decoder_logits, reg_net_out = model_out
    else:
        net_out, decoder_logits = model_out
        reg_net_out = None

    # CE per-token (used on decoder-mode rows).
    log_probs = F.log_softmax(decoder_logits.to(torch.float32), dim=-1)
    ce_per_token = -log_probs.gather(-1, decoder_targets.unsqueeze(-1)).squeeze(-1)

    # L2 per-token (used on denoiser-mode rows). v_pred is extracted with
    # (denoiser_z, t) — meaningful only for denoiser rows; decoder rows are
    # masked out below.
    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, denoiser_t, t_eps)
    v_final_target, reg_v_final_target = get_v_target(
        denoiser_z, denoiser_t, base_v_target=v_target, x_tokens=x0,
        shared_net_out_uncond=shared_net_out_uncond,
        reg_z=denoiser_reg_z, base_reg_v_target=reg_v_target,
    )
    l2_per_token = ((v_pred - v_final_target) ** 2).mean(dim=-1)

    # Masks: each position is "alive" for exactly one branch.
    loss_mask_f = loss_mask.to(ce_per_token.dtype)
    ce_mask = loss_mask_f * decoder_mask_B1
    l2_mask = loss_mask_f * (1.0 - decoder_mask_B1)

    # Combined loss with a single denominator. In expectation this is
    # decoder_prob * mean_CE + (1 - decoder_prob) * mean_L2.
    total_sum = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    base_loss = total_sum / torch.clamp(loss_mask_f.sum(), min=1.0)
    loss = base_loss

    # Per-branch metrics: mean per-token within each branch.
    ce_loss_val = ((ce_per_token * ce_mask).sum()
                   / torch.clamp(ce_mask.sum(), min=1.0)).detach()
    l2_loss_val = ((l2_per_token * l2_mask).sum()
                   / torch.clamp(l2_mask.sum(), min=1.0)).detach()

    reg_loss = base_loss.new_zeros(())
    reg_conditional_loss = base_loss.new_zeros(())
    reg_valid_frac = base_loss.new_zeros(())
    if reg_enabled:
        reg_v_pred, _ = net_out_to_v_x(reg_net_out, denoiser_reg_z, denoiser_t, t_eps)
        reg_l2_per_row = ((reg_v_pred - reg_v_final_target) ** 2).mean(dim=(1, 2))
        reg_row_mask = 1.0 - decoder_step_active.to(reg_l2_per_row.dtype)
        reg_loss, reg_conditional_loss, reg_valid_frac = _reg_loss_terms(
            reg_l2_per_row, reg_row_mask, decoder_prob, reg_net_out,
        )
        reg_valid_frac = reg_valid_frac.detach()
        loss = loss + reg_loss_weight * reg_loss

    repa_loss = base_loss.new_zeros(())
    repa_prompt_loss = base_loss.new_zeros(())
    repa_response_loss = base_loss.new_zeros(())
    repa_w = 0.0
    repa_valid_frac = base_loss.new_zeros(())
    repa_prompt_valid_frac = base_loss.new_zeros(())
    repa_response_valid_frac = base_loss.new_zeros(())
    if repa_active:
        inner_model = unwrap_model(model)
        if inner_model.repa_projector is None:
            raise ValueError("repa_enabled=True requires model.repa_projector")
        if inner_model.repa_projector_type != repa_projector_type:
            raise ValueError(
                "REPA projector type in config does not match model "
                f"({repa_projector_type} vs {inner_model.repa_projector_type})"
            )
        align_decoder_rows = bool(config.repa_align_decoder_rows)
        repa_attention_mask = build_auxiliary_content_mask(
            input_ids, attention_mask, repa_special_token_ids,
        ).to(attention_mask.dtype)
        prompt_mask, response_mask = build_repa_masks(
            repa_attention_mask, cond_seq_mask, decoder_step_active, label_drop_mask,
            align_decoder_rows=align_decoder_rows,
        )
        prompt_mask = prompt_mask.to(device)
        response_mask = response_mask.to(device)
        adapter_eligible_mask = repa_attention_mask.to(device)
        if reg_enabled:
            reg_repa_mask = torch.ones(
                (batch_size, 1), dtype=response_mask.dtype, device=response_mask.device,
            )
            if not align_decoder_rows:
                reg_repa_mask = reg_repa_mask * (1.0 - decoder_step_active.view(-1, 1))
            prompt_mask = torch.cat([torch.zeros_like(reg_repa_mask), prompt_mask], dim=1)
            response_mask = torch.cat([reg_repa_mask, response_mask], dim=1)
            adapter_eligible_mask = torch.cat([
                torch.ones_like(reg_repa_mask), adapter_eligible_mask,
            ], dim=1)
        if repa_shift > 0:
            # With shift=2, P1 P2 P3 | y1 y2 y3 targets P2 P3 y1 at y1 y2 y3.
            text_start = int(reg_enabled)
            text_target = repa_target[:, text_start:]
            shifted_text_target = torch.zeros_like(text_target)
            shifted_source_mask = torch.zeros_like(repa_attention_mask)
            if repa_shift < text_target.shape[1]:
                shifted_text_target[:, repa_shift:] = text_target[:, :-repa_shift]
                shifted_source_mask[:, repa_shift:] = repa_attention_mask[:, :-repa_shift]
            if reg_enabled:
                repa_target = torch.cat([repa_target[:, :1], shifted_text_target,], dim=1)
                shifted_source_mask = torch.cat([torch.ones_like(reg_repa_mask), shifted_source_mask,], dim=1)
            else:
                repa_target = shifted_text_target
            response_mask = response_mask * shifted_source_mask
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            if repa_projector_type == "transformer":
                adapter_attention_mask = build_repa_adapter_attention_mask(
                    attention_mask,
                    encoder_attention_mask,
                    repa_adapter_attention_mode,
                    has_reg_token=reg_enabled,
                )
                repa_input_mask = sample_repa_input_mask(
                    adapter_eligible_mask, repa_mask_ratio,
                )
                repa_projected = inner_model.repa_projector(
                    repa_hidden,
                    input_mask=repa_input_mask,
                    attention_mask=adapter_attention_mask,
                )
            else:
                repa_projected = inner_model.repa_projector(repa_hidden)
        repa_t_mask = ((t >= repa_t_min) & (t <= repa_t_max)).to(prompt_mask.dtype).view(-1, 1)
        repa_t_mask = torch.where(
            (decoder_step_active > 0).view(-1, 1),
            torch.ones_like(repa_t_mask),
            repa_t_mask,
        )
        prompt_mask = prompt_mask * repa_t_mask
        response_mask = response_mask * repa_t_mask
        if repa_projected.shape != repa_target.shape:
            raise ValueError(
                f"REPA projection and target shapes must match "
                f"({tuple(repa_projected.shape)} vs {tuple(repa_target.shape)})"
            )
        loss_type = getattr(config, "repa_loss_type", "cos_sim")
        if repa_prompt_loss_weight > 0:
            repa_prompt_loss, repa_prompt_valid_frac = repa_alignment_loss(
                repa_projected, repa_target, prompt_mask, loss_type,
            )
        else:
            repa_prompt_loss = repa_projected.sum() * 0.0
        if repa_response_loss_weight > 0:
            repa_response_loss, repa_response_valid_frac = repa_alignment_loss(
                repa_projected, repa_target, response_mask, loss_type,
            )
        else:
            repa_response_loss = repa_projected.sum() * 0.0
        repa_loss = (
            repa_prompt_loss_weight * repa_prompt_loss
            + repa_response_loss_weight * repa_response_loss
        )
        active_mask = torch.zeros_like(prompt_mask)
        if repa_prompt_loss_weight > 0:
            active_mask = active_mask + prompt_mask
        if repa_response_loss_weight > 0:
            active_mask = active_mask + response_mask
        repa_valid_frac = active_mask.clamp_max(1.0).sum() / max(active_mask.numel(), 1)
        repa_w = float(config.repa_strength)
        loss = loss + repa_w * repa_loss
    elif repa_enabled:  # ensure REPA parameters are used if they exist, due to potentially turning off REPA
        inner_model = unwrap_model(model)
        if inner_model.repa_projector is None:
            raise ValueError("repa_enabled=True requires model.repa_projector")
        loss = loss + 0.0 * sum(p.sum() for p in inner_model.repa_projector.parameters())

    if backward_loss_component == "dlm":
        backward_loss = base_loss
    elif backward_loss_component == "reg":
        if not reg_enabled:
            raise ValueError("REG loss cannot be selected when REG is disabled")
        backward_loss = reg_loss
    elif backward_loss_component == "repa":
        if not repa_active:
            raise ValueError("REPA loss cannot be selected when REPA is inactive")
        backward_loss = repa_loss
    else:
        backward_loss = loss

    state.step += 1
    is_optimizer_step = (state.step % accum_steps) == 0

    sync_ctx = model.no_sync() if (not is_optimizer_step and hasattr(model, 'no_sync')) else contextlib.nullcontext()
    with sync_ctx:
        (backward_loss / accum_steps).backward()

    if is_optimizer_step:
        torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=1.0)
        state.optimizer.step()
        if state.lr_scheduler is not None:
            state.lr_scheduler.step()
        optimizer_step = state.step // accum_steps
        for target_decay in config.ema_decay1:
            key = ema_decay_key(target_decay)
            ema_decay = ema_decay_for_update(target_decay, optimizer_step, config.ema_warmup_updates)
            ema_update(state.ema_params1[key], state.model, ema_decay)
        state.optimizer.zero_grad(set_to_none=True)

    metrics = {
        "loss": loss.detach(),
        "base_loss": base_loss.detach(),
        "l2_loss": l2_loss_val,
        "ce_loss": ce_loss_val,
        "reg": reg_loss.detach(),
        "reg_conditional": reg_conditional_loss.detach(),
        "reg_w": torch.as_tensor(reg_loss_weight if reg_enabled else 0.0, device=device, dtype=torch.float32),
        "reg_valid_frac": reg_valid_frac.detach(),
        "repa": repa_loss.detach(),
        "repa_prompt": repa_prompt_loss.detach(),
        "repa_prompt_w": torch.as_tensor(repa_prompt_loss_weight if repa_enabled else 0.0, device=device, dtype=torch.float32),
        "repa_prompt_valid_frac": repa_prompt_valid_frac.detach(),
        "repa_response": repa_response_loss.detach(),
        "repa_response_w": torch.as_tensor(repa_response_loss_weight if repa_enabled else 0.0, device=device, dtype=torch.float32),
        "repa_response_valid_frac": repa_response_valid_frac.detach(),
        "repa_w": torch.as_tensor(repa_w, device=device, dtype=torch.float32),
        "repa_valid_frac": repa_valid_frac.detach(),
    }
    return state, metrics
