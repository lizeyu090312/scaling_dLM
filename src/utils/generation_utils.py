from typing import Optional

import torch
import torch.nn as nn

from configs.config import Config, SamplingConfig
from utils.sampling_utils import restore_cond, _ode_step, _sde_step


# ============================================
# Generation utilities
# ============================================

def mask_after_eos(predicted_ids: torch.Tensor, eos_token_id: int, pad_token_id: int) -> torch.Tensor:
    """Mask everything at/after first EOS token per sequence."""
    eos_mask = (predicted_ids == eos_token_id)
    keep_mask = (eos_mask.to(torch.int32).cumsum(dim=1) == 0)
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def shift_left(x: torch.Tensor, shift_per_sample: torch.Tensor, pad_value=0, axis: int = 1) -> torch.Tensor:
    """Shift each sample left along the sequence axis; pad emptied positions."""
    if x.dim() < 2:
        raise ValueError("x must have at least batch and sequence dimensions")
    if axis < 0:
        axis = x.dim() + axis
    if axis == 0:
        raise ValueError("axis=0 is the batch axis and cannot be shifted")
    shift_per_sample = shift_per_sample.to(torch.long)
    if axis != 1:
        x = x.movedim(axis, 1)
    seq_len = x.shape[1]
    base_idx = torch.arange(seq_len, device=x.device)[None, :]
    gather_idx = shift_per_sample[:, None].to(x.device) + base_idx
    valid = gather_idx < seq_len
    gather_idx = gather_idx.clamp(0, seq_len - 1)
    if x.dim() == 2:
        shifted = torch.gather(x, 1, gather_idx)
        shifted = torch.where(valid, shifted, torch.full_like(shifted, pad_value))
    else:
        expand_shape = [-1, -1] + list(x.shape[2:])
        idx = gather_idx.view(*gather_idx.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        valid_b = valid.view(*valid.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        shifted = torch.gather(x, 1, idx)
        shifted = torch.where(valid_b, shifted, torch.full_like(shifted, pad_value))
    if axis != 1:
        shifted = shifted.movedim(1, axis)
    return shifted


def build_generation_attention_mask(
    cond_len_per_sample: torch.Tensor, max_length: int, gen_length: Optional[int],
) -> torch.Tensor:
    """Valid-token mask for conditional sampling: prompt plus generation canvas."""
    if gen_length is None:
        return torch.ones((cond_len_per_sample.shape[0], int(max_length)),
                          dtype=torch.float32, device=cond_len_per_sample.device,)
    valid_len = cond_len_per_sample.to(torch.long) + int(gen_length)
    pos = torch.arange(int(max_length), device=cond_len_per_sample.device)[None, :]
    return (pos < valid_len[:, None]).to(torch.float32)


def extract_generated_responses(
    predicted_ids: torch.Tensor, cond_len_per_sample: torch.Tensor,
    reserved_generation_length: int, eos_token_id: int, pad_token_id: int,
    truncate_generation: bool = True,
):
    """Remove prompts and return EOS-masked responses and their effective lengths."""
    cond_len_per_sample = cond_len_per_sample.to(torch.long)
    if truncate_generation:
        max_response_lengths = torch.full_like(cond_len_per_sample, int(reserved_generation_length),)
    else:
        max_response_lengths = predicted_ids.shape[1] - cond_len_per_sample

    if bool((max_response_lengths < 0).any()):
        raise ValueError("conditioning length exceeds the decoded sequence length")

    response_width = int(max_response_lengths.max().item())
    response_ids = shift_left(predicted_ids, cond_len_per_sample, pad_value=pad_token_id,)[:, :response_width]
    positions = torch.arange(response_width, device=predicted_ids.device)[None, :]
    valid = positions < max_response_lengths[:, None]
    eos_mask = (response_ids == eos_token_id) & valid
    eos_emitted = eos_mask.any(dim=1)
    first_eos = eos_mask.to(torch.int64).argmax(dim=1) + 1
    response_lengths = torch.where(eos_emitted, first_eos, max_response_lengths,)
    response_ids = torch.where(valid, response_ids, torch.full_like(response_ids, pad_token_id),)
    response_ids = mask_after_eos(response_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id,)
    return response_ids, eos_emitted, response_lengths


# ============================================
# Single-batch sampling (PyTorch)
# ============================================

@torch.no_grad()
def _generate_samples_single_batch(
    model: nn.Module,
    generator: torch.Generator,
    z: torch.Tensor,
    t_steps: torch.Tensor,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
    config: Config,
    sampling_config: SamplingConfig,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    attention_mask: Optional[torch.Tensor] = None,
    reg_z: Optional[torch.Tensor] = None,
    text_sde_generators=None,
    reg_sde_generators=None,
    *,
    stop_after: Optional[int] = None,
    step_callback=None,
    velocity_callback=None,
    x_pred_prev: Optional[torch.Tensor] = None,
    reg_x_pred_prev: Optional[torch.Tensor] = None,
):
    """Generate a batch, optionally stopping or continuing a trajectory.

    The callback receives (completed_updates, z, x_pred, reg_z, reg_x_pred).
    It must not mutate these tensors. Early-stop decoding uses its predicted
    clean states; the return value remains the updated latent state.
    For continuation, pass the remaining time grid and previous predictions.
    The optional ODE velocity callback receives (t, text_v, reg_v) before each
    update and must not mutate its inputs.
    """
    method = sampling_config.sampling_method
    if velocity_callback is not None and method != "ode":
        raise ValueError("velocity_callback requires ODE sampling")
    batch_size, max_length, d_model = z.shape
    if cond_seq is None:
        cond_seq = torch.zeros((batch_size, max_length, d_model), dtype=z.dtype, device=z.device)
        cond_seq_mask = torch.zeros((batch_size, max_length), dtype=z.dtype, device=z.device)

    step_kwargs = dict(
        model=model, config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        attention_mask=attention_mask,
    )

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(
        torch.zeros_like(z) if x_pred_prev is None else x_pred_prev,
        cond_seq, cond_seq_mask,
    )
    reg_x_pred = reg_x_pred_prev
    if reg_z is not None and reg_x_pred is None:
        reg_x_pred = torch.zeros_like(reg_z)

    n = t_steps.shape[0]
    updates = n - 1 if stop_after is None else stop_after
    if not 1 <= updates <= n - 1:
        raise ValueError("stop_after must be between 1 and the number of updates")
    sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)

    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(min(n - 2, updates)):
            t = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            if method == "sde":
                result = _sde_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    gamma=sde_gamma, generator=generator, **step_kwargs,
                    reg_z=reg_z, reg_x_pred_prev=reg_x_pred,
                    text_generators=text_sde_generators,
                    reg_generators=reg_sde_generators,
                )
            elif method == "ode":
                result = _ode_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    reg_z=reg_z, reg_x_pred_prev=reg_x_pred,
                    velocity_callback=velocity_callback, **step_kwargs,
                )
            else:
                raise ValueError(f"Invalid sampling method: {method}")
            if reg_z is None:
                z, x_pred = result
            else:
                z, x_pred, reg_z, reg_x_pred = result
            if step_callback is not None:
                step_callback(i + 1, z, x_pred, reg_z, reg_x_pred)

        # Last step always with ODE.
        if updates == n - 1:
            t = t_steps[-2].item()
            t_next = t_steps[-1].item()
            result = _ode_step(
                z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                reg_z=reg_z, reg_x_pred_prev=reg_x_pred,
                velocity_callback=velocity_callback, **step_kwargs,
            )
            if reg_z is None:
                z, x_pred = result
            else:
                z, x_pred, reg_z, reg_x_pred = result
            if step_callback is not None:
                step_callback(updates, z, x_pred, reg_z, reg_x_pred)
    return z if reg_z is None else (z, reg_z)


@torch.no_grad()
def _dlm_decode_batch(z: torch.Tensor, model: nn.Module, t_final_val,
                      config, self_cond_cfg_scale: float,
                      attention_mask: Optional[torch.Tensor] = None,
                      reg_z: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Decode z -> tokens with the DLM decoder head."""
    batch_size = z.shape[0]
    if isinstance(t_final_val, torch.Tensor) and t_final_val.dim() == 0:
        t_final = torch.full((batch_size,), t_final_val.item(), dtype=z.dtype, device=z.device)
    else:
        t_final = torch.full((batch_size,), float(t_final_val), dtype=z.dtype, device=z.device)
    sc_batch = (
        torch.full((batch_size,), float(self_cond_cfg_scale), dtype=z.dtype, device=z.device)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    reg_input = None
    if reg_z is not None:
        reg_input = (
            torch.cat([reg_z, torch.zeros_like(reg_z)], dim=-1)
            if config.self_cond_prob > 0 else reg_z
        )
    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        reg_kwargs = {"reg_x": reg_input} if reg_input is not None else {}
        model_out = model(
            z_input, t_final, deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
            attention_mask=attention_mask,
            **reg_kwargs,
        )
    decoder_logits = model_out[1]
    return decoder_logits.argmax(dim=-1)


def _build_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                    time_schedule, sde_gamma, suffix):
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    sde_str = f"-gamma{sde_gamma}" if sampling_method == "sde" else ""
    return f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}{sccfg_str}{ts_str}{sde_str}-{suffix}"
