from typing import Optional

import torch
import torch.nn.functional as F


def randn_per_example(shape, generators, *, dtype, device):
    """Draw a batch without coupling any sample to its batch neighbours."""
    if len(shape) < 1 or len(generators) != shape[0]:
        raise ValueError("one generator is required for each batch row")
    row_shape = (1, *shape[1:])
    return torch.cat([
        torch.randn(row_shape, generator=generator, dtype=dtype, device=device)
        for generator in generators
    ], dim=0)


# ============================================
# Noise Schedulers (how to compute z from x0 and noise)
# ============================================

def add_noise(x0, noise, t, config, cond_seq_mask=None):
    """Flow-matching interpolation z = t*x0 + (1-t)*noise*scale, preserving cond tokens."""
    t_expanded = t.reshape(-1, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z
    return z


# ============================================
# Time Schedulers (how to sample t)
# ============================================

def sample_timesteps(
    batch_size: int,
    P_mean: float = -0.8,
    P_std: float = 0.8,
    time_schedule: str = 'logit_normal',
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
):
    """Sample timesteps using various time schedules.

    Args:
        batch_size: Number of samples
        P_mean: Mean for logit-normal distribution
        P_std: Std for logit-normal distribution
        time_schedule: 'logit_normal' or 'uniform'

    Returns:
        Sampled timesteps in [0, 1]
    """
    if time_schedule == 'logit_normal':
        z = torch.randn((batch_size,), generator=generator, dtype=dtype, device=device) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == 'uniform':
        return torch.rand((batch_size,), generator=generator, dtype=dtype, device=device)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(
    n_steps: int, time_schedule: str = "logit_normal",
    P_mean: float = -0.8, P_std: float = 0.8,
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return a length-(n_steps+1) tensor of t values in [0, 1] for a sampling run.

    - "uniform": evenly-spaced linspace from 0 to 1 (deterministic).
    - "logit_normal": sorted logit-normal samples with 0 / 1 endpoints (random).
    """
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype, device=device)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(
            batch_size=n_steps - 1,
            P_mean=P_mean, P_std=P_std, time_schedule=time_schedule,
            device=device, dtype=dtype, generator=generator,
        )
        steps = torch.sort(steps).values
        endpoints_lo = torch.zeros((1,), dtype=dtype, device=steps.device)
        endpoints_hi = torch.ones((1,), dtype=dtype, device=steps.device)
        return torch.cat([endpoints_lo, steps, endpoints_hi], dim=0)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


# ============================================
# CFG Scale Sampling (how to sample cfg scale)
# ============================================

def sample_cfg_scale(batch_size, cfg_min=0.0, cfg_max=3.0,
                     dtype=torch.float32, device=None):
    """Sample CFG scale from log-uniform distribution in [cfg_min, cfg_max]."""
    u = torch.rand((batch_size,), dtype=dtype, device=device)
    a = float(1.0 + cfg_min)
    b = float(1.0 + cfg_max)
    log_ratio = torch.tensor(b / a, dtype=dtype, device=u.device).log()
    return a * torch.exp(u * log_ratio) - 1.0


# ============================================
# Conditioning helpers (preserve clean tokens during sampling)
# ============================================

def restore_cond(z_updated, cond_seq, cond_seq_mask):
    """Restore clean conditioning tokens in z after a denoising step."""
    mask = cond_seq_mask
    target_ndim = max(z_updated.dim(), cond_seq.dim())
    while mask.dim() < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(v, x, cond_seq, cond_seq_mask):
    """Restore cond positions: x -> clean cond_seq, v -> 0 (cond tokens don't move)."""
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


# ============================================
# Flow-matching forward passes (with optional self-cond / CFG)
# ============================================

def net_out_to_v_x(net_out, z, t, t_eps=5e-2):
    """Convert x_pred network output to v and x.

    When the model returns a tuple (denoised_output, decoder_logits),
    decoder logits are discarded here (used separately in training).
    """
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.reshape(-1, 1, 1)
    x = net_out
    denom = torch.clamp(1.0 - t_reshaped, min=t_eps)
    v = (x - z) / denom
    return v, x


def _flow_model_vx(
    model, text_input, text_z, t_batch, t_eps,
    *, reg_input=None, reg_z=None, self_cond_cfg_scale=None, attention_mask=None,
):
    reg_kwargs = {"reg_x": reg_input} if reg_input is not None else {}
    model_out = model(
        text_input,
        t_batch,
        deterministic=True,
        self_cond_cfg_scale=self_cond_cfg_scale,
        attention_mask=attention_mask,
        **reg_kwargs,
    )
    if reg_input is None:
        return net_out_to_v_x(model_out, text_z, t_batch, t_eps)
    text_out, _, reg_out = model_out
    text_v, text_x = net_out_to_v_x(text_out, text_z, t_batch, t_eps)
    reg_v, reg_x = net_out_to_v_x(reg_out, reg_z, t_batch, t_eps)
    return text_v, text_x, reg_v, reg_x


def _forward_sample_self_cond(
    model, z, t_batch, x_pred_prev, config,
    self_cond_cfg_scale, cond_seq, cond_seq_mask,
    attention_mask=None, reg_z=None, reg_x_pred_prev=None,
):
    """Forward pass with self-conditioning."""
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob

    def _restore(v, x):
        return restore_vx(v, x, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)

    has_reg = reg_z is not None
    if has_reg != bool(getattr(config, "reg_enabled", False)):
        raise ValueError("REG sampling state does not match config.reg_enabled")

    def _run(text_input, reg_input, scale=None):
        return _flow_model_vx(
            model, text_input, z, t_batch, t_eps,
            reg_input=reg_input, reg_z=reg_z,
            self_cond_cfg_scale=scale, attention_mask=attention_mask,
        )

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        if has_reg and reg_x_pred_prev is None:
            reg_x_pred_prev = torch.zeros_like(reg_z)
        z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
        reg_input_cond = torch.cat([reg_z, reg_x_pred_prev], dim=-1) if has_reg else None
        self_cond_scale_batch = torch.full((z.shape[0],), float(self_cond_cfg_scale),
                                           dtype=z.dtype, device=z.device)
        result = _run(z_input_cond, reg_input_cond, self_cond_scale_batch)
        if not has_reg:
            v_cond, x_cond = result
            return _restore(v_cond, x_cond)
        v_cond, x_cond, reg_v_cond, reg_x_cond = result
        v_cond, x_cond = _restore(v_cond, x_cond)
        return v_cond, x_cond, reg_v_cond, reg_x_cond

    # No self-conditioning
    if self_cond_prob == 0:
        result = _run(z, reg_z)
        if not has_reg:
            v, x = result
            return _restore(v, x)
        v, x, reg_v, reg_x = result
        v, x = _restore(v, x)
        return v, x, reg_v, reg_x

    # Combined unconditional and conditional forward pass
    v_uncond = x_uncond = None
    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        reg_input_uncond = torch.cat([reg_z, torch.zeros_like(reg_z)], dim=-1) if has_reg else None
        result = _run(z_input_uncond, reg_input_uncond)
        if has_reg:
            v_uncond, x_uncond, reg_v_uncond, reg_x_uncond = result
        else:
            v_uncond, x_uncond = result
        v_uncond, x_uncond = _restore(v_uncond, x_uncond)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            if has_reg:
                return v_uncond, x_uncond, reg_v_uncond, reg_x_uncond
            return v_uncond, x_uncond

    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    reg_input_cond = torch.cat([reg_z, reg_x_pred_prev], dim=-1) if has_reg else None
    result = _run(z_input_cond, reg_input_cond)
    if has_reg:
        v_cond, x_cond, reg_v_cond, reg_x_cond = result
    else:
        v_cond, x_cond = result
    v_cond, x_cond = _restore(v_cond, x_cond)
    if self_cond_cfg_scale == 1:
        if has_reg:
            return v_cond, x_cond, reg_v_cond, reg_x_cond
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    v_out, x_out = _restore(v_out, x_out)
    if not has_reg:
        return v_out, x_out
    reg_v_out = reg_v_uncond + self_cond_cfg_scale * (reg_v_cond - reg_v_uncond)
    reg_x_out = reg_x_uncond + self_cond_cfg_scale * (reg_x_cond - reg_x_uncond)
    return v_out, x_out, reg_v_out, reg_x_out


def _forward_sample(
    model, z, t_batch, x_pred_prev, config,
    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask, attention_mask=None,
    reg_z=None, reg_x_pred_prev=None,
):
    """Forward pass with optional self-conditioning and CFG."""
    cond_result = _forward_sample_self_cond(
        model, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask, attention_mask=attention_mask,
        reg_z=reg_z, reg_x_pred_prev=reg_x_pred_prev,
    )
    has_reg = reg_z is not None
    if has_reg:
        v_cond, x_cond, reg_v_cond, reg_x_cond = cond_result
    else:
        v_cond, x_cond = cond_result
    if cfg_scale == 1.0:
        if has_reg:
            return v_cond, x_cond, reg_v_cond, reg_x_cond
        return v_cond, x_cond

    # Unconditional forward: zero out cond prefix, no self-cond state, no restore
    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = (
        None if x_pred_prev is None
        else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
    )
    uncond_result = _forward_sample_self_cond(
        model, z_uncond, t_batch, x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=torch.zeros_like(cond_seq), cond_seq_mask=cond_seq_mask, attention_mask=attention_mask,
        reg_z=reg_z, reg_x_pred_prev=reg_x_pred_prev,
    )
    if has_reg:
        v_uncond, x_uncond, reg_v_uncond, reg_x_uncond = uncond_result
    else:
        v_uncond, x_uncond = uncond_result

    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    v_out, x_out = restore_vx(v_out, x_out, cond_seq, cond_seq_mask)
    if not has_reg:
        return v_out, x_out
    reg_v_out = reg_v_uncond + cfg_scale * (reg_v_cond - reg_v_uncond)
    reg_x_out = reg_x_uncond + cfg_scale * (reg_x_cond - reg_x_uncond)
    return v_out, x_out, reg_v_out, reg_x_out


def _ode_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, attention_mask=None,
    reg_z=None, reg_x_pred_prev=None, velocity_callback=None,
):
    """Single ODE (Euler) step for sampling."""
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)
    result = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        attention_mask=attention_mask,
        reg_z=reg_z, reg_x_pred_prev=reg_x_pred_prev,
    )
    if reg_z is None:
        v_pred, x_pred = result
        if velocity_callback is not None:
            velocity_callback(t, v_pred, None)
        return z + (t_next - t) * v_pred, x_pred
    v_pred, x_pred, reg_v_pred, reg_x_pred = result
    if velocity_callback is not None:
        velocity_callback(t, v_pred, reg_v_pred)
    return (
        z + (t_next - t) * v_pred,
        x_pred,
        reg_z + (t_next - t) * reg_v_pred,
        reg_x_pred,
    )


def _sde_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, gamma, generator, attention_mask=None,
    reg_z=None, reg_x_pred_prev=None,
    text_generators=None, reg_generators=None,
):
    """Per-step SDE-style sampler with hybrid (t-and-step) noise scaling.

    t_back = t * (1 - gamma * h), where h = t_next - t. alpha = 1 - gamma*h is the
    signal-preservation fraction, constant in t. gamma=0 degenerates to a plain ODE step.
    Uniform-N-step equivalence with old multiplicative gamma_old: gamma_hybrid = gamma_old * N.
    """
    h = float(t_next - t)
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * float(t)
    eps = (
        randn_per_example(
            z.shape, text_generators, dtype=z.dtype, device=z.device,
        ) if text_generators is not None else
        torch.randn(z.shape, generator=generator, dtype=z.dtype, device=z.device)
    ) * config.denoiser_noise_scale
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    reg_z_back = None
    if reg_z is not None:
        reg_eps = (
            randn_per_example(
                reg_z.shape, reg_generators, dtype=reg_z.dtype,
                device=reg_z.device,
            ) if reg_generators is not None else
            torch.randn(
                reg_z.shape, generator=generator, dtype=reg_z.dtype,
                device=reg_z.device,
            )
        ) * config.denoiser_noise_scale
        reg_z_back = alpha * reg_z + (1.0 - alpha) * reg_eps
    t_batch = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
    result = _forward_sample(
        model=model, z=z_back, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        attention_mask=attention_mask,
        reg_z=reg_z_back, reg_x_pred_prev=reg_x_pred_prev,
    )
    if reg_z is None:
        v_pred, x_pred = result
        return z_back + (t_next - t_back) * v_pred, x_pred
    v_pred, x_pred, reg_v_pred, reg_x_pred = result
    return (
        z_back + (t_next - t_back) * v_pred,
        x_pred,
        reg_z_back + (t_next - t_back) * reg_v_pred,
        reg_x_pred,
    )
