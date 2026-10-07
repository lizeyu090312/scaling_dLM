"""Fixed p=0.5, eta=1.2 clean-prediction re-noising for headline evaluation."""

import torch

from utils.sampling_utils import _forward_sample, get_sampling_steps, randn_per_example, restore_cond


def schedule(config, seed, nfe, device, dtype):
    if nfe < 3:
        raise ValueError("Clean-prediction re-noising requires NFE >= 3")
    generator = torch.Generator(device=device).manual_seed(int(seed) + 3_000_000)
    master = get_sampling_steps(
        n_steps=8*nfe-1, time_schedule=config.time_schedule,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        device=device, dtype=dtype, generator=generator)
    end = float(master[nfe-2])
    nodes = end * torch.linspace(0, 1, nfe-1, dtype=torch.float64).pow(0.5)
    nodes[0], nodes[-1] = 0., end
    nodes = nodes.to(device=master.device, dtype=master.dtype)
    if not bool(torch.all(nodes[1:] > nodes[:-1])):
        raise ValueError("evaluation times must be strictly increasing")
    return nodes


def _draw_noise(state, generators):
    return randn_per_example(state.shape, generators, dtype=state.dtype, device=state.device)


@torch.no_grad()
def sample(model, config, values, times, *, seed, self_cond_cfg):
    z, reg_z = values["z"], values["reg_z"]
    generators = [[torch.Generator(device=z.device).manual_seed(seed+offset+1009*int(i))
                   for i in values["indices"]] for offset in (4_000_000, 5_000_000)]
    nodes = [float(t) for t in times]
    x = reg_x = None
    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                            enabled=z.device.type == "cuda" and config.use_bf16):
        for i, t in enumerate(nodes):
            result = _forward_sample(
                model=model, z=z, t_batch=torch.full((len(z),), t, device=z.device, dtype=z.dtype),
                x_pred_prev=x, config=config, cfg_scale=1., self_cond_cfg_scale=self_cond_cfg,
                cond_seq=values["clean"], cond_seq_mask=values["cond_seq_mask"], attention_mask=None,
                reg_z=reg_z, reg_x_pred_prev=reg_x)
            if reg_z is None:
                _, x = result
            else:
                _, x, _, reg_x = result
            if i == len(nodes)-1:
                break
            next_t = nodes[i+1]
            z = next_t*x + (1-next_t)*1.2*config.denoiser_noise_scale*_draw_noise(z, generators[0])
            if reg_z is not None:
                reg_z = next_t*reg_x + (1-next_t)*1.2*config.denoiser_noise_scale*_draw_noise(reg_z, generators[1])
            z = restore_cond(z, values["clean"], values["cond_seq_mask"])
    if not bool(torch.isfinite(x).all()) or (reg_x is not None and not bool(torch.isfinite(reg_x).all())):
        raise FloatingPointError("nonfinite terminal prediction")
    return x, reg_x
