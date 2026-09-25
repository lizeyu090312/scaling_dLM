"""Sampling and model loading shared by the three headline evaluations."""

import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Subset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modules.model import ELF_models
from modules.text_encoder import get_encoder
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.generation_utils import (
    extract_generated_responses, _dlm_decode_batch, _generate_samples_single_batch,
)
from utils.metrics_utils import (
    compute_gsm8k_accuracy, compute_math_accuracy, parse_gsm8k_answer,
    parse_gsm8k_permissive_answer,
)
from utils.repa_utils import truncate_qwen3_model_for_layers
from utils.sampling_utils import get_sampling_steps, randn_per_example, restore_cond
from utils.teacher_utils import validate_tokenizer_compatibility

EMA = "0.9999"
NUM_STEPS = 64
_STREAM_OFFSETS = {"text": 1_000_000, "reg": 2_000_000}


def load_clean_encoder(config, tokenizer, device):
    encoder_config, encoder = get_encoder(
        config.encoder_model_name,
        torch.float32,
        encoder_dim=config.encoder_dim,
        encoder_layer=config.encoder_layer,
    )
    if encoder_config.family == "qwen3":
        truncate_qwen3_model_for_layers(encoder.model, [encoder_config.encoder_layer])
    encoder_tokenizer = (
        tokenizer
        if (config.tokenizer_name or config.encoder_model_name) == config.encoder_model_name
        else AutoTokenizer.from_pretrained(config.encoder_model_name)
    )
    compatibility = validate_tokenizer_compatibility(
        tokenizer,
        encoder_tokenizer,
        None,
        encoder,
        None,
        dataset_name=config.tokenizer_name or config.encoder_model_name,
        encoder_name=config.encoder_model_name,
        teacher_name=None,
        encoder_family=encoder_config.family,
        teacher_family=None,
    )
    encoder.incompatible_dataset_token_ids = tuple(
        compatibility["dataset_encoder_incompatible_added_token_ids"]
    )
    encoder.token_consumer_name = f"clean encoder {config.encoder_model_name}"
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    return encoder_config, encoder



def _checkpoint_weights(checkpoint, ema: str):
    key = format(float(ema), ".12g")
    states = checkpoint.get("ema_params1")
    if not isinstance(states, dict) or key not in states:
        available = [] if not isinstance(states, dict) else sorted(map(str, states))
        raise ValueError(f"EMA {key} is unavailable; checkpoint contains {available}")
    return states[key]



def load_elf_model(spec, config, tokenizer, device, *, ema: str = EMA):
    checkpoint = torch.load(
        spec.checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    weights = _checkpoint_weights(checkpoint, ema)
    repa_dim = int(config.repa_teacher_dim) if bool(config.repa_enabled) else None
    reg_dim = int(config.reg_teacher_dim) if bool(config.reg_enabled) else None
    model = ELF_models[config.model](
        text_encoder_dim=int(config.encoder_dim),
        max_length=int(config.max_length),
        attn_drop=float(config.attn_dropout),
        proj_drop=float(config.proj_dropout),
        num_time_tokens=int(config.num_time_tokens),
        num_self_cond_cfg_tokens=int(config.num_self_cond_cfg_tokens),
        num_model_mode_tokens=int(config.num_model_mode_tokens),
        vocab_size=len(tokenizer),
        bottleneck_dim=int(config.bottleneck_dim),
        gradient_checkpointing=False,
        repa_target_dim=repa_dim,
        repa_projector_dim=int(config.repa_projector_dim),
        repa_projector_layers=int(config.repa_projector_layers),
        repa_projector_type=config.repa_projector_type,
        repa_adapter_layers=int(config.repa_adapter_layers),
        reg_target_dim=reg_dim,
        reg_projection_topology=config.reg_projection_topology,
        olt_enabled=bool(config.olt_enabled),
    )
    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()
    metadata = {
        "epoch": int(checkpoint["epoch"]),
        "stored_step": int(checkpoint["step"]),
        "optimizer_step": checkpoint.get("optimizer_step"),
        "ema": str(ema),
    }
    del checkpoint, weights
    return model, metadata



def dataloader(dataset, config, tokenizer, *, batch_size: int, start: int = 0):
    indices = list(range(start, len(dataset)))
    return get_dataloader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length,
        distributed=False,
    )



def device_batch(batch, device) -> dict:
    return {
        "input_ids": torch.as_tensor(batch["input_ids"], device=device).long(),
        "encoder_attention_mask": torch.as_tensor(
            batch["encoder_attention_mask"], device=device,
        ).float(),
        "attention_mask": torch.as_tensor(batch["attention_mask"], device=device).float(),
        "cond_seq_mask": torch.as_tensor(batch["cond_seq_mask"], device=device).float(),
        "indices": [int(value) for value in batch["index"]],
        "inputs": [str(value) for value in batch["input"]],
        "targets": [str(value) for value in batch["target"]],
    }



def per_example_noise(indices, shape, *, seed: int, stream: str, device, dtype):
    offset = _STREAM_OFFSETS[stream]
    generators = [
        torch.Generator(device=device).manual_seed(int(seed) + offset + 1009 * int(index))
        for index in indices
    ]
    return randn_per_example(
        (len(indices), *shape), generators, dtype=dtype, device=device,
    )



def sampling_steps(config, *, seed: int, device, dtype, n_steps: int = NUM_STEPS):
    generator = torch.Generator(device=device).manual_seed(int(seed) + 3_000_000)
    return get_sampling_steps(
        n_steps=n_steps,
        time_schedule=config.time_schedule,
        P_mean=config.denoiser_p_mean,
        P_std=config.denoiser_p_std,
        device=device,
        dtype=dtype,
        generator=generator,
    )



def prepare_generation(batch, config, tokenizer, encoder, encoder_config, model, *, seed: int):
    device = next(model.parameters()).device
    values = device_batch(batch, device)
    dtype = next(model.parameters()).dtype
    clean = encode_text(
        values["input_ids"],
        values["encoder_attention_mask"],
        encoder,
        config.latent_mean,
        config.latent_std,
    ).to(dtype)
    cond_mask = values["cond_seq_mask"]
    text_noise = per_example_noise(
        values["indices"],
        (config.max_length, encoder_config.d_model),
        seed=seed,
        stream="text",
        device=device,
        dtype=dtype,
    )
    z = text_noise * config.denoiser_noise_scale
    z = restore_cond(z, clean, cond_mask)
    reg_z = None
    if model.reg_target_dim is not None:
        reg_z = per_example_noise(
            values["indices"],
            (1, model.reg_target_dim),
            seed=seed,
            stream="reg",
            device=device,
            dtype=dtype,
        ) * config.denoiser_noise_scale
    values.update({"clean": clean, "z": z, "reg_z": reg_z})
    return values



def decode_responses(token_ids, cond_mask, tokenizer, config):
    cond_lengths = cond_mask.to(torch.long).sum(dim=1)
    responses, eos, lengths = extract_generated_responses(
        token_ids,
        cond_lengths,
        reserved_generation_length=config.max_length - config.max_input_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        truncate_generation=bool(getattr(config, "truncate_generation", True)),
    )
    texts = [
        tokenizer.decode(row[: int(length)].detach().cpu().tolist(), skip_special_tokens=True)
        for row, length in zip(responses, lengths)
    ]
    return responses, eos, lengths, texts



def gsm_metrics(texts, references):
    return compute_gsm8k_accuracy(list(texts), list(references))



def sample(model, config, values, steps, *, state=None, stop_after=None, callback=None,
           self_cond_cfg_scale=3.0, velocity_callback=None):
    z, x, reg_z, reg_x = state if state is not None else (
        values["z"], None, values["reg_z"], None)
    return _generate_samples_single_batch(
        model=model, generator=None, z=z, t_steps=steps, cond_seq=values["clean"],
        cond_seq_mask=values["cond_seq_mask"], config=config,
        sampling_config=SimpleNamespace(sampling_method="ode"), cfg_scale=1.0,
        self_cond_cfg_scale=self_cond_cfg_scale, reg_z=reg_z, stop_after=stop_after,
        step_callback=callback, x_pred_prev=x, reg_x_pred_prev=reg_x,
        velocity_callback=velocity_callback,
    )



def decode(model, config, tokenizer, values, z, reg_z):
    ids = _dlm_decode_batch(z, model, 1.0, config, 3.0, reg_z=reg_z)
    responses, eos, lengths, texts = decode_responses(ids, values["cond_seq_mask"], tokenizer, config)
    rows = []
    for i, text in enumerate(texts):
        scores = gsm_metrics([text], [values["targets"][i]])
        permissive = parse_gsm8k_permissive_answer(text)
        rows.append(dict(example_id=values["indices"][i], generated_text=text,
                         strict_final_answer=parse_gsm8k_answer(text),
                         permissive_final_answer=None if permissive is None else str(permissive),
                         gsm8k_correct=scores["gsm8k_correct"],
                         gsm8k_permissive_correct=scores["gsm8k_permissive_correct"],
                         response_length=int(lengths[i]), eos_emitted=bool(eos[i])))
    return rows, responses, lengths



def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)



def generate_batch(model, config, tokenizer, values, schedules, requested, *,
                   sample_fn=None, decode_fn=None):
    sample_fn = sample if sample_fn is None else sample_fn
    decode_fn = decode if decode_fn is None else decode_fn
    groups = defaultdict(dict)
    for ratio, nfe in requested:
        groups[ratio * nfe][nfe - 1] = (ratio, nfe)
    rows, timings = [], []
    for master, exits in sorted(groups.items()):
        decoder_seconds = 0.0

        def capture(step, z, x, reg_z, reg_x):
            nonlocal decoder_seconds
            if step not in exits:
                return
            ratio, nfe = exits[step]
            synchronize(z.device)
            start = time.perf_counter()
            decoded, _, _ = decode_fn(model, config, tokenizer, values,
                                   z if ratio == 1 else x, reg_z if ratio == 1 else reg_x)
            synchronize(z.device)
            decoder_seconds += time.perf_counter() - start
            rows.extend(dict(row, ratio=ratio, nfe=nfe, master_nfe=master,
                             denoiser_evaluations=nfe - 1, decoder_evaluations=1)
                        for row in decoded)

        synchronize(values["z"].device)
        start = time.perf_counter()
        sample_fn(model, config, values, schedules[master], stop_after=max(exits), callback=capture)
        synchronize(values["z"].device)
        total = time.perf_counter() - start
        timings.append(dict(record_type="timing", master_nfe=master,
                            actual_denoiser_evaluations=max(exits), decoder_evaluations=len(exits),
                            batch_start=values["indices"][0], sample_count=len(values["indices"]),
                            sampling_seconds=total - decoder_seconds, decoder_seconds=decoder_seconds,
                            total_seconds=total))
    return rows, timings



def decode_math(model, config, tokenizer, values, z, reg_z):
    ids = _dlm_decode_batch(z, model, 1.0, config, 2.0, reg_z=reg_z)
    responses, eos, lengths, texts = decode_responses(ids, values["cond_seq_mask"], tokenizer, config)
    rows = []
    for i, text in enumerate(texts):
        gold = values["gold_answers"][i]
        score = compute_math_accuracy([text], [gold])
        rows.append(dict(example_id=values["indices"][i], generated_text=text, gold_answer=gold,
                         problem_group_id=values["problem_group_ids"][i],
                         math_correct=score["math_correct"], math_invalid=score["math_invalid"],
                         eos_emitted=bool(eos[i]), response_length=int(lengths[i])))
    return rows, responses, lengths



def decode_code(model, config, tokenizer, values, z, reg_z):
    ids = _dlm_decode_batch(z, model, 1.0, config, 2.0, reg_z=reg_z)
    responses, eos, lengths, texts = decode_responses(ids, values["cond_seq_mask"], tokenizer, config)
    rows = [dict(id=values["indices"][i], task_id=values["task_ids"][i], generated=text,
                 eos_emitted=bool(eos[i]), response_length=int(lengths[i]))
            for i, text in enumerate(texts)]
    return rows, responses, lengths

