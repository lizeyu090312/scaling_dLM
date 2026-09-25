import copy
import gc
import itertools
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Subset
from tqdm import tqdm

from configs.config import Config, SamplingConfig
from modules.text_encoder import feature_standardize
from utils.logging_utils import log_for_0
from utils.train_utils import ema_decay_key, is_multi_ema_state, unwrap_model
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.metrics_utils import (
    Metrics as PPLMetrics,
    compute_bleu,
    compute_gsm8k_accuracy,
    compute_math_accuracy,
    compute_mmlu_accuracy,
    compute_mmlu_rationale_accuracy,
    compute_rouge,
)
from utils.sampling_utils import get_sampling_steps, randn_per_example
from utils.generation_utils import (
    build_generation_attention_mask, extract_generated_responses, mask_after_eos,
    _generate_samples_single_batch, _dlm_decode_batch,
    _build_run_name,
)
from utils.generation_resume import (
    append_jsonl_rows, append_metrics_once, cleanup_rank_shards,
    final_output_complete, merge_rank_shards, metrics_has_entry,
    prepare_rank_shard, rank_range, read_generated_rows, shard_path,
    read_generated_records,
    stable_eval_seed,
)

try:
    import wandb
except ImportError:
    wandb = None


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def _validate_evalplus_dataset(config, dataset, num_samples: int):
    if bool(getattr(config, "online_eval", False)):
        raise ValueError("conditional_eval_metric='evalplus' requires online_eval: false")
    if len(dataset) < num_samples:
        raise ValueError(f"eval dataset has {len(dataset)} samples, expected at least {num_samples}")
    for sample_id in range(num_samples):
        item = dataset[sample_id]
        task_id = item.get("task_id") if hasattr(item, "get") else None
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"EvalPlus sample {sample_id} has no nonempty task_id")


def _batch_generator(seed: int, device: torch.device) -> torch.Generator:
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=gen_device).manual_seed(int(seed))


EVAL_RNG_PROTOCOL = "per_example_v1"


def _per_example_generators(eval_seed, rank, sample_ids, stream, device):
    return [
        _batch_generator(
            stable_eval_seed(
                EVAL_RNG_PROTOCOL, int(eval_seed), int(rank), int(sample_id), stream,
            ),
            device,
        )
        for sample_id in sample_ids
    ]


def _sampling_schedule_generator(eval_seed, rank, device):
    return _batch_generator(
        stable_eval_seed(EVAL_RNG_PROTOCOL, int(eval_seed), int(rank), "schedule"),
        device,
    )


def _run_metadata(
    config: Config,
    sampling_config: SamplingConfig,
    *,
    mode: str,
    run_name: str,
    epoch: int,
    step: int,
    num_samples: int,
    batch_size: int,
    eval_seed: int,
    num_sampling_steps: int,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    ema_key,
    eval_data_path=None,
    evaluation_name=None,
):
    metadata = {
        "version": 2,
        "eval_rng_protocol": EVAL_RNG_PROTOCOL,
        "mode": mode,
        "run_name": run_name,
        "epoch": int(epoch),
        "step": int(step),
        "num_samples": int(num_samples),
        "batch_size": int(batch_size),
        "world": int(_world()),
        "seed": int(eval_seed),
        "sampling_method": sampling_config.sampling_method,
        "num_sampling_steps": int(num_sampling_steps),
        "cfg_scale": float(cfg_scale),
        "self_cond_cfg_scale": float(self_cond_cfg_scale),
        "time_schedule": sampling_config.time_schedule,
        "sde_gamma": float(getattr(sampling_config, "sde_gamma", 0.0)),
        "ema": 0.0 if ema_key is None else float(ema_key),
        "ema_label": ema_variant_label(ema_key),
        "max_length": int(config.max_length),
        "max_input_length": config.max_input_length,
        "denoiser_p_mean": float(config.denoiser_p_mean),
        "denoiser_p_std": float(config.denoiser_p_std),
        "denoiser_noise_scale": float(config.denoiser_noise_scale),
        "use_model_attention_mask": bool(getattr(config, "use_model_attention_mask", False)),
        "reg_enabled": bool(getattr(config, "reg_enabled", False)),
        "reg_teacher_dim": getattr(config, "reg_teacher_dim", None),
        "reg_teacher_pooling": getattr(config, "reg_teacher_pooling", None),
        "reg_projection_topology": getattr(config, "reg_projection_topology", None),
        "olt_enabled": bool(getattr(config, "olt_enabled", False)),
        "conditional_eval_metric": getattr(config, "conditional_eval_metric", None),
        "eval_data_path": (
            config.eval_data_path if eval_data_path is None else eval_data_path
        ),
    }
    if evaluation_name is not None:
        metadata["evaluation_name"] = evaluation_name
    if not bool(getattr(config, "truncate_generation", True)):
        metadata["truncate_generation"] = False
    return metadata


def _shard_metadata(metadata, rank: int, start_id: int, end_id: int):
    shard_metadata = dict(metadata)
    shard_metadata.update({
        "rank": int(rank),
        "rank_start": int(start_id),
        "rank_end": int(end_id),
    })
    return shard_metadata


def _load_cond_text(dataset, sample_id: int):
    item = dataset[int(sample_id)]
    return item["target"], item["input"]


def _load_math_reference(dataset, sample_id: int):
    return dataset[int(sample_id)]["gold_answer"]


def _load_mmlu_rationale_reference(dataset, sample_id: int):
    return dataset[int(sample_id)]["gold_answer"]


def ema_variant_label(ema_key) -> str:
    if ema_key is None:
        return "raw"
    return "ema" + str(ema_key).replace(".", "p")


def parse_eval_ema_arg(value):
    if value is None:
        return None
    keys = []
    for item in value.split(","):
        decay = float(item.strip())
        keys.append(None if decay == 0.0 else ema_decay_key(decay))
    return keys


def resolve_eval_ema_keys(state, requested=None):
    available = list(state.ema_params1.keys()) if is_multi_ema_state(state.ema_params1) else []
    if requested is None:
        return [None] + available
    for key in requested:
        if key is not None and key not in available:
            raise ValueError(f"requested EMA {key} not found in checkpoint; available EMAs: {available}")
    return requested


def _build_eval_model(state, use_compile: bool = False, ema_key=None) -> nn.Module:
    """Return an eval-mode model copy, using raw params or a selected EMA."""
    model = unwrap_model(state.model)
    eval_model = copy.deepcopy(model)
    if ema_key is not None:
        eval_model.load_state_dict(state.ema_params1[ema_key])
    eval_model.eval()
    if use_compile:
        log_for_0("Compiling eval model with torch.compile (first batch will be slower)...")
        eval_model = torch.compile(eval_model)
    return eval_model


# ============================================
# Generation Helper
# ============================================
def run_generation(
    state,
    encoder: nn.Module,
    eval_dataset,
    tokenizer,
    config,
    generator: torch.Generator,
    generation_batch_size: int,
    eval_seed: int = None,
    ema_keys=None,
    num_samples: int = None,
    eval_data_path: str = None,
    evaluation_name: str = None,
):
    """Run test generation."""
    if isinstance(config.eval_data_path, list) and evaluation_name is None:
        if getattr(config, "conditional_eval_metric", None) == "evalplus":
            for spec, dataset in zip(config.eval_data_path, eval_dataset):
                _validate_evalplus_dataset(config, dataset, spec["num_samples"])
        for spec, dataset in zip(config.eval_data_path, eval_dataset):
            run_generation(
                state=state, encoder=encoder, eval_dataset=dataset,
                tokenizer=tokenizer, config=config, generator=generator,
                generation_batch_size=generation_batch_size,
                eval_seed=eval_seed, ema_keys=ema_keys,
                num_samples=spec["num_samples"],
                eval_data_path=spec["path"], evaluation_name=spec["name"],
            )
        return

    if evaluation_name is not None:
        log_for_0(f"\n=== Evaluation dataset: {evaluation_name} ===")
    for ema_key in resolve_eval_ema_keys(state, ema_keys):
        log_for_0(f"\n--- Model variant: {ema_variant_label(ema_key)} ---")
        for sc_idx, sc in enumerate(config.sampling_configs):
            if len(config.sampling_configs) > 1:
                log_for_0(f"\n--- Sampling config {sc_idx + 1}/{len(config.sampling_configs)} ---")
            common_kwargs = dict(
                state=state,
                tokenizer=tokenizer,
                generator=generator,
                config=config,
                sampling_config=sc,
                batch_size=generation_batch_size,
                num_samples=config.num_samples if num_samples is None else num_samples,
                eval_seed=config.seed if eval_seed is None else eval_seed,
                ema_key=ema_key,
            )
            if eval_dataset is None:
                test_generation_uncond(**common_kwargs)
            else:
                test_generation_cond(
                    **common_kwargs, encoder=encoder, dataset=eval_dataset,
                    eval_data_path=eval_data_path,
                    evaluation_name=evaluation_name,
                )


# ============================================
# Unconditional generation
# ============================================
def test_generation_uncond(
    state,
    tokenizer,
    generator: torch.Generator,
    config: Config,
    sampling_config: SamplingConfig,
    num_samples: int = 64,
    batch_size: int = 64,
    eval_seed: int = None,
    ema_key=None,
):
    """Test unconditional generation."""
    eval_seed = int(config.seed if eval_seed is None else eval_seed)
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule
    log_for_0(f"Config: {sampling_config}")

    log_for_0("\n" + "=" * 70)
    log_for_0("              UNCONDITIONAL GENERATION EXAMPLES")
    log_for_0("=" * 70)

    log_for_0(f"Model variant: {ema_variant_label(ema_key)}")
    model = _build_eval_model(state, use_compile=bool(getattr(config, "use_compile", False)), ema_key=ema_key)
    device = next(model.parameters()).device
    d_model = model.text_encoder_dim
    log_for_0(f"Per-device batch size: {batch_size}")

    pad_token_id = get_pad_token_id(tokenizer)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    cfg_list = [1]
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales
    wandb_tables = {}
    ppl_metrics = None

    world = _world()
    rank = _rank()
    param_dtype = next(model.parameters()).dtype
    epoch_val = int(state.epoch)
    step_val = int(state.step)

    for num_sampling_steps, cfg_scale, self_cond_cfg_scale in itertools.product(
        steps_list, cfg_list, self_cond_cfg_scales_list
    ):
        log_for_0(f"\n--- Method: {sampling_method}, Steps: {num_sampling_steps}, "
                  f"CFG Scale: {cfg_scale}, SC-CFG: {self_cond_cfg_scale} ---")

        name = _build_run_name(
            sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0),
            suffix=f"uncond_{ema_variant_label(ema_key)}",
        )
        run_dir = os.path.join(config.output_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, f"all_generated_{epoch_val}_{step_val}.jsonl")
        metadata = _run_metadata(
            config, sampling_config, mode="uncond", run_name=name,
            epoch=epoch_val, step=step_val, num_samples=num_samples,
            batch_size=batch_size, eval_seed=eval_seed,
            num_sampling_steps=num_sampling_steps, cfg_scale=cfg_scale,
            self_cond_cfg_scale=self_cond_cfg_scale, ema_key=ema_key,
        )

        generation_time = 0.0
        decode_time = 0.0
        created_final = False
        already_complete = final_output_complete(out_path, metadata, num_samples)

        if already_complete:
            log_for_0(f"Skipping completed generation: {out_path}")
            if rank == 0:
                cleanup_rank_shards(out_path, world)
        else:
            start_id, end_id = rank_range(num_samples, world, rank)
            if start_id < end_id:
                shard = shard_path(out_path, rank)
                resume_id = prepare_rank_shard(
                    shard, _shard_metadata(metadata, rank, start_id, end_id),
                    start_id, end_id, batch_size,
                )
                remaining_batches = (end_id - resume_id + batch_size - 1) // batch_size
                if resume_id > start_id:
                    log_for_0(f"Rank {rank}: resuming {name} at sample {resume_id}")
                pbar = tqdm(
                    range(resume_id, end_id, batch_size),
                    total=remaining_batches,
                    desc="Generating samples", disable=(rank != 0),
                )
                with open(shard, "a", encoding="utf-8") as shard_f:
                    for batch_start in pbar:
                        current_batch = min(batch_size, end_id - batch_start)
                        sample_ids = range(batch_start, batch_start + current_batch)
                        batch_gen = _sampling_schedule_generator(eval_seed, rank, device)
                        t_steps = get_sampling_steps(
                            n_steps=num_sampling_steps,
                            time_schedule=time_schedule,
                            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                            device=device, dtype=param_dtype, generator=batch_gen,
                        )
                        z = randn_per_example(
                            (current_batch, config.max_length, d_model),
                            _per_example_generators(
                                eval_seed, rank, sample_ids, "text_initial", device,
                            ),
                            dtype=param_dtype, device=device,
                        ) * config.denoiser_noise_scale
                        reg_z = None
                        if model.reg_target_dim is not None:
                            reg_z = randn_per_example(
                                (current_batch, 1, model.reg_target_dim),
                                _per_example_generators(
                                    eval_seed, rank, sample_ids, "reg_initial", device,
                                ),
                                dtype=param_dtype, device=device,
                            ) * config.denoiser_noise_scale
                        text_sde_generators = _per_example_generators(
                            eval_seed, rank, sample_ids, "text_sde", device,
                        )
                        reg_sde_generators = (
                            _per_example_generators(
                                eval_seed, rank, sample_ids, "reg_sde", device,
                            ) if reg_z is not None else None
                        )

                        gen_start = time.time()
                        generated = _generate_samples_single_batch(
                            model=model, generator=batch_gen, z=z, t_steps=t_steps,
                            cond_seq=None, cond_seq_mask=None,
                            config=config, sampling_config=sampling_config,
                            cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
                            reg_z=reg_z,
                            text_sde_generators=text_sde_generators,
                            reg_sde_generators=reg_sde_generators,
                        )
                        if reg_z is None:
                            latent, reg_latent = generated, None
                        else:
                            latent, reg_latent = generated
                        generation_time += time.time() - gen_start

                        dec_start = time.time()
                        t_final_val = t_steps[-1].item()
                        predicted_ids = _dlm_decode_batch(
                            z=latent, model=model, t_final_val=t_final_val,
                            config=config, self_cond_cfg_scale=self_cond_cfg_scale,
                            reg_z=reg_latent,
                        )
                        decode_time += time.time() - dec_start

                        predicted_ids = mask_after_eos(
                            predicted_ids, eos_token_id=eos_token_id,
                            pad_token_id=pad_token_id,
                        )

                        rows = []
                        for i in range(predicted_ids.shape[0]):
                            sample_id = batch_start + i
                            text = tokenizer.decode(
                                predicted_ids[i].detach().cpu().numpy(),
                                skip_special_tokens=True,
                            )
                            rows.append({"id": sample_id, "generated": text})
                        append_jsonl_rows(shard_f, rows)
                        del z, reg_z, latent, reg_latent, predicted_ids, t_steps, batch_gen
                pbar.close()

            _barrier()
            if rank == 0:
                merge_rank_shards(out_path, metadata, num_samples, world)
                created_final = True
                cleanup_rank_shards(out_path, world)
                log_for_0(f"Saved {num_samples} generated texts to {out_path}")
            _barrier()

        if not already_complete:
            log_for_0(f"Generation: {generation_time:.2f}s ({num_sampling_steps} steps) | Decode: {decode_time:.2f}s")
            log_for_0("-" * 70)

        ppl_results = None
        if rank == 0:
            all_generated = read_generated_rows(out_path)
            metrics_path = os.path.join(run_dir, "metrics.jsonl")
            if config.online_eval:
                if metrics_has_entry(metrics_path, epoch_val, step_val):
                    log_for_0(f"Skipping existing metrics: {metrics_path}")
                else:
                    if ppl_metrics is None:
                        ppl_metrics = PPLMetrics(
                            gen_ppl_eval_model_name_or_path=config.eval_ppl_model,
                            eval_ppl_batch_size=config.eval_ppl_batch_size,
                            eval_context_size=config.eval_ppl_max_length,
                        )
                    log_for_0("\n" + "=" * 70)
                    log_for_0("              PPL EVALUATION")
                    log_for_0("=" * 70)
                    ppl_metrics.reset()
                    text_samples = [gen for _, gen in all_generated]
                    nonempty_samples = [s for s in text_samples if isinstance(s, str) and s.strip()]
                    skipped = len(text_samples) - len(nonempty_samples)
                    if skipped > 0:
                        log_for_0(f"PPL eval: skipped {skipped} empty samples")
                    if not nonempty_samples:
                        log_for_0("PPL eval: all samples empty; skipping perplexity computation")
                    else:
                        ppl_results = ppl_metrics.record_generative_perplexity(
                            text_samples=nonempty_samples,
                            max_length=config.eval_ppl_max_length,
                            retokenize=True,
                        )
                        log_for_0(f"Perplexity: {ppl_results['ppl']:.4f}")
                        log_for_0(f"Mean Entropy: {ppl_results['mean_entropy']:.4f}")
                    log_for_0("=" * 70 + "\n")

            if ppl_results is not None:
                metrics_line = {
                    "epoch": epoch_val, "step": step_val,
                    "ppl": ppl_results["ppl"], "mean_entropy": ppl_results["mean_entropy"],
                }
                append_metrics_once(metrics_path, metrics_line)

            if config.use_wandb and wandb is not None and (created_final or ppl_results is not None):
                table = wandb.Table(columns=["sample_id", "text"])
                for tid, gen in all_generated[:min(10, len(all_generated))]:
                    table.add_data(tid, gen)
                wandb_tables[f"generated_samples_uncond_steps{num_sampling_steps}_cfg{cfg_scale}"] = table
                if ppl_results is not None:
                    wandb_tables.update({
                        f"generation/{name}/ppl": ppl_results["ppl"],
                        f"generation/{name}/mean_entropy": ppl_results["mean_entropy"],
                    })
        _barrier()

    if _rank() == 0 and config.use_wandb and wandb_tables and wandb is not None:
        try:
            wandb.log(wandb_tables)
        except Exception as e:
            log_for_0(f"Warning: wandb.log failed: {e}")

    model.to("cpu")
    if ppl_metrics is not None and ppl_metrics._eval_model is not None:
        ppl_metrics._eval_model.to("cpu")
    del model, ppl_metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_for_0("=" * 70 + "\n")


# ============================================
# Conditional generation
# ============================================
def test_generation_cond(
    state,
    encoder: nn.Module,
    tokenizer,
    generator: torch.Generator,
    config: Config,
    sampling_config: SamplingConfig,
    dataset,
    num_samples: int = 64,
    batch_size: int = 64,
    eval_seed: int = None,
    ema_key=None,
    eval_data_path: str = None,
    evaluation_name: str = None,
):
    """Test conditional generation."""
    is_evalplus = getattr(config, "conditional_eval_metric", None) == "evalplus"
    if is_evalplus:
        _validate_evalplus_dataset(config, dataset, num_samples)
    elif len(dataset) < num_samples:
        raise ValueError(f"eval dataset has {len(dataset)} samples, expected at least {num_samples}")

    eval_seed = int(config.seed if eval_seed is None else eval_seed)
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule
    log_for_0(f"Config: {sampling_config}")

    log_for_0("\n" + "=" * 70)
    log_for_0("              CONDITIONAL GENERATION EXAMPLES")
    log_for_0("=" * 70)

    log_for_0(f"Model variant: {ema_variant_label(ema_key)}")
    model = _build_eval_model(state, use_compile=bool(getattr(config, "use_compile", False)), ema_key=ema_key)
    device = next(model.parameters()).device
    d_model = model.text_encoder_dim

    encode_latent_mean, encode_latent_std = config.latent_mean, config.latent_std
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id

    wandb_tables = {}
    cfg_list = sampling_config.cfgs
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales
    world = _world()
    rank = _rank()
    param_dtype = next(model.parameters()).dtype
    epoch_val = int(state.epoch)
    step_val = int(state.step)

    for num_sampling_steps, cfg_scale, self_cond_cfg_scale in itertools.product(
        steps_list, cfg_list, self_cond_cfg_scales_list
    ):
        log_for_0(f"\n--- Steps: {num_sampling_steps}, CFG Scale: {cfg_scale}, "
                  f"SC-CFG: {self_cond_cfg_scale} ---")

        suffix = f"cond_{ema_variant_label(ema_key)}"
        if evaluation_name is not None:
            suffix = f"{suffix}_{evaluation_name}"
        name = _build_run_name(
            sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0),
            suffix=suffix,
        )
        run_dir = os.path.join(config.output_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, f"all_generated_{epoch_val}_{step_val}.jsonl")
        metadata = _run_metadata(
            config, sampling_config, mode="cond", run_name=name,
            epoch=epoch_val, step=step_val, num_samples=num_samples,
            batch_size=batch_size, eval_seed=eval_seed,
            num_sampling_steps=num_sampling_steps, cfg_scale=cfg_scale,
            self_cond_cfg_scale=self_cond_cfg_scale, ema_key=ema_key,
            eval_data_path=eval_data_path, evaluation_name=evaluation_name,
        )

        generation_time = 0.0
        decode_time = 0.0
        created_final = False
        already_complete = final_output_complete(out_path, metadata, num_samples)

        if already_complete:
            log_for_0(f"Skipping completed generation: {out_path}")
            if rank == 0:
                cleanup_rank_shards(out_path, world)
        else:
            start_id, end_id = rank_range(num_samples, world, rank)
            if start_id < end_id:
                shard = shard_path(out_path, rank)
                resume_id = prepare_rank_shard(
                    shard, _shard_metadata(metadata, rank, start_id, end_id),
                    start_id, end_id, batch_size,
                )
                remaining_batches = (end_id - resume_id + batch_size - 1) // batch_size
                if resume_id > start_id:
                    log_for_0(f"Rank {rank}: resuming {name} at sample {resume_id}")
                subset = Subset(dataset, range(resume_id, end_id))
                dataloader = get_dataloader(
                    subset, batch_size=batch_size,
                    shuffle=False, num_workers=0, drop_last=False,
                    max_seq_length=config.max_length, pad_token_id=pad_token_id,
                    max_input_seq_length=config.max_input_length, distributed=False,
                )
                pbar = tqdm(
                    total=remaining_batches,
                    desc="Generating samples (cond)", disable=(rank != 0),
                )
                batch_start = resume_id
                with open(shard, "a", encoding="utf-8") as shard_f:
                    for batch in dataloader:
                        bsz = batch["input_ids"].shape[0]
                        input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
                        encoder_attention_mask = torch.from_numpy(
                            np.array(batch["encoder_attention_mask"])
                        ).to(device).float()
                        cond_seq_mask_arr = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(device).float()

                        sample_ids = range(batch_start, batch_start + bsz)
                        batch_gen = _sampling_schedule_generator(eval_seed, rank, device)
                        t_steps = get_sampling_steps(
                            n_steps=num_sampling_steps,
                            time_schedule=time_schedule,
                            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                            device=device, dtype=param_dtype, generator=batch_gen,
                        )

                        if "cached_text_latents" in batch:
                            cond_seq = torch.from_numpy(
                                np.asarray(batch["cached_text_latents"]),
                            ).to(device=device, dtype=param_dtype)
                            cond_seq = feature_standardize(cond_seq)
                        else:
                            cond_seq = encode_text(
                                input_ids=input_ids, attention_mask=encoder_attention_mask,
                                encoder=encoder, latent_mean=encode_latent_mean,
                                latent_std=encode_latent_std,
                            ).to(param_dtype)

                        z = randn_per_example(
                            (bsz, config.max_length, d_model),
                            _per_example_generators(
                                eval_seed, rank, sample_ids, "text_initial", device,
                            ),
                            dtype=param_dtype, device=device,
                        ) * config.denoiser_noise_scale
                        reg_z = None
                        if model.reg_target_dim is not None:
                            reg_z = randn_per_example(
                                (bsz, 1, model.reg_target_dim),
                                _per_example_generators(
                                    eval_seed, rank, sample_ids, "reg_initial", device,
                                ),
                                dtype=param_dtype, device=device,
                            ) * config.denoiser_noise_scale
                        text_sde_generators = _per_example_generators(
                            eval_seed, rank, sample_ids, "text_sde", device,
                        )
                        reg_sde_generators = (
                            _per_example_generators(
                                eval_seed, rank, sample_ids, "reg_sde", device,
                            ) if reg_z is not None else None
                        )
                        gen_length = config.max_length - config.max_input_length
                        truncate_generation = bool(getattr(config, "truncate_generation", True))
                        cond_len_per_sample = cond_seq_mask_arr.to(torch.int32).sum(dim=1)
                        model_attention_mask = None
                        if bool(getattr(config, "use_model_attention_mask", False)):
                            model_attention_mask = build_generation_attention_mask(
                                cond_len_per_sample, max_length=config.max_length,
                                gen_length=gen_length if truncate_generation else None,
                            ).to(device=device, dtype=cond_seq_mask_arr.dtype)

                        gen_start = time.time()
                        generated = _generate_samples_single_batch(
                            model=model, generator=batch_gen, z=z, t_steps=t_steps,
                            cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_arr,
                            config=config, sampling_config=sampling_config,
                            cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
                            attention_mask=model_attention_mask,
                            reg_z=reg_z,
                            text_sde_generators=text_sde_generators,
                            reg_sde_generators=reg_sde_generators,
                        )
                        if reg_z is None:
                            latent, reg_latent = generated, None
                        else:
                            latent, reg_latent = generated
                        generation_time += time.time() - gen_start

                        dec_start = time.time()
                        t_final_val = t_steps[-1].item()
                        predicted_ids = _dlm_decode_batch(
                            z=latent, model=model, t_final_val=t_final_val,
                            config=config, self_cond_cfg_scale=self_cond_cfg_scale,
                            attention_mask=model_attention_mask,
                            reg_z=reg_latent,
                        )
                        predicted_ids, eos_emitted, response_lengths = (
                            extract_generated_responses(
                                predicted_ids, cond_len_per_sample, reserved_generation_length=gen_length,
                                eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                                truncate_generation=truncate_generation,
                            )
                        )
                        decode_time += time.time() - dec_start

                        rows = []
                        for i in range(bsz):
                            sample_id = batch_start + i
                            text = tokenizer.decode(
                                predicted_ids[i].detach().cpu().numpy(),
                                skip_special_tokens=True,
                            )
                            row = {
                                "id": sample_id,
                                "generated": text,
                                "eos_emitted": bool(eos_emitted[i].item()),
                                "response_length": int(response_lengths[i].item()),
                            }
                            if is_evalplus:
                                row["task_id"] = batch["task_id"][i]
                            rows.append(row)
                        append_jsonl_rows(shard_f, rows)
                        batch_start += bsz
                        pbar.update(1)
                        del (
                            input_ids, encoder_attention_mask, cond_seq_mask_arr,
                            cond_seq, z, reg_z, latent, reg_latent, cond_len_per_sample,
                            predicted_ids, t_steps, batch_gen, model_attention_mask,
                        )
                pbar.close()

            _barrier()
            if rank == 0:
                merge_rank_shards(out_path, metadata, num_samples, world)
                created_final = True
                cleanup_rank_shards(out_path, world)
                log_for_0(f"Saved {num_samples} generated texts to {out_path}")
            _barrier()

        if not already_complete:
            log_for_0(f"Generation: {generation_time:.2f}s ({num_sampling_steps} steps) | Decode: {decode_time:.2f}s")
            log_for_0("-" * 70)

        cond_eval_results = None
        if rank == 0:
            all_generated = read_generated_rows(out_path)
            metrics_path = os.path.join(run_dir, "metrics.jsonl")
            metric = getattr(config, "conditional_eval_metric", "bleu_rouge")
            if config.online_eval:
                if metrics_has_entry(
                    metrics_path, epoch_val, step_val, metric=metric,
                ):
                    log_for_0(f"Skipping existing metrics: {metrics_path}")
                elif all_generated:
                    hypotheses = [gen for _, gen in all_generated]
                    if metric == "bleu_rouge":
                        references = [_load_cond_text(dataset, tid)[0] for tid, _ in all_generated]
                        bleu_score = compute_bleu(hypotheses, references)
                        rouge_scores = compute_rouge(hypotheses, references)
                        cond_eval_results = {"bleu": bleu_score, **rouge_scores}
                        log_for_0(
                            f"BLEU: {bleu_score:.2f}  ROUGE-1: {rouge_scores['rouge1']:.2f}  "
                            f"ROUGE-2: {rouge_scores['rouge2']:.2f}  ROUGE-L: {rouge_scores['rougeL']:.2f}"
                        )
                    elif metric == "mmlu_accuracy":
                        references = [_load_cond_text(dataset, tid)[0] for tid, _ in all_generated]
                        cond_eval_results = compute_mmlu_accuracy(hypotheses, references)
                        log_for_0(
                            f"MMLU accuracy: {cond_eval_results['mmlu_accuracy']:.2f} "
                            f"({cond_eval_results['mmlu_correct']}/{cond_eval_results['mmlu_total']}, "
                            f"invalid={cond_eval_results['mmlu_invalid']})"
                        )
                    elif metric == "mmlu_rationale_accuracy":
                        references = [
                            _load_mmlu_rationale_reference(dataset, tid)
                            for tid, _ in all_generated
                        ]
                        cond_eval_results = compute_mmlu_rationale_accuracy(
                            hypotheses, references,
                        )
                        log_for_0(
                            "Rationale-MMLU exact: "
                            f"{cond_eval_results['mmlu_rationale_exact_accuracy']:.2f} "
                            f"({cond_eval_results['mmlu_rationale_exact_correct']}/"
                            f"{cond_eval_results['mmlu_rationale_exact_total']}, "
                            f"invalid={cond_eval_results['mmlu_rationale_exact_invalid']}) | "
                            "permissive: "
                            f"{cond_eval_results['mmlu_rationale_permissive_accuracy']:.2f} "
                            f"({cond_eval_results['mmlu_rationale_permissive_correct']}/"
                            f"{cond_eval_results['mmlu_rationale_permissive_total']}, "
                            f"invalid={cond_eval_results['mmlu_rationale_permissive_invalid']})"
                        )
                    elif metric == "gsm8k_accuracy":
                        references = [_load_cond_text(dataset, tid)[0] for tid, _ in all_generated]
                        records = read_generated_records(out_path)
                        cond_eval_results = compute_gsm8k_accuracy(
                            hypotheses, references,
                            eos_emitted=[row["eos_emitted"] for row in records],
                            response_lengths=[row["response_length"] for row in records],
                        )
                        log_for_0(
                            f"GSM8K strict: {cond_eval_results['gsm8k_accuracy']:.2f} "
                            f"({cond_eval_results['gsm8k_correct']}/{cond_eval_results['gsm8k_total']}, "
                            f"invalid={cond_eval_results['gsm8k_invalid']}) | "
                            f"permissive: {cond_eval_results['gsm8k_permissive_accuracy']:.2f} "
                            f"({cond_eval_results['gsm8k_permissive_correct']}/"
                            f"{cond_eval_results['gsm8k_permissive_total']}, "
                            f"invalid={cond_eval_results['gsm8k_permissive_invalid']}) | "
                            f"EOS={cond_eval_results['gsm8k_eos_rate']:.2f}%"
                        )
                    elif metric == "math_accuracy":
                        records = read_generated_records(out_path)
                        math_references = [
                            _load_math_reference(dataset, tid)
                            for tid, _ in all_generated
                        ]
                        cond_eval_results = compute_math_accuracy(
                            hypotheses, math_references,
                            eos_emitted=[row["eos_emitted"] for row in records],
                            response_lengths=[row["response_length"] for row in records],
                        )
                        log_for_0(
                            f"MATH: {cond_eval_results['math_accuracy']:.2f} "
                            f"({cond_eval_results['math_correct']}/"
                            f"{cond_eval_results['math_total']}, "
                            f"invalid={cond_eval_results['math_invalid']}) | "
                            f"EOS={cond_eval_results['math_eos_rate']:.2f}%"
                        )
                    else:
                        raise ValueError(f"Unknown conditional_eval_metric: {metric}")

            if config.use_wandb and wandb is not None and (created_final or cond_eval_results is not None):
                table = wandb.Table(columns=["sample_id", "context", "original", "generated"])
                for tid, gen in all_generated[:min(10, len(all_generated))]:
                    orig, ctx = _load_cond_text(dataset, tid)
                    table.add_data(tid, ctx, orig, gen)
                table_key = f"generated_samples_cond_steps{num_sampling_steps}_cfg{cfg_scale}"
                if evaluation_name is not None:
                    table_key = f"{evaluation_name}/{table_key}"
                wandb_tables[table_key] = table
                if cond_eval_results is not None:
                    wandb_tables.update({
                        f"generation/{name}/{key}": value for key, value in cond_eval_results.items()
                    })
            if cond_eval_results is not None:
                metrics_line = {
                    "epoch": epoch_val, "step": step_val, "metric": metric,
                    **cond_eval_results,
                }
                append_metrics_once(metrics_path, metrics_line)
        _barrier()

    if _rank() == 0 and config.use_wandb and wandb_tables and wandb is not None:
        try:
            wandb.log(wandb_tables)
        except Exception as e:
            log_for_0(f"Warning: wandb.log failed: {e}")

    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_for_0("=" * 70 + "\n")
