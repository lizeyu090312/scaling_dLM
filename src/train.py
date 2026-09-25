#!/usr/bin/env python
"""Training script for the ELF."""

import argparse
import logging
import math
import os
import sys
import time

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

from modules.text_encoder import get_encoder
from utils.logging_utils import log_for_0, add_file_logging
from utils.checkpoint_utils import (
    AUTORESUME_CHECKPOINT_NAME, save_checkpoint, save_autoresume_checkpoint,
    load_checkpoint, load_weights_only_checkpoint, find_all_checkpoints,
    find_latest_checkpoint, infer_reg_projection_config, load_checkpoint_optimizer_step,
)
from utils.train_utils import (
    TrainState, prefetch_to_device, get_optimizer, create_learning_rate_fn,
    attach_lr_scheduler, stable_train_seed, ema_decay_key,
)
from utils.repa_utils import (
    REPA_ADAPTER_ATTENTION_MODES,
    reg_target_dim, repa_target_dim, same_repa_teacher, same_repa_teacher_model,
    truncate_qwen3_model_for_layers, validate_external_repa_teacher,
    validate_repa_shift,
)
from utils.teacher_utils import (
    TEACHER_INPUT_FORMATS, TeacherTargetProvider, validate_tokenizer_compatibility,
)
from generation import parse_eval_ema_arg, run_generation
from configs.config import (
    SamplingConfig, apply_config_overrides, config_field_is_explicit,
    load_config_from_yaml, load_sampling_configs, normalize_ema_decay1,
)
from modules.model import ELF_models
from utils.data_utils import get_dataloader, prepare_batch, load_dataset, get_pad_token_id
from utils.latent_cache_utils import attach_latent_caches
from train_step import train_step

try:
    import wandb
except ImportError:
    wandb = None

# Logging: no timestamps; suppress noisy checkpoint loggers; unbuffered stdout
logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)

AUTORESUME_SAVE_STEPS_BASE = int(os.environ.get("AUTORESUME_SAVE_STEPS_BASE", 1000))


def _init_distributed():
    """Initialize torch.distributed if launched via torchrun."""
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            from datetime import timedelta
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30),)
        else:
            dist.init_process_group(backend="gloo")


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _initial_group_by_length_mode(desired: bool, checkpoint_active: bool, resume_step: int, steps_to_skip_in_epoch: int, steps_per_epoch: int,) -> bool:
    resumed_mid_epoch = resume_step > 0 and 0 < steps_to_skip_in_epoch < steps_per_epoch
    return bool(checkpoint_active) if resumed_mid_epoch else bool(desired)


def _resume_steps_to_skip(resume_step: int, start_epoch: int, steps_per_epoch: int, grad_accum_steps: int) -> int:
    steps_to_skip = resume_step - start_epoch * steps_per_epoch
    if steps_to_skip >= 0:
        return steps_to_skip
    # Final saves can align the step to the previous optimizer boundary.
    if -steps_to_skip < grad_accum_steps:
        return 0
    raise ValueError(
        f"Checkpoint step {resume_step} is incompatible with epoch {start_epoch}: "
        f"expected at least {start_epoch * steps_per_epoch}"
    )


def _resume_position(
    checkpoint_step: int, optimizer_step: int, checkpoint_epoch: int, checkpoint_sample_offset, 
    dataset_size: int, global_batch_size: int, local_batch_size: int, world_size: int, checkpoint_group_by_length: bool = False,
):
    if optimizer_step == 0:
        if checkpoint_step != 0:
            raise ValueError("Cannot infer the previous GPU count from optimizer step 0")
        old_grad_accum_steps = global_batch_size // (local_batch_size * world_size)
    else:
        if checkpoint_step % optimizer_step != 0:
            raise ValueError("Checkpoint step is not aligned with its optimizer step")
        old_grad_accum_steps = checkpoint_step // optimizer_step

    old_total_batch_size = global_batch_size // old_grad_accum_steps
    if old_total_batch_size % local_batch_size != 0:
        raise ValueError("Checkpoint batch size is incompatible with the configured per-GPU batch size")
    old_world_size = old_total_batch_size // local_batch_size
    old_steps_per_epoch = dataset_size // old_total_batch_size

    if checkpoint_sample_offset is None:
        old_steps_to_skip = _resume_steps_to_skip(checkpoint_step, checkpoint_epoch, old_steps_per_epoch,old_grad_accum_steps,)
        checkpoint_sample_offset = old_steps_to_skip * old_total_batch_size

    if checkpoint_group_by_length and old_world_size != world_size and checkpoint_sample_offset > 0:
        raise ValueError("Cannot change GPU count while resuming a length-grouped epoch")

    new_total_batch_size = local_batch_size * world_size
    new_steps_per_epoch = dataset_size // new_total_batch_size
    steps_to_skip = min(math.ceil(checkpoint_sample_offset / new_total_batch_size), new_steps_per_epoch,)
    aligned_sample_offset = max(checkpoint_sample_offset, steps_to_skip * new_total_batch_size,)
    omitted_samples = aligned_sample_offset - checkpoint_sample_offset
    new_grad_accum_steps = global_batch_size // new_total_batch_size
    translated_step = optimizer_step * new_grad_accum_steps
    return (translated_step, steps_to_skip, aligned_sample_offset, omitted_samples, old_world_size,)


def _fractional_generation_step(eval_epoch: float, steps_per_epoch: int, grad_accum_steps: int,):
    threshold = math.ceil(eval_epoch * steps_per_epoch)
    optimizer_boundary = (
        (threshold + grad_accum_steps - 1) // grad_accum_steps
    ) * grad_accum_steps
    epoch_boundary = (
        (threshold + steps_per_epoch - 1) // steps_per_epoch
    ) * steps_per_epoch
    return min(optimizer_boundary, epoch_boundary)


def _next_scheduled_generation_step(epoch: int, steps_per_epoch: int, grad_accum_steps: int, 
                                    eval_freq: float, next_fractional_eval_epoch,):
    if next_fractional_eval_epoch is not None:
        return _fractional_generation_step(next_fractional_eval_epoch, steps_per_epoch, grad_accum_steps,)

    if eval_freq >= 1:
        eval_epoch = epoch + 1
        while eval_epoch % eval_freq != 0:
            eval_epoch += 1
        return eval_epoch * steps_per_epoch

    return None


def _should_save_pre_generation_autoresume(step: int, next_generation_step, grad_accum_steps: int,) -> bool:
    return next_generation_step is not None and step <= next_generation_step < step + grad_accum_steps


def _is_resuming_fractional_generation(
    resume_step: int, resume_epoch: int, total_epochs: int, steps_per_epoch: int, 
    grad_accum_steps: int, fractional_eval_interval: float, next_fractional_eval_epoch: float,
) -> bool:
    if resume_epoch >= total_epochs:
        return False
    previous_eval_epoch = next_fractional_eval_epoch - fractional_eval_interval
    generation_step = _fractional_generation_step(previous_eval_epoch, steps_per_epoch, grad_accum_steps,)
    return resume_step > 0 and resume_step == generation_step


def parse_args():
    parser = argparse.ArgumentParser(description="Train ELF Diffusion Model (PyTorch).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file to override defaults.")
    parser.add_argument(
        "--config_override", action="append", default=[],
        help="Override config values (field_name=value). Can be specified multiple times.",
    )
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def _validate_explicit_resume_config(config):
    if config.resume and config.resume_only_weights:
        raise ValueError("resume and resume_only_weights cannot both be set")
    source_ema = getattr(config, "resume_only_weights_ema", None)
    if source_ema is not None:
        if not config.resume_only_weights:
            raise ValueError("resume_only_weights_ema requires resume_only_weights")
        if not 0.0 < float(source_ema) < 1.0:
            raise ValueError("resume_only_weights_ema must be in (0, 1)")


def _find_auto_resume_checkpoint(output_dir, validate_loadable: bool = False):
    autoresume_ckpt = os.path.abspath(os.path.expanduser(
        os.path.join(output_dir, AUTORESUME_CHECKPOINT_NAME)
    ))
    output_ckpts = find_all_checkpoints(output_dir)
    autoresume_exists = os.path.isfile(autoresume_ckpt)
    if not validate_loadable:
        if autoresume_exists:
            return autoresume_ckpt, "latest autoresume"
        if output_ckpts:
            return output_ckpts[-1], "output_dir"
        return None, None

    autoresume_step = None
    if autoresume_exists:
        try:
            autoresume_step = load_checkpoint_optimizer_step(autoresume_ckpt)
        except Exception as e:
            log_for_0(f"Skipping unloadable latest autoresume checkpoint {autoresume_ckpt}: {e}", level=logging.WARNING,)

    for output_ckpt in reversed(output_ckpts):
        try:
            output_step = load_checkpoint_optimizer_step(output_ckpt)
        except Exception as e:
            log_for_0(f"Skipping unloadable output_dir checkpoint {output_ckpt}: {e}", level=logging.WARNING,)
            continue
        if autoresume_step is None or output_step > autoresume_step:
            return output_ckpt, "output_dir"
        return autoresume_ckpt, "latest autoresume"

    if autoresume_step is not None:
        return autoresume_ckpt, "latest autoresume"
    if autoresume_exists or output_ckpts:
        raise ValueError(f"No loadable auto-resume checkpoint found in {output_dir}")
    return None, None


def _find_auto_resume_checkpoint_distributed(output_dir):
    if not dist.is_available() or not dist.is_initialized():
        return _find_auto_resume_checkpoint(output_dir, validate_loadable=True)

    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = {
                "result": _find_auto_resume_checkpoint(output_dir, validate_loadable=True),
                "error": None,
            }
        except Exception as exc:
            payload[0] = {"result": None, "error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(payload, src=0)
    if payload[0]["error"] is not None:
        raise RuntimeError(f"Rank 0 auto-resume checkpoint probe failed: {payload[0]['error']}")
    return tuple(payload[0]["result"])


def _validate_resume_config(config):
    if _find_auto_resume_checkpoint(config.output_dir)[0] is None:
        _validate_explicit_resume_config(config)


def _validate_ema_config(config):
    decays = normalize_ema_decay1(config.ema_decay1)
    if not decays:
        raise ValueError("ema_decay1 must contain at least one decay")
    for decay in decays:
        if not (0.0 < decay < 1.0):
            raise ValueError("all ema_decay1 values must be in (0, 1)")
    keys = [ema_decay_key(decay) for decay in decays]
    if len(set(keys)) != len(keys):
        raise ValueError("ema_decay1 values must be unique")
    config.ema_decay1 = decays

    requested = parse_eval_ema_arg(getattr(config, "training_generation_ema", None))
    if requested is None:
        return [None] + keys
    missing = [key for key in requested if key is not None and key not in keys]
    if missing:
        raise ValueError(f"training_generation_ema requests unavailable EMA(s) {missing}; configured EMA(s): {keys}")
    return requested


def _resolve_resume_reg_projection_topology(config, checkpoint_path):
    if checkpoint_path is None or not bool(getattr(config, "reg_enabled", False)):
        return
    checkpoint_reg_dim, checkpoint_topology = infer_reg_projection_config(
        checkpoint_path
    )
    if checkpoint_reg_dim is None:
        return
    configured_topology = getattr(config, "reg_projection_topology", "separate")
    topology_explicit = bool(getattr(
        config, "_reg_projection_topology_explicit", True,
    ))
    if topology_explicit and configured_topology != checkpoint_topology:
        raise ValueError(
            "REG projection topology in resume config does not match checkpoint "
            f"({configured_topology} vs {checkpoint_topology})"
        )
    config.reg_projection_topology = checkpoint_topology


def run_training(config, *, force_cpu: bool = False):
    _init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cpu") if force_cpu or not torch.cuda.is_available() else torch.device(f"cuda:{local_rank}")
    rank = _rank()
    world = _world_size()

    training_generation_ema_keys = _validate_ema_config(config)
    if config.ema_warmup_updates < 0:
        raise ValueError("ema_warmup_updates must be non-negative")
    validate_repa_shift(config)
    if float(getattr(config, "reg_loss_weight", 0.03)) < 0:
        raise ValueError("reg_loss_weight must be non-negative")
    if bool(config.olt_enabled) and bool(config.reg_enabled):
        raise ValueError("olt_enabled and reg_enabled are mutually exclusive")
    if config.reg_teacher_pooling not in {"eot", "last", "mean"}:
        raise ValueError("reg_teacher_pooling must be 'eot', 'last', or 'mean'")
    if config.repa_reg_target_source not in {"repa", "reg"}:
        raise ValueError("repa_reg_target_source must be 'repa' or 'reg'")
    if config.repa_enabled and config.reg_enabled and config.reg_teacher_pooling == "last" and config.reg_teacher_layer == config.repa_teacher_layer:
        raise ValueError("REG 'last' pooling requires reg_teacher_layer to differ from repa_teacher_layer")
    if config.repa_projector_type not in {"mlp", "transformer"}:
        raise ValueError("repa_projector_type must be 'mlp' or 'transformer'")
    if int(config.repa_adapter_layers) < 1:
        raise ValueError("repa_adapter_layers must be positive")
    if config.repa_adapter_attention_mode not in REPA_ADAPTER_ATTENTION_MODES:
        raise ValueError(
            f"repa_adapter_attention_mode must be one of "
            f"{sorted(REPA_ADAPTER_ATTENTION_MODES)}"
        )
    if not 0.0 <= float(config.repa_mask_ratio) <= 1.0:
        raise ValueError("repa_mask_ratio must be in [0, 1]")
    if config.repa_projector_type == "mlp" and float(config.repa_mask_ratio) != 0.0:
        raise ValueError("repa_mask_ratio must be 0 for the MLP REPA projector")
    if config.repa_projector_type == "mlp" and config.repa_adapter_attention_mode != "full":
        raise ValueError("non-full REPA adapter attention requires the transformer REPA projector")
    for field_name in ("repa_prompt_loss_weight", "repa_response_loss_weight"):
        if float(getattr(config, field_name)) < 0:
            raise ValueError(f"{field_name} must be non-negative")
    for field_name in ("encoder_dim", "repa_teacher_dim", "reg_teacher_dim"):
        value = getattr(config, field_name, None)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{field_name} must be positive")
    if getattr(config, "repa_teacher_input_format", "raw") not in TEACHER_INPUT_FORMATS:
        raise ValueError(
            f"repa_teacher_input_format must be one of {sorted(TEACHER_INPUT_FORMATS)}"
        )
    if config.repa_teacher_input_format == "qwen3_chat":
        # Prepared datasets may already contain terminal special tokens.
        raise ValueError(
            "repa_teacher_input_format='qwen3_chat' is not supported for training; "
            "use 'raw'"
        )

    ckpt_path, ckpt_source = _find_auto_resume_checkpoint_distributed(
        config.output_dir
    )
    if ckpt_path is None:
        _validate_explicit_resume_config(config)
        if config.resume:
            ckpt_path = find_latest_checkpoint(config.resume) or config.resume
            ckpt_source = "explicit resume"
    _resolve_resume_reg_projection_topology(config, ckpt_path)

    log_for_0("=" * 60)
    log_for_0("ELF Diffusion Model Training (PyTorch)")
    log_for_0("=" * 60)
    log_for_0(f"Model: {config.model}")
    log_for_0(f"Encoder Model: {config.encoder_model_name}")
    log_for_0(f"Encoder Checkpoint: {config.encoder_checkpoint}")
    log_for_0(f"Data: {config.data_path}")
    log_for_0(f"Max sequence length: {config.max_length}")
    log_for_0(f"Output dir: {config.output_dir}")
    log_for_0(f"Batch size per device: {config.batch_size}")
    log_for_0(f"Generation batch size per device: {config.generation_batch_size}")
    log_for_0(f"Number of epochs: {config.epochs}")
    log_for_0(f"PyTorch device: {device}, world_size={world}")
    log_for_0(f"BF16 autocast: {bool(getattr(config, 'use_bf16', True)) and device.type == 'cuda'}")
    log_for_0(f"Gradient checkpointing: {bool(getattr(config, 'gradient_checkpointing', True))}")
    log_for_0(f"REPA enabled: {bool(getattr(config, 'repa_enabled', False))}")
    if config.repa_enabled:
        log_for_0(
            f"REPA projector: {config.repa_projector_type}, "
            f"adapter_layers={config.repa_adapter_layers}, "
            f"adapter_attention={config.repa_adapter_attention_mode}, "
            f"mask_ratio={config.repa_mask_ratio}"
        )
    log_for_0(f"OLT enabled: {bool(getattr(config, 'olt_enabled', False))}")
    log_for_0(f"REG enabled: {bool(getattr(config, 'reg_enabled', False))}")
    log_for_0(f"EMA decays: {config.ema_decay1}")
    log_for_0("=" * 60)

    if config.use_wandb and rank == 0 and wandb is not None:
        wandb_config = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
        wandb_tags = config.wandb_tag.split(",") if config.wandb_tag else None
        wandb.init(
            project=config.wandb_project, entity=config.wandb_entity,
            name=config.wandb_run_name, id=config.wandb_run_name, resume=config.wandb_resume,
            tags=wandb_tags, config=wandb_config, dir="/tmp",
        )
        resume_suffix = f" (resume={config.wandb_resume}, id={config.wandb_run_name})"
        log_for_0(f"Wandb initialized: {wandb.run.url}{resume_suffix}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # Per-rank seed so stochastic draws (decoder/denoiser branch coin,
    # timesteps, noise) diverge across ranks. A shared seed would make every
    # rank take the same branch in lockstep, producing spiky decoder gradients
    # instead of an evenly-mixed CE/L2 reduction.
    g = torch.Generator(device="cpu").manual_seed(config.seed + rank)

    # TF32 for fp32 matmuls on Ampere/Hopper (no hyperparameter change).
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")

    train_dataset, eval_dataset = load_dataset(config)
    if config.latent_cache_config_path:
        train_dataset, eval_dataset, cache_info = attach_latent_caches(
            config, tokenizer, train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        validate_repa_shift(config, "qwen3")
        encoder = None
        teacher_target_provider = None
        auxiliary_special_token_ids = tuple(
            int(token_id) for token_id in tokenizer.all_special_ids
        )
        text_encoder_dim = cache_info.text_dim
        repa_dim = cache_info.teacher_dim if config.repa_enabled else None
        reg_dim = cache_info.teacher_dim if config.reg_enabled else None
        log_for_0(
            f"Using compressed latent caches: text dim={text_encoder_dim}, "
            f"REPA dim={repa_dim}, REG dim={reg_dim}; Qwen weights are not loaded"
        )
    else:
        log_for_0(f"Loading Encoder config: {config.encoder_model_name}...")
        encoder_config, encoder = get_encoder(
            config.encoder_model_name, torch.float32,
            encoder_dim=config.encoder_dim, encoder_layer=config.encoder_layer,
        )
        log_for_0(f"Encoder d_model: {encoder_config.d_model}")
        text_encoder_dim = encoder_config.d_model
        auxiliary_special_token_ids = (
            tuple(int(token_id) for token_id in tokenizer.all_special_ids)
            if encoder_config.family == "qwen3" else ()
        )

        repa_dim = repa_target_dim(config, encoder_config) if config.repa_enabled else None
        reg_dim = reg_target_dim(config, encoder_config) if config.reg_enabled else None
        if reg_dim is not None:
            config.reg_teacher_dim = reg_dim
        teacher_target_provider = None
        needs_teacher = bool(config.reg_enabled) or (
            bool(config.repa_enabled) and not same_repa_teacher(config, encoder_config)
        )
        if config.repa_enabled and not needs_teacher:
            log_for_0("REPA target: raw clean-encoder state (shared forward)")
        teacher = encoder
        teacher_config = encoder_config
        if needs_teacher:
            if same_repa_teacher_model(config, encoder_config):
                log_for_0(f"Teacher target: shared encoder weights ({config.repa_teacher_model_name})")
            else:
                log_for_0(f"Loading teacher: {config.repa_teacher_model_name}...")
                teacher_config, teacher = get_encoder(
                    config.repa_teacher_model_name, torch.float32,
                )

        validate_repa_shift(config, teacher_config.family)

        if config.repa_enabled or config.reg_enabled:
            validate_external_repa_teacher(teacher, teacher_config, encoder_config)

        encoder_layers = [encoder_config.encoder_layer]
        teacher_layers = []
        if needs_teacher:
            if config.repa_enabled:
                teacher_layers.append(config.repa_teacher_layer)
            if config.reg_enabled:
                teacher_layers.append(config.reg_teacher_layer)
        if teacher.model is encoder.model:
            encoder_layers.extend(teacher_layers)
        elif teacher_config.family == "qwen3":
            truncated = truncate_qwen3_model_for_layers(teacher.model, teacher_layers)
            if truncated is not None:
                log_for_0(f"Teacher early exit: Qwen3 hidden state {truncated}")
        if encoder_config.family == "qwen3":
            truncated = truncate_qwen3_model_for_layers(encoder.model, encoder_layers)
            if truncated is not None:
                log_for_0(f"Encoder early exit: Qwen3 hidden state {truncated}")

        encoder = encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        if teacher is not encoder:
            teacher = teacher.to(device).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        dataset_tokenizer_name = config.tokenizer_name or config.encoder_model_name
        encoder_tokenizer = (
            tokenizer if dataset_tokenizer_name == config.encoder_model_name
            else AutoTokenizer.from_pretrained(config.encoder_model_name)
        )
        teacher_tokenizer = (
            AutoTokenizer.from_pretrained(config.repa_teacher_model_name)
            if needs_teacher else None
        )
        report_path = (
            os.path.join(config.output_dir, "tokenizer_compatibility.json")
            if rank == 0 else None
        )
        compatibility = validate_tokenizer_compatibility(
            tokenizer,
            encoder_tokenizer,
            teacher_tokenizer,
            encoder,
            teacher if needs_teacher else None,
            dataset_name=dataset_tokenizer_name,
            encoder_name=config.encoder_model_name,
            teacher_name=config.repa_teacher_model_name if needs_teacher else None,
            encoder_family=encoder_config.family,
            teacher_family=teacher_config.family if needs_teacher else None,
            report_path=report_path,
        )
        encoder.incompatible_dataset_token_ids = tuple(
            compatibility["dataset_encoder_incompatible_added_token_ids"]
        )
        encoder.token_consumer_name = f"clean encoder {config.encoder_model_name}"

        if needs_teacher:
            teacher_target_provider = TeacherTargetProvider(
                teacher,
                teacher_tokenizer,
                config.repa_teacher_input_format,
                require_reg=bool(config.reg_enabled),
                reg_teacher_pooling=config.reg_teacher_pooling,
                incompatible_token_ids=compatibility[
                    "dataset_teacher_incompatible_added_token_ids"
                ],
                consumer_name=f"teacher {config.repa_teacher_model_name}",
                source_special_token_ids=auxiliary_special_token_ids,
            )
            log_for_0(
                f"Teacher targets: format={config.repa_teacher_input_format}, "
                f"REPA dim={repa_dim}, REG dim={reg_dim}, "
                f"REG pooling={config.reg_teacher_pooling}"
            )

    log_for_0(f"Creating {config.model} model...")
    # Use the full tokenizer length for CE heads; tokenizer.vocab_size can exclude
    # added special tokens that still appear in tokenized Qwen targets.
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    log_for_0(f"Tokenizer vocab: CE head={vocab_size}")
    # Reset after frozen HF loads so ELF init is independent of Transformers RNG side effects.
    torch.manual_seed(config.seed)
    model = ELF_models[config.model](
        text_encoder_dim=text_encoder_dim, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        gradient_checkpointing=bool(getattr(config, "gradient_checkpointing", True)),
        repa_target_dim=repa_dim, repa_projector_dim=config.repa_projector_dim,
        repa_projector_layers=config.repa_projector_layers,
        repa_projector_type=config.repa_projector_type,
        repa_adapter_layers=config.repa_adapter_layers,
        reg_target_dim=reg_dim,
        reg_projection_topology=config.reg_projection_topology,
        olt_enabled=bool(config.olt_enabled),
    ).to(device)
    if config.repa_enabled and not (1 <= int(config.repa_depth) <= int(model.depth)):
        raise ValueError(f"repa_depth must be in [1, {model.depth}], got {config.repa_depth}")

    total_params = sum(p.numel() for p in model.parameters())
    log_for_0(f"ELF parameters: {total_params:,}")
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_for_0(f"Total trainable parameters: {total_trainable:,}")

    # Keep initialization identical across ranks, then make runtime stochastic
    # ops (e.g. dropout) rank-specific.
    torch.manual_seed(config.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + rank)

    if config.global_batch_size is None or config.batch_size is None:
        raise ValueError("Both global_batch_size and batch_size must be specified")
    if config.global_batch_size <= 0 or config.batch_size <= 0:
        raise ValueError("global_batch_size and batch_size must be positive")

    local_batch_size = config.batch_size
    total_batch_size = local_batch_size * world
    if config.global_batch_size % total_batch_size != 0:
        raise ValueError(
            "global_batch_size must be divisible by batch_size * world_size "
            f"({config.global_batch_size} vs {local_batch_size} * {world} = {total_batch_size})"
        )
    grad_accum_steps = config.global_batch_size // total_batch_size
    config.grad_accum_steps = grad_accum_steps
    autoresume_save_steps = (AUTORESUME_SAVE_STEPS_BASE + grad_accum_steps // 2) // grad_accum_steps

    steps_per_epoch = len(train_dataset) // total_batch_size
    num_train_steps = steps_per_epoch * config.epochs
    fractional_eval_interval = None
    if 0 < config.eval_freq < 1:
        fractional_eval_interval = float(config.eval_freq)
        if math.ceil(fractional_eval_interval * steps_per_epoch) < 1:
            raise ValueError(f"eval_freq={config.eval_freq} is too small for {steps_per_epoch} steps/epoch")
    if config.warmup_steps >= 0:
        num_warmup_steps = config.warmup_steps
    elif config.warmup_epochs is not None:
        num_warmup_steps = int(config.warmup_epochs * steps_per_epoch)
    else:
        num_warmup_steps = 0

    # Gradient accumulation: LR schedule is parameterized in optimizer steps
    num_optimizer_steps = num_train_steps // grad_accum_steps
    if config.warmup_steps >= 0:
        num_warmup_optimizer_steps = num_warmup_steps
    else:
        num_warmup_optimizer_steps = num_warmup_steps // grad_accum_steps

    # Effective learning rate (scaled with effective global batch size)
    if config.lr is None or config.lr <= 0:
        if config.lr is not None:
            log_for_0(f"Configured lr={config.lr} is non-positive; recomputing from blr={config.blr}")
        config.lr = config.blr * config.global_batch_size / 256

    log_for_0(
        f"World={world} | batch local={local_batch_size}, total={total_batch_size}, "
        f"effective={config.global_batch_size} | "
        f"steps/epoch={steps_per_epoch}, total_train={num_train_steps}, "
        f"warmup={num_warmup_steps}, lr={config.lr:.2e}"
    )
    log_for_0(f"Grad accum={grad_accum_steps}, optimizer steps={num_optimizer_steps}, warmup optimizer steps={num_warmup_optimizer_steps}")
    log_for_0(f"Group training batches by length: {bool(config.group_by_length)}")

    lr_fn = create_learning_rate_fn(
        num_train_steps=num_optimizer_steps, num_warmup_steps=num_warmup_optimizer_steps,
        learning_rate=config.lr, schedule=config.lr_schedule, min_lr=config.min_lr,
    )
    optimizer = get_optimizer(model, config, lr=config.lr, grad_accum_steps=grad_accum_steps)
    lr_scheduler = attach_lr_scheduler(optimizer, lr_fn)

    state = TrainState(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
        ema_params1=TrainState.init_emas(model, config.ema_decay1),
        step=0, epoch=0, dropout_generator=g,
    )

    start_epoch, resume_step = 0, 0
    steps_to_skip_in_epoch = 0
    resume_epoch_fractional = 0.0  # Fractional epoch for save-point tracking
    if ckpt_source == "latest autoresume":
        log_for_0(f"Auto-resuming from latest autoresume checkpoint: {ckpt_path}")
    elif ckpt_source == "output_dir":
        log_for_0(f"Auto-resuming from output_dir checkpoint: {ckpt_path}")
    elif ckpt_source == "explicit resume":
        log_for_0(f"Bootstrapping from resume checkpoint: {ckpt_path}")
    elif config.resume_only_weights:
        state = load_weights_only_checkpoint(config.resume_only_weights, state, source_ema_decay=getattr(config, "resume_only_weights_ema", None),)
        log_for_0("Initialized from checkpoint weights only; training state starts from scratch")

    if ckpt_path:
        state, checkpoint_step = load_checkpoint(ckpt_path, state)
        resume_epoch_fractional = float(state.epoch)
        start_epoch = int(state.epoch)
        optimizer_step = int(state.lr_scheduler.last_epoch)
        (resume_step, steps_to_skip_in_epoch, sample_offset_in_epoch, omitted_samples, old_world_size,) = \
            _resume_position(
                checkpoint_step=checkpoint_step, optimizer_step=optimizer_step, checkpoint_epoch=start_epoch,
                checkpoint_sample_offset=state.sample_offset_in_epoch, dataset_size=len(train_dataset), 
                global_batch_size=config.global_batch_size, local_batch_size=local_batch_size, world_size=world,
                checkpoint_group_by_length=state.group_by_length_active or bool(config.group_by_length),
            )
        checkpoint_sample_offset = sample_offset_in_epoch - omitted_samples
        state.step = resume_step
        state.sample_offset_in_epoch = sample_offset_in_epoch
        log_for_0(
            f"Resume progress: optimizer step {optimizer_step}, GPUs {old_world_size}->{world}, "
            f"training step {checkpoint_step}->{resume_step}, epoch sample offset "
            f"{checkpoint_sample_offset}->{sample_offset_in_epoch}, omitted={omitted_samples}"
        )
        log_for_0(f"Resumed from step {resume_step} (epoch {resume_epoch_fractional:.2f})")

    # torch.compile before DDP so only the inner module is compiled and
    # checkpoint I/O (which uses unwrap_model -> _orig_mod) still works.
    if device.type == "cuda":
        log_for_0("Compiling ELF model with torch.compile (first step will be slower)...")
        state = state.replace(model=torch.compile(state.model))

    if world > 1:
        # find_unused_parameters=False is safe: 0-mult sinks in train_step
        # (`0 * net_out.sum()` for CE, `0 * decoder_logits.sum()` for L2)
        # keep every head in the autograd graph on every step.
        state = state.replace(model=DDP(
            state.model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        ))

    os.makedirs(config.output_dir, exist_ok=True)

    if rank == 0:
        config_dict = {
            k: ([vars(sc) for sc in v] if isinstance(v, list) and v and isinstance(v[0], SamplingConfig) else v)
            for k, v in (vars(type(config)) | vars(config)).items() if not k.startswith("_")
        }
        config_path = os.path.join(config.output_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        log_for_0(f"Config saved to {config_path}")

    global_step = start_epoch * steps_per_epoch + steps_to_skip_in_epoch

    desired_group_by_length = bool(config.group_by_length)
    group_by_length_active = _initial_group_by_length_mode(desired_group_by_length, state.group_by_length_active, 
                                                           global_step, steps_to_skip_in_epoch, steps_per_epoch,)
    state.group_by_length_active = group_by_length_active

    def make_train_dataloader(*, group_by_length, skip_first_batches=0):
        return get_dataloader(
            train_dataset, batch_size=local_batch_size, shuffle=True, num_workers=config.num_workers, 
            drop_last=True, max_seq_length=config.max_length, pad_token_id=pad_token_id,
            max_input_seq_length=config.max_input_length, distributed=(world > 1), seed=config.seed,
            skip_first_batches=skip_first_batches, group_by_length=group_by_length,
        )

    train_dataloader = make_train_dataloader(
        group_by_length=group_by_length_active,
        skip_first_batches=steps_to_skip_in_epoch,
    )
    log_for_0(f"Length grouping active for current epoch: {group_by_length_active}")
    if group_by_length_active != desired_group_by_length:
        log_for_0(f"Length grouping will switch to {desired_group_by_length} at the next epoch boundary.")

    log_for_0("\n" + "=" * 60)
    log_for_0("Checkpoint and Evaluation Schedule")
    log_for_0("=" * 60)
    log_for_0(
        f"Steps/epoch={steps_per_epoch}, epochs={config.epochs}, total={steps_per_epoch * config.epochs} | "
        f"save every {config.save_freq} epoch(s), eval every {config.eval_freq} epoch(s)"
    )

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)
    log_for_0(f"Sampling configs: {len(config.sampling_configs)} config(s)")
    log_for_0(f"Training generation model variants: {training_generation_ema_keys}")

    log_for_0("\n" + "=" * 60)
    log_for_0("Starting Training")
    log_for_0("=" * 60)

    save_every_optimizer_steps = max(
        1, math.ceil(config.save_freq * steps_per_epoch / config.grad_accum_steps)
    )
    next_fractional_eval_epoch = None
    if fractional_eval_interval is not None:
        progress = global_step / steps_per_epoch
        next_fractional_eval_epoch = (
            math.floor(progress / fractional_eval_interval) + 1
        ) * fractional_eval_interval
        if _is_resuming_fractional_generation(global_step, start_epoch, config.epochs, steps_per_epoch, config.grad_accum_steps, fractional_eval_interval, next_fractional_eval_epoch,):
            log_for_0(f"Resuming generation from step {global_step}")
            run_generation(
                state=state, encoder=encoder, eval_dataset=eval_dataset, tokenizer=tokenizer, 
                config=config, generator=g, generation_batch_size=config.generation_batch_size,
                ema_keys=training_generation_ema_keys,
            )

    last_log_step = global_step
    train_metrics = []
    last_log_time = time.time()

    for epoch in range(start_epoch, config.epochs):
        log_for_0(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Free device buffers from previous epoch before allocating new ones, to avoid
        # transient OOM at epoch boundaries.
        if epoch > start_epoch:
            del train_loader, train_iterator
            train_metrics = []
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if group_by_length_active != desired_group_by_length:
                group_by_length_active = desired_group_by_length
                state.group_by_length_active = group_by_length_active
                train_dataloader = make_train_dataloader(group_by_length=group_by_length_active,)
                log_for_0(f"Length grouping is now active: {group_by_length_active}")

        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        train_iterator = iter(train_dataloader)
        train_loader = prefetch_to_device(train_iterator, size=4)

        initial_pbar = steps_to_skip_in_epoch if epoch == start_epoch else 0
        epoch_pbar = tqdm(
            total=steps_per_epoch, desc=f"Epoch {epoch + 1}", initial=initial_pbar,
            mininterval=1.0, disable=rank != 0,
        )

        for step_in_epoch, batch in enumerate(train_loader, start=initial_pbar):
            is_first_step = step_in_epoch == initial_pbar and epoch == start_epoch
            if is_first_step:
                log_for_0("Performing initial training step, this may take longer...")
            step_seed = stable_train_seed(config.seed, rank, state.step)
            torch.manual_seed(step_seed)
            if device.type == "cuda": torch.cuda.manual_seed_all(step_seed)
            if state.dropout_generator is not None: state.dropout_generator.manual_seed(step_seed)
            batch = prepare_batch(batch, config, generator=g)
            state, metrics = train_step(
                state,
                encoder=encoder,
                batch=batch,
                config=config,
                total_optimizer_steps=num_optimizer_steps,
                teacher_target_provider=teacher_target_provider,
                repa_special_token_ids=auxiliary_special_token_ids,
            )

            # Sync only on first step to measure torch.compile time;
            # float() on the loss below already forces a device-to-host sync.
            if is_first_step:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                log_for_0("First training step (torch.compile + execution) completed...")

            global_step += 1
            state.sample_offset_in_epoch += total_batch_size
            train_metrics.append(metrics)
            epoch_pbar.update(1)

            if state.step % config.grad_accum_steps == 0:
                optimizer_step = state.step // config.grad_accum_steps
                numbered_checkpoint_due = optimizer_step % save_every_optimizer_steps == 0
                next_generation_step = _next_scheduled_generation_step(
                    epoch=epoch, steps_per_epoch=steps_per_epoch, grad_accum_steps=config.grad_accum_steps, 
                    eval_freq=config.eval_freq, next_fractional_eval_epoch=next_fractional_eval_epoch,
                )
                autoresume_due = (
                    optimizer_step % autoresume_save_steps == 0
                    or _should_save_pre_generation_autoresume(global_step, next_generation_step, config.grad_accum_steps,)
                )
                if numbered_checkpoint_due:
                    save_checkpoint(state, config.output_dir, optimizer_step)
                    log_for_0(f"Saved checkpoint at optimizer step {optimizer_step}")
                if autoresume_due:
                    save_autoresume_checkpoint(state, config.output_dir)
                if (
                    numbered_checkpoint_due or autoresume_due
                ) and dist.is_available() and dist.is_initialized():
                    dist.barrier()

            if global_step % config.log_freq == 0:
                metric_names = ["loss", "base_loss", "l2_loss", "ce_loss"]
                if config.reg_enabled:
                    metric_names.extend(["reg", "reg_conditional", "reg_w", "reg_valid_frac"])
                metric_names.extend(["repa", "repa_w", "repa_valid_frac"])
                if config.repa_enabled:
                    metric_names.extend([
                        "repa_prompt", "repa_prompt_w", "repa_prompt_valid_frac",
                        "repa_response", "repa_response_w", "repa_response_valid_frac",
                    ])
                metric_values = [
                    torch.stack([m[name] for m in train_metrics]).mean()
                    for name in metric_names
                ]
                stacked = torch.stack(metric_values)
                # Average each metric across DDP ranks before logging — done
                # once per log_freq so we never sync on every train step.
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
                    stacked = stacked / dist.get_world_size()
                averages = dict(zip(metric_names, (float(x) for x in stacked.tolist())))
                avg_loss = averages["loss"]
                avg_base = averages["base_loss"]
                avg_l2 = averages["l2_loss"]
                avg_ce = averages["ce_loss"]
                avg_repa = averages["repa"]
                avg_repa_w = averages["repa_w"]
                avg_repa_valid = averages["repa_valid_frac"]
                now = time.time()
                steps_per_sec = (global_step - last_log_step) / max(now - last_log_time, 1e-8)
                current_lr = state.optimizer.param_groups[0]["lr"]

                postfix_dict = {
                    "step": f"{global_step}", "loss": f"{avg_loss:.4f}",
                    "base": f"{avg_base:.4f}",
                    "l2": f"{avg_l2:.4f}", "ce": f"{avg_ce:.4f}",
                    "repa": f"{avg_repa:.4f}", "repa_w": f"{avg_repa_w:.3f}",
                    "sps": f"{steps_per_sec:.1f}", "lr": f"{current_lr:.2e}",
                }
                if config.reg_enabled:
                    postfix_dict.update({
                        "reg": f"{averages['reg']:.4f}", "reg_w": f"{averages['reg_w']:.3f}",
                    })
                log_for_0(postfix_dict)
                epoch_pbar.set_postfix(**postfix_dict)

                if rank == 0:
                    reg_message = (
                        f"reg={averages['reg']:.4f}, "
                        f"reg_conditional={averages['reg_conditional']:.4f}, "
                        f"reg_w={averages['reg_w']:.3f}, "
                        f"reg_valid={averages['reg_valid_frac']:.3f}, "
                        if config.reg_enabled else ""
                    )
                    repa_group_message = (
                        f"repa_prompt={averages['repa_prompt']:.4f}, "
                        f"repa_response={averages['repa_response']:.4f}, "
                        if config.repa_enabled else ""
                    )
                    tqdm.write(
                        f"INFO - engine - Step {global_step}: loss={avg_loss:.4f}, "
                        f"base={avg_base:.4f}, l2={avg_l2:.4f}, ce={avg_ce:.4f}, "
                        f"{reg_message}"
                        f"repa={avg_repa:.4f}, repa_w={avg_repa_w:.3f}, "
                        f"repa_valid={avg_repa_valid:.3f}, "
                        f"{repa_group_message}"
                        f"lr={current_lr:.2e}, steps/sec={steps_per_sec:.2f}"
                    )
                    if config.use_wandb and wandb is not None:
                        current_epoch_progress = epoch + (step_in_epoch + 1) / steps_per_epoch
                        try:
                            wandb_metrics = {
                                "train_loss": avg_loss, "train_base_loss": avg_base,
                                "train_l2_loss": avg_l2,
                                "train_ce_loss": avg_ce, "lr": current_lr,
                                "train_repa": avg_repa, "train_repa_w": avg_repa_w,
                                "train_repa_valid_frac": avg_repa_valid,
                                "epoch": current_epoch_progress, "step": global_step,
                            }
                            if config.reg_enabled:
                                wandb_metrics.update({
                                    "train_reg": averages["reg"],
                                    "train_reg_conditional": averages["reg_conditional"],
                                    "train_reg_w": averages["reg_w"],
                                    "train_reg_valid_frac": averages["reg_valid_frac"],
                                })
                            if config.repa_enabled:
                                wandb_metrics.update({
                                    "train_repa_prompt": averages["repa_prompt"],
                                    "train_repa_prompt_w": averages["repa_prompt_w"],
                                    "train_repa_prompt_valid_frac": averages["repa_prompt_valid_frac"],
                                    "train_repa_response": averages["repa_response"],
                                    "train_repa_response_w": averages["repa_response_w"],
                                    "train_repa_response_valid_frac": averages["repa_response_valid_frac"],
                                })
                            wandb.log(wandb_metrics, step=global_step)
                        except Exception:
                            pass

                train_metrics = []
                last_log_step = global_step
                last_log_time = now

            if (
                next_fractional_eval_epoch is not None
                and (global_step / steps_per_epoch) >= next_fractional_eval_epoch
                and state.step % config.grad_accum_steps == 0
            ):
                del batch
                run_generation(
                    state=state, encoder=encoder, eval_dataset=eval_dataset,
                    tokenizer=tokenizer, config=config, generator=g,
                    generation_batch_size=config.generation_batch_size,
                    ema_keys=training_generation_ema_keys,
                )
                progress = global_step / steps_per_epoch
                while next_fractional_eval_epoch <= progress:
                    next_fractional_eval_epoch += fractional_eval_interval
                last_log_step = global_step
                last_log_time = time.time()

        epoch_pbar.close()
        current_epoch = epoch + 1
        state.epoch = current_epoch
        state.sample_offset_in_epoch = 0

        if (
            next_fractional_eval_epoch is not None
            and (global_step / steps_per_epoch) >= next_fractional_eval_epoch
        ):
            run_generation(
                state=state, encoder=encoder, eval_dataset=eval_dataset,
                tokenizer=tokenizer, config=config, generator=g,
                generation_batch_size=config.generation_batch_size,
                ema_keys=training_generation_ema_keys,
            )
            progress = global_step / steps_per_epoch
            while next_fractional_eval_epoch <= progress:
                next_fractional_eval_epoch += fractional_eval_interval
            last_log_step = global_step
            last_log_time = time.time()

        if config.eval_freq >= 1 and current_epoch % config.eval_freq == 0:
            run_generation(
                state=state, encoder=encoder, eval_dataset=eval_dataset,
                tokenizer=tokenizer, config=config, generator=g,
                generation_batch_size=config.generation_batch_size,
                ema_keys=training_generation_ema_keys,
            )
            last_log_step = global_step
            last_log_time = time.time()

    log_for_0("\n" + "=" * 60)
    log_for_0("Final Generation")
    log_for_0("=" * 60)
    if (
        next_fractional_eval_epoch is not None
        and (global_step / steps_per_epoch) >= next_fractional_eval_epoch
    ):
        run_generation(
            state=state, encoder=encoder, eval_dataset=eval_dataset,
            tokenizer=tokenizer, config=config, generator=g,
            generation_batch_size=config.generation_batch_size,
            ema_keys=training_generation_ema_keys,
        )
    optimizer_step = state.step // config.grad_accum_steps
    original_step = state.step
    state.step = optimizer_step * config.grad_accum_steps
    save_checkpoint(state, config.output_dir, optimizer_step)
    save_autoresume_checkpoint(state, config.output_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    state.step = original_step
    log_for_0(f"Final checkpoint saved to {config.output_dir}")
    if config.use_wandb and rank == 0 and wandb is not None:
        wandb.finish()


def main():
    """CLI entry point: parse args, load config, then run training."""
    args = parse_args()
    config = load_config_from_yaml(args.config)
    config._reg_projection_topology_explicit = config_field_is_explicit(
        args.config, args.config_override, "reg_projection_topology",
    )
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
    _validate_resume_config(config)
    add_file_logging(config.output_dir)
    if args.config_override:
        log_for_0(f"Applied {len(args.config_override)} config override(s)")
    run_training(config, force_cpu=args.use_cpu)


if __name__ == "__main__":
    main()
