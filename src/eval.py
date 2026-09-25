#!/usr/bin/env python
"""Evaluation script for trained ELF models: load a checkpoint and generate text samples."""

import argparse
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from modules.text_encoder import get_encoder
from modules.model import ELF_models
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import (
    infer_olt_enabled, infer_reg_projection_config,
    infer_repa_module_config,
    load_checkpoint,
)
from utils.train_utils import TrainState, get_optimizer
from utils.data_utils import load_jsonl_dataset, load_dataset_split, get_pad_token_id
from utils.latent_cache_utils import attach_latent_caches
from utils.repa_utils import truncate_qwen3_model_for_layers
from utils.teacher_utils import validate_tokenizer_compatibility
from generation import parse_eval_ema_arg, resolve_eval_ema_keys, run_generation
from configs.config import (
    apply_config_overrides, config_field_is_explicit, load_config_from_yaml,
    load_sampling_configs,
)

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)


def _resolve_checkpoint_reg_dim(config, checkpoint_path):
    checkpoint_reg_dim, checkpoint_reg_topology = infer_reg_projection_config(
        checkpoint_path
    )
    checkpoint_olt_enabled = infer_olt_enabled(checkpoint_path)
    config_reg_enabled = bool(getattr(config, "reg_enabled", False))
    config_olt_enabled = bool(getattr(config, "olt_enabled", False))
    if config_reg_enabled and config_olt_enabled:
        raise ValueError("Evaluation config cannot enable both OLT and REG")
    if checkpoint_reg_dim is not None and checkpoint_olt_enabled:
        raise ValueError("Checkpoint cannot contain both OLT and REG parameters")
    if config_reg_enabled != (checkpoint_reg_dim is not None):
        raise ValueError(
            "REG mode in evaluation config does not match checkpoint "
            f"(config={config_reg_enabled}, checkpoint={checkpoint_reg_dim is not None})"
        )
    if config_olt_enabled != checkpoint_olt_enabled:
        raise ValueError(
            "OLT mode in evaluation config does not match checkpoint "
            f"(config={config_olt_enabled}, checkpoint={checkpoint_olt_enabled})"
        )
    configured_reg_dim = getattr(config, "reg_teacher_dim", None)
    if (
        checkpoint_reg_dim is not None
        and configured_reg_dim is not None
        and not getattr(config, "latent_cache_config_path", None)
        and int(configured_reg_dim) != checkpoint_reg_dim
    ):
        raise ValueError(
            "reg_teacher_dim in evaluation config does not match checkpoint "
            f"({configured_reg_dim} vs {checkpoint_reg_dim})"
        )
    if checkpoint_reg_dim is not None:
        config.reg_teacher_dim = checkpoint_reg_dim
        configured_topology = getattr(config, "reg_projection_topology", "separate")
        topology_explicit = bool(getattr(
            config, "_reg_projection_topology_explicit", True,
        ))
        if topology_explicit and configured_topology != checkpoint_reg_topology:
            raise ValueError(
                "REG projection topology in evaluation config does not match checkpoint "
                f"({configured_topology} vs {checkpoint_reg_topology})"
            )
        config.reg_projection_topology = checkpoint_reg_topology
    config.olt_enabled = checkpoint_olt_enabled
    return checkpoint_reg_dim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ELF model by generating text samples")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--config_override", action="append", default=[],
                        help="Override config values (field_name=value). Repeatable.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used when --seeds is not specified)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds to evaluate (e.g. '42,123,456'). Overrides --seed.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to a local checkpoint file.")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Force CPU even when CUDA is available.")
    parser.add_argument("--ema", type=str, default=None,
                        help="Comma-separated EMA decays to evaluate; use 0 for raw. Omit to evaluate raw plus all EMAs.")
    return parser.parse_args()


def _init_distributed():
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # Eval only gathers CPU objects (text strings), so gloo is sufficient.
        dist.init_process_group(backend="gloo")


def main():
    args = parse_args()
    _init_distributed()

    device = torch.device("cpu") if args.use_cpu or not torch.cuda.is_available() else torch.device("cuda")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    log_for_0("Loading configuration...")
    config = load_config_from_yaml(args.config)
    config._reg_projection_topology_explicit = config_field_is_explicit(
        args.config, args.config_override, "reg_projection_topology",
    )
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} config override(s)")

    world = dist.get_world_size() if dist.is_initialized() else 1
    if config.global_batch_size is not None:
        log_for_0(f"Using global batch size for evaluation: {config.global_batch_size}")
        total_batch_size = config.global_batch_size
        local_batch_size = total_batch_size // world
        config.batch_size = local_batch_size
    elif config.batch_size is not None:
        log_for_0(f"Using batch size per device: {config.batch_size}")
        total_batch_size = config.batch_size * world
        local_batch_size = config.batch_size
        config.global_batch_size = total_batch_size
    else:
        raise ValueError("Either global_batch_size or batch_size must be specified")

    log_for_0(f"Config loaded from {args.config}")
    log_for_0(f"Model: {config.model}")
    log_for_0(f"Encoder Model: {config.encoder_model_name}")
    log_for_0(f"Encoder Checkpoint: {config.encoder_checkpoint}")
    log_for_0(f"Max length: {config.max_length}")
    log_for_0(f"Max input length: {config.max_input_length}")
    log_for_0(f"Num samples: {config.num_samples}")
    log_for_0(f"Sampling configs: {len(config.sampling_configs)} config(s)")
    log_for_0(f"BF16 autocast (sampling): {bool(getattr(config, 'use_bf16', True)) and device.type == 'cuda'}")
    log_for_0(f"torch.compile (eval model): {bool(getattr(config, 'use_compile', False))}")
    log_for_0(f"REPA enabled in model config: {bool(getattr(config, 'repa_enabled', False))}")
    log_for_0(f"REG enabled in model config: {bool(getattr(config, 'reg_enabled', False))}")
    log_for_0(f"OLT enabled in model config: {bool(getattr(config, 'olt_enabled', False))}")

    seed_list = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds is not None else [args.seed]
    log_for_0(f"Seeds to evaluate: {seed_list}")

    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")

    eval_dataset = None
    if isinstance(config.eval_data_path, list):
        log_for_0("Loading datasets for conditional generation...")
        eval_dataset = []
        for spec in config.eval_data_path:
            path = spec["path"]
            if path.endswith(".jsonl"):
                dataset = load_jsonl_dataset(path, tokenizer, input_key="input", output_key="output",)
            else:
                dataset = load_dataset_split(path)
            eval_dataset.append(dataset)
            log_for_0(f"Eval dataset {spec['name']} size: {len(dataset)}")
    elif config.eval_data_path is not None:
        log_for_0("Loading dataset for conditional generation...")
        if config.eval_data_path.endswith(".jsonl"):
            eval_dataset = load_jsonl_dataset(
                config.eval_data_path, tokenizer,
                input_key="input", output_key="output",
            )
        else:
            eval_dataset = load_dataset_split(config.eval_data_path)
        log_for_0(f"Eval dataset size: {len(eval_dataset)}")

    if config.latent_cache_config_path:
        _, eval_dataset, cache_info = attach_latent_caches(
            config, tokenizer, eval_dataset=eval_dataset,
        )
        encoder = None
        text_encoder_dim = cache_info.text_dim
        log_for_0(
            f"Using compressed evaluation latents: text dim={text_encoder_dim}; "
            "Qwen weights are not loaded"
        )
    else:
        log_for_0(f"Loading Encoder: {config.encoder_model_name}...")
        encoder_config, encoder = get_encoder(
            config.encoder_model_name, torch.float32,
            encoder_dim=config.encoder_dim, encoder_layer=config.encoder_layer,
        )
        text_encoder_dim = encoder_config.d_model
        if encoder_config.family == "qwen3":
            truncated = truncate_qwen3_model_for_layers(
                encoder.model, [encoder_config.encoder_layer],
            )
            if truncated is not None:
                log_for_0(f"Encoder early exit: Qwen3 hidden state {truncated}")
        dataset_tokenizer_name = config.tokenizer_name or config.encoder_model_name
        encoder_tokenizer = (
            tokenizer if dataset_tokenizer_name == config.encoder_model_name
            else AutoTokenizer.from_pretrained(config.encoder_model_name)
        )
        compatibility = validate_tokenizer_compatibility(
            tokenizer, encoder_tokenizer, None, encoder, None,
            dataset_name=dataset_tokenizer_name,
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
        for p in encoder.parameters():
            p.requires_grad_(False)

    # ELF model
    log_for_0(f"Creating {config.model} model...")
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    repa_module_config = infer_repa_module_config(args.checkpoint_path)
    checkpoint_reg_dim = _resolve_checkpoint_reg_dim(config, args.checkpoint_path)
    if checkpoint_reg_dim is not None:
        log_for_0(f"REG projections from checkpoint: target_dim={checkpoint_reg_dim}")
    elif config.olt_enabled:
        log_for_0("OLT parameter found in checkpoint")
    repa_dim = None
    if repa_module_config is not None:
        projector_type, repa_dim, layer_count, repa_projector_dim = repa_module_config
        config.repa_projector_type = projector_type
        if projector_type == "mlp":
            config.repa_projector_layers = layer_count
            if repa_projector_dim is not None:
                config.repa_projector_dim = repa_projector_dim
        else:
            config.repa_adapter_layers = layer_count
        log_for_0(
            f"REPA projector from checkpoint: type={projector_type}, "
            f"target_dim={repa_dim}, layers={layer_count}"
        )
    model = ELF_models[config.model](
        text_encoder_dim=text_encoder_dim, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        repa_target_dim=repa_dim, repa_projector_dim=config.repa_projector_dim,
        repa_projector_layers=config.repa_projector_layers,
        repa_projector_type=config.repa_projector_type,
        repa_adapter_layers=config.repa_adapter_layers,
        reg_target_dim=checkpoint_reg_dim,
        reg_projection_topology=config.reg_projection_topology,
        olt_enabled=config.olt_enabled,
    ).to(device)

    # Train state template (only used to plumb EMA params + step/epoch).
    optimizer = get_optimizer(model, config, lr=1e-4)
    g = torch.Generator(device="cpu").manual_seed(config.seed)
    state = TrainState(
        model=model, optimizer=optimizer, lr_scheduler=None,
        ema_params1=TrainState.init_emas(model, config.ema_decay1), step=0, epoch=0,
        dropout_generator=g,
    )

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    log_for_0(f"Loading checkpoint from: {args.checkpoint_path}")
    state, _ = load_checkpoint(args.checkpoint_path, state, strict_ema_decay_match=False)
    state.model = state.model.to(device).eval()
    selected_ema_keys = resolve_eval_ema_keys(state, parse_eval_ema_arg(args.ema))
    log_for_0(f"Model variants to evaluate: {selected_ema_keys}")

    rank = dist.get_rank() if dist.is_initialized() else 0

    for seed_idx, seed_val in enumerate(seed_list):
        if len(seed_list) > 1:
            log_for_0(f"\n{'#' * 70}")
            log_for_0(f"Seed {seed_idx + 1}/{len(seed_list)}: {seed_val}")
            log_for_0(f"{'#' * 70}")

        # Per-rank offset so ranks generate different samples when sharding;
        # rank 0 keeps the original seed for single-GPU reproducibility.
        per_rank_seed = seed_val + rank * 1_000_003
        seed_gen = torch.Generator(device="cpu").manual_seed(per_rank_seed)
        torch.manual_seed(per_rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(per_rank_seed)
        original_output_dir = config.output_dir
        if len(seed_list) > 1:
            config.output_dir = os.path.join(original_output_dir, f"seed_{seed_val}")

        run_generation(
            state=state, encoder=encoder, eval_dataset=eval_dataset,
            tokenizer=tokenizer, config=config, generator=seed_gen,
            generation_batch_size=local_batch_size, eval_seed=seed_val,
            ema_keys=selected_ema_keys,
        )

        config.output_dir = original_output_dir

    log_for_0("\nEvaluation complete!")


if __name__ == "__main__":
    main()
