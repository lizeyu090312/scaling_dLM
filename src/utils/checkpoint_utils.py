import logging
import os
import re
import time
from typing import Any, Optional, Tuple

import torch

from utils.logging_utils import log_for_0, _process_index
from utils.train_utils import ema_decay_key, is_multi_ema_state, unwrap_model


AUTORESUME_CHECKPOINT_NAME = "latest_autoresume.pt"


def _local_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))






def _checkpoint_payload(state):
    inner_model = unwrap_model(state.model)
    grad_accum_buffers = {}
    if getattr(state, "grad_accum_buffers", None):
        for name, param in inner_model.named_parameters():
            buf = state.grad_accum_buffers.get(id(param))
            if buf is not None:
                grad_accum_buffers[name] = buf.detach().cpu()
    optimizer_step = (
        int(state.lr_scheduler.last_epoch)
        if state.lr_scheduler is not None else int(state.step)
    )
    payload = {
        "params": inner_model.state_dict(),
        "ema_params1": state.ema_params1,
        "ema_decays1": [float(k) for k in state.ema_params1.keys()],
        "opt_state": state.optimizer.state_dict(),
        "lr_scheduler": state.lr_scheduler.state_dict() if state.lr_scheduler is not None else None,
        "step": int(state.step),
        "optimizer_step": optimizer_step,
        "epoch": int(state.epoch),
        "sample_offset_in_epoch": int(state.sample_offset_in_epoch),
        "group_by_length_active": bool(getattr(state, "group_by_length_active", False)),
        "grad_accum_buffers": grad_accum_buffers,
    }
    return payload


def _state_ema_keys(state):
    return list(state.ema_params1.keys()) if is_multi_ema_state(state.ema_params1) else []


def _to_device_state(source, device_map, fallback_device, *, copy: bool = False):
    return {
        n: t.to(device_map.get(n, fallback_device), copy=copy)
        for n, t in source.items()
    }


def _normalize_nested_ema(ema_params):
    return {ema_decay_key(float(k)): v for k, v in ema_params.items()}


def _load_ema_params(ckpt, state, device_map, fallback_device, strict_decay_match: bool):
    ema_src = ckpt.get("ema_params1")
    desired_keys = _state_ema_keys(state)
    if ema_src is None:
        raise ValueError("checkpoint restore missing keys: ['ema_params1']")

    if is_multi_ema_state(ema_src):
        nested = _normalize_nested_ema(ema_src)
        if strict_decay_match and list(nested.keys()) != desired_keys:
            raise ValueError(
                f"checkpoint EMA decays {list(nested.keys())} do not match configured ema_decay1 {desired_keys}"
            )
    else:
        if len(desired_keys) == 1:
            key = desired_keys[0]
        elif strict_decay_match:
            raise ValueError(
                "old flat ema_params1 checkpoint can only resume when configured ema_decay1 has one value"
            )
        else:
            key = ema_decay_key(0.9999)
        nested = {key: ema_src}

    state.ema_params1 = {
        key: _to_device_state(params, device_map, fallback_device)
        for key, params in nested.items()
    }


def _write_checkpoint(payload, out_path: str):
    tmp_path = os.path.join(os.path.dirname(out_path), f".{os.path.basename(out_path)}.tmp.{os.getpid()}")
    log_for_0(f"Saving checkpoint to {out_path}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    log_for_0(f"Checkpoint written to {out_path}")


def save_checkpoint(state, output_dir: str, step: int):
    """Save model checkpoint locally as a single `checkpoint_<step>` file."""
    if _process_index() != 0:
        return
    ckpt_dir = _local_path(output_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = _checkpoint_payload(state)
    out_path = os.path.join(ckpt_dir, f"checkpoint_{step}")
    _write_checkpoint(payload, out_path)


def save_autoresume_checkpoint(state, output_dir: str):
    """Save the fixed-name checkpoint used for auto-resume."""
    if _process_index() != 0:
        return
    ckpt_dir = _local_path(output_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = _checkpoint_payload(state)
    out_path = os.path.join(ckpt_dir, AUTORESUME_CHECKPOINT_NAME)
    _write_checkpoint(payload, out_path)


def _checkpoint_step(checkpoint_name: str) -> int:
    """Extract the trailing checkpoint step from a name; -1 if absent."""
    match = re.search(r"(\d+)$", checkpoint_name)
    return int(match.group(1)) if match else -1


def find_all_checkpoints(ckpt_dir: str, prefix: str = "checkpoint_"):
    """Find local checkpoint paths in a directory, sorted by step ascending."""
    ckpt_dir = _local_path(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        return []
    pattern = re.compile(rf"{re.escape(prefix)}\d+$")
    names = sorted(
        [f for f in os.listdir(ckpt_dir) if pattern.fullmatch(f)],
        key=_checkpoint_step,
    )
    return [os.path.join(ckpt_dir, name) for name in names]


def find_latest_checkpoint(ckpt_dir: str, prefix: str = "checkpoint_"):
    """Return the latest local checkpoint path, or None."""
    all_ckpts = find_all_checkpoints(ckpt_dir, prefix)
    return all_ckpts[-1] if all_ckpts else None




def _restore_checkpoint(checkpoint_path: str, *, mmap: bool = False) -> Any:
    """Restore a checkpoint from a file or directory (latest inside dir)."""
    local = _local_path(checkpoint_path)
    resolved = local
    if os.path.isdir(local):
        latest = find_latest_checkpoint(local)
        if latest is not None and os.path.isfile(latest):
            resolved = latest
    if os.path.isfile(resolved):
        start = time.monotonic()
        mode = "mmap" if mmap else "eager"
        try:
            if mmap:
                checkpoint = torch.load(resolved, map_location="cpu", mmap=True)
            else:
                checkpoint = torch.load(resolved, map_location="cpu")
        except RuntimeError as error:
            if not mmap or "mmap can only be used with files saved with" not in str(error):
                raise
            log_for_0(f"Checkpoint does not support mmap; loading eagerly: {resolved}", level=logging.WARNING,)
            mode = "eager fallback"
            checkpoint = torch.load(resolved, map_location="cpu")
        elapsed = time.monotonic() - start
        log_for_0(f"Loaded checkpoint {resolved} using {mode} in {elapsed:.2f}s")
        return checkpoint
    raise FileNotFoundError(f"Local checkpoint not found: {checkpoint_path}")


def _repa_module_config_from_params(params):
    if not params:
        return None

    mlp_weights = []
    transformer_blocks = set()
    for key, weight in params.items():
        mlp_match = re.fullmatch(r"repa_projector\.net\.(\d+)\.weight", key)
        if mlp_match:
            mlp_weights.append((int(mlp_match.group(1)), weight))
        block_match = re.match(r"repa_projector\.blocks\.(\d+)\.", key)
        if block_match:
            transformer_blocks.add(int(block_match.group(1)))

    transformer_output = params.get("repa_projector.output.weight")
    transformer_mask = params.get("repa_projector.mask_token")
    has_mlp = bool(mlp_weights)
    has_transformer = (
        transformer_output is not None
        or transformer_mask is not None
        or bool(transformer_blocks)
    )
    if has_mlp and has_transformer:
        raise ValueError("Checkpoint contains both MLP and transformer REPA projectors")

    if has_mlp:
        final_idx, final_weight = max(mlp_weights, key=lambda item: item[0])
        if final_idx % 2 != 0:
            raise ValueError(f"Unexpected REPA projector weight index: net.{final_idx}.weight")
        num_layers = final_idx // 2 + 1
        projector_dim = None
        first_weight = params.get("repa_projector.net.0.weight")
        if num_layers > 1 and first_weight is not None:
            projector_dim = int(first_weight.shape[0])
        return "mlp", int(final_weight.shape[0]), num_layers, projector_dim

    if not has_transformer:
        return None
    if transformer_output is None or transformer_mask is None or not transformer_blocks:
        raise ValueError("Incomplete transformer REPA adapter in checkpoint")
    expected_blocks = set(range(max(transformer_blocks) + 1))
    if transformer_blocks != expected_blocks:
        raise ValueError(
            f"Transformer REPA adapter blocks must be contiguous, got {sorted(transformer_blocks)}"
        )
    if transformer_output.dim() != 2:
        raise ValueError("Transformer REPA output weight must be a matrix")
    if transformer_mask.dim() != 3 or tuple(transformer_mask.shape[:2]) != (1, 1):
        raise ValueError(
            "Transformer REPA mask token must have shape (1, 1, hidden_size)"
        )
    if transformer_output.shape[1] != transformer_mask.shape[2]:
        raise ValueError("Transformer REPA output and mask-token dimensions do not match")
    return (
        "transformer",
        int(transformer_output.shape[0]),
        len(transformer_blocks),
        None,
    )


def infer_repa_module_config(
    checkpoint_path: str,
) -> Optional[Tuple[str, int, int, Optional[int]]]:
    """Infer (type, target_dim, layer_count, MLP projector_dim)."""
    ckpt = _restore_checkpoint(checkpoint_path, mmap=True)
    params = ckpt.get("params") if isinstance(ckpt, dict) else None
    return _repa_module_config_from_params(params)


def infer_repa_projector_config(checkpoint_path: str) -> Optional[Tuple[int, int, Optional[int]]]:
    """Infer optional REPA (target_dim, layer_count, MLP projector_dim)."""
    module_config = infer_repa_module_config(checkpoint_path)
    if module_config is None:
        return None
    _, target_dim, layer_count, projector_dim = module_config
    return target_dim, layer_count, projector_dim


def infer_repa_target_dim(checkpoint_path: str) -> Optional[int]:
    """Infer optional REPA projector output dim from checkpoint weights."""
    projector_config = infer_repa_projector_config(checkpoint_path)
    if projector_config is None:
        return None
    return projector_config[0]


def _reg_projection_config_from_params(params):
    if not params:
        return None, None
    input_weight = params.get("reg_input_proj.weight")
    separate_output = params.get("final_layer.reg_linear.weight")
    shared_output = params.get("reg_output_proj.weight")
    output_weights = [
        ("separate", separate_output),
        ("shared_text", shared_output),
    ]
    present_outputs = [(name, weight) for name, weight in output_weights if weight is not None]
    if input_weight is None and not present_outputs:
        return None, None
    if input_weight is None or len(present_outputs) != 1:
        raise ValueError("REG checkpoint must contain both input and output projection weights")
    topology, output_weight = present_outputs[0]
    if input_weight.dim() != 2 or output_weight.dim() != 2:
        raise ValueError("REG projection weights must be matrices")
    if tuple(input_weight.shape) != tuple(reversed(output_weight.shape)):
        raise ValueError(f"REG projection shapes are inconsistent ({tuple(input_weight.shape)} vs {tuple(output_weight.shape)})")
    return int(input_weight.shape[1]), topology


def _reg_target_dim_from_params(params) -> Optional[int]:
    return _reg_projection_config_from_params(params)[0]


def _load_checkpoint_params(checkpoint_path: str):
    ckpt = _restore_checkpoint(checkpoint_path, mmap=True)
    return ckpt.get("params") if isinstance(ckpt, dict) else None


def infer_reg_projection_config(checkpoint_path: str):
    """Infer optional REG dimension and projection topology from checkpoint weights."""
    return _reg_projection_config_from_params(_load_checkpoint_params(checkpoint_path))


def infer_reg_target_dim(checkpoint_path: str) -> Optional[int]:
    """Infer the optional heterogeneous REG dimension from checkpoint projections."""
    return infer_reg_projection_config(checkpoint_path)[0]


def infer_reg_projection_topology(checkpoint_path: str) -> Optional[str]:
    """Infer the optional REG projection topology from checkpoint weights."""
    return infer_reg_projection_config(checkpoint_path)[1]


def _olt_enabled_from_params(params) -> bool:
    if not params or "olt_token" not in params:
        return False
    token = params["olt_token"]
    if token.dim() != 3 or tuple(token.shape[:2]) != (1, 1) or token.shape[2] <= 0:
        raise ValueError(f"OLT token must have shape (1, 1, hidden_size), got {tuple(token.shape)}")
    return True


def infer_olt_enabled(checkpoint_path: str) -> bool:
    """Infer whether a checkpoint contains the learned OLT parameter."""
    ckpt = _restore_checkpoint(checkpoint_path, mmap=True)
    params = ckpt.get("params") if isinstance(ckpt, dict) else None
    return _olt_enabled_from_params(params)


def _validate_checkpoint(ckpt: Any):
    if ckpt is None:
        raise ValueError("checkpoint restore returned None")
    required_keys = ("params", "opt_state", "step", "epoch")
    missing_keys = [key for key in required_keys if key not in ckpt]
    if missing_keys:
        raise ValueError(f"checkpoint restore missing keys: {missing_keys}")


def _validate_weights_checkpoint(ckpt: Any):
    if ckpt is None:
        raise ValueError("checkpoint restore returned None")
    if "params" not in ckpt:
        raise ValueError("checkpoint restore missing keys: ['params']")


def _load_valid_checkpoint(checkpoint_path: str, validate_fn, *, mmap: bool = False) -> Tuple[Any, str]:
    ckpt = _restore_checkpoint(checkpoint_path, mmap=mmap)
    validate_fn(ckpt)
    return ckpt, "local"


def load_checkpoint_step(checkpoint_path: str) -> int:
    ckpt, _ = _load_valid_checkpoint(checkpoint_path, _validate_checkpoint, mmap=True,)
    return int(ckpt["step"])


def _checkpoint_optimizer_step(ckpt: Any) -> int:
    optimizer_step = ckpt.get("optimizer_step")
    if optimizer_step is None:
        scheduler = ckpt.get("lr_scheduler")
        if isinstance(scheduler, dict):
            optimizer_step = scheduler.get("last_epoch")
    if isinstance(optimizer_step, bool) or not isinstance(optimizer_step, int) or optimizer_step < 0:
        raise ValueError("checkpoint does not contain a valid optimizer step")
    return optimizer_step


def load_checkpoint_optimizer_step(checkpoint_path: str) -> int:
    ckpt, _ = _load_valid_checkpoint(checkpoint_path, _validate_checkpoint, mmap=True,)
    return _checkpoint_optimizer_step(ckpt)


def load_checkpoint(checkpoint_path: str, state, strict_ema_decay_match: bool = True) -> Tuple[Any, int]:
    """Load an ELF checkpoint.

    Uses an existing local path first; otherwise tries HF and then local fallback.
    """
    log_for_0(f"Loading ELF checkpoint from {checkpoint_path}...")
    ckpt, loaded_from = _load_valid_checkpoint(checkpoint_path, _validate_checkpoint)

    log_for_0(f"Loaded checkpoint keys: {list(ckpt.keys())}")

    inner_model = unwrap_model(state.model)
    inner_model.load_state_dict(ckpt["params"])
    device_map = {n: p.device for n, p in inner_model.named_parameters()}
    for n, b in inner_model.named_buffers():
        device_map.setdefault(n, b.device)
    fallback_device = next(iter(device_map.values()), torch.device("cpu"))
    _load_ema_params(ckpt, state, device_map, fallback_device, strict_ema_decay_match)
    state.optimizer.load_state_dict(ckpt["opt_state"])
    if state.lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
        state.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    state.step = int(ckpt["step"])
    state.epoch = int(ckpt["epoch"])
    state.sample_offset_in_epoch = ckpt.get("sample_offset_in_epoch")
    state.group_by_length_active = bool(ckpt.get("group_by_length_active", False))
    if ckpt.get("grad_accum_buffers"):
        buffers = ckpt["grad_accum_buffers"]
        state.grad_accum_buffers = {}
        param_ids = []
        for name, param in inner_model.named_parameters():
            if not param.requires_grad:
                continue
            param_ids.append(id(param))
            saved = buffers.get(name)
            state.grad_accum_buffers[id(param)] = (
                saved.to(device=param.device, dtype=param.dtype)
                if saved is not None else torch.zeros_like(param)
            )
        state.grad_accum_param_ids = tuple(param_ids)

    step = int(ckpt["step"])
    log_for_0(f"Loaded {loaded_from} checkpoint from step {step} (epoch {state.epoch})")
    return state, step


def _is_reg_projection_key(name: str) -> bool:
    return (
        name.startswith("reg_input_proj.")
        or name.startswith("reg_output_proj.")
        or name.startswith("reg_self_cond_proj.")
        or name.startswith("final_layer.reg_linear.")
    )


def _is_optional_model_key(name: str) -> bool:
    return name.startswith("repa_projector.") or _is_reg_projection_key(name) or name == "olt_token"


def _merge_repa_compatible_state(source, target, label: str, *, load_reg: bool = True):
    source_keys = set(source.keys())
    target_keys = set(target.keys())
    missing = sorted(k for k in target_keys - source_keys if not _is_optional_model_key(k))
    unexpected = sorted(k for k in source_keys - target_keys if not _is_optional_model_key(k))
    mismatched = sorted(
        (k, tuple(source[k].shape), tuple(target[k].shape))
        for k in source_keys & target_keys
        if tuple(source[k].shape) != tuple(target[k].shape) and not _is_optional_model_key(k)
    )
    if missing or unexpected or mismatched:
        parts = []
        if missing:
            parts.append(f"missing non-optional keys: {missing}")
        if unexpected:
            parts.append(f"unexpected non-optional keys: {unexpected}")
        if mismatched:
            parts.append(f"shape mismatches: {mismatched}")
        raise ValueError(f"Weight-only checkpoint {label} is incompatible: " + "; ".join(parts))

    merged = target.copy()
    for key in source_keys & target_keys:
        if tuple(source[key].shape) == tuple(target[key].shape) and (
            load_reg or not _is_reg_projection_key(key)
        ):
            merged[key] = source[key]
    return merged


def load_weights_only_checkpoint(checkpoint_path: str, state, *, source_ema_decay=None):
    """Initialize model + EMA weights from a checkpoint without restoring train state.

    When source_ema_decay is set, use that source EMA for the model and every target EMA.
    """
    source_ema_key = None
    if source_ema_decay is not None:
        source_ema_decay = float(source_ema_decay)
        if not 0.0 < source_ema_decay < 1.0:
            raise ValueError("resume_only_weights_ema must be in (0, 1)")
        source_ema_key = ema_decay_key(source_ema_decay)

    log_for_0(f"Loading ELF weights from {checkpoint_path}...")
    ckpt, loaded_from = _load_valid_checkpoint(checkpoint_path, _validate_weights_checkpoint)
    log_for_0(f"Loaded checkpoint keys: {list(ckpt.keys())}")
    ema_src = ckpt.get("ema_params1")
    if ema_src is None:
        raise ValueError("resume_only_weights requires ema_params1")
    selected_source_ema = None
    if is_multi_ema_state(ema_src):
        source_emas = _normalize_nested_ema(ema_src)
        declared_decays = ckpt.get("ema_decays1")
        if declared_decays is not None:
            declared_keys = {ema_decay_key(float(decay)) for decay in declared_decays}
            if declared_keys != set(source_emas):
                raise ValueError(
                    "Weight-only checkpoint ema_decays1 does not match nested ema_params1 keys"
                )
        if source_ema_key is not None:
            if source_ema_key not in source_emas:
                raise ValueError(f"Weight-only checkpoint has no EMA decay {source_ema_key}; available EMA decays: {sorted(source_emas)}")
            selected_source_ema = source_emas[source_ema_key]
            source_emas = {key: selected_source_ema for key in state.ema_params1}
            log_for_0(f"Initializing model and all target EMAs from checkpoint EMA {source_ema_key}")
        else:
            missing_decays = sorted(set(state.ema_params1) - set(source_emas))
            if missing_decays:
                raise ValueError(f"Weight-only checkpoint is missing configured EMA decays: {missing_decays}")
    else:
        if source_ema_key is not None:
            raise ValueError("resume_only_weights_ema requires a checkpoint with nested EMA decays")
        source_emas = {key: ema_src for key in state.ema_params1}

    params_reg_config = _reg_projection_config_from_params(ckpt["params"])
    params_reg_dim = params_reg_config[0]
    params_olt_enabled = _olt_enabled_from_params(ckpt["params"])
    if params_reg_dim is not None and params_olt_enabled:
        raise ValueError("Weight-only checkpoint cannot contain both OLT and REG parameters")
    for key, source_ema in source_emas.items():
        ema_reg_config = _reg_projection_config_from_params(source_ema)
        if params_reg_config != ema_reg_config:
            raise ValueError(f"Weight-only checkpoint REG projections differ between params and ema_params1[{key}] ({params_reg_config} vs {ema_reg_config})")
        if params_reg_dim is not None:
            reg_names = {name for name in ckpt["params"] if _is_reg_projection_key(name)}
            ema_reg_names = {name for name in source_ema if _is_reg_projection_key(name)}
            if reg_names != ema_reg_names:
                raise ValueError(f"Weight-only checkpoint REG parameter keys differ between params and ema_params1[{key}]")
            for name in reg_names:
                if tuple(ckpt["params"][name].shape) != tuple(source_ema[name].shape):
                    raise ValueError(f"Weight-only checkpoint {name} shape differs between params and ema_params1[{key}]")
        ema_olt_enabled = _olt_enabled_from_params(source_ema)
        if params_olt_enabled != ema_olt_enabled:
            raise ValueError(
                f"Weight-only checkpoint OLT mode differs between params and ema_params1[{key}]"
            )
        if params_olt_enabled and tuple(ckpt["params"]["olt_token"].shape) != tuple(source_ema["olt_token"].shape):
            raise ValueError(
                f"Weight-only checkpoint olt_token shape differs between params and ema_params1[{key}]"
            )

    inner_model = unwrap_model(state.model)
    model_state = inner_model.state_dict()
    source_model_state = ckpt["params"]
    if selected_source_ema is not None:
        source_model_state = source_model_state.copy()
        source_model_state.update(selected_source_ema)
    target_reg_config = _reg_projection_config_from_params(model_state)
    target_olt_enabled = _olt_enabled_from_params(model_state)
    if (
        params_olt_enabled
        and target_olt_enabled
        and tuple(ckpt["params"]["olt_token"].shape) != tuple(model_state["olt_token"].shape)
    ):
        raise ValueError("Weight-only checkpoint olt_token shape does not match target model")
    load_reg = params_reg_config == target_reg_config
    merged_model_state = _merge_repa_compatible_state(
        source_model_state, model_state, "params", load_reg=load_reg,
    )
    inner_model.load_state_dict(merged_model_state)

    device_map = {n: p.device for n, p in inner_model.named_parameters()}
    fallback_device = next(iter(device_map.values()), torch.device("cpu"))
    state.ema_params1 = {
        key: _to_device_state(
            _merge_repa_compatible_state(
                source_emas[key], target, f"ema_params1[{key}]", load_reg=load_reg,
            ),
            device_map, fallback_device, copy=selected_source_ema is not None,
        )
        for key, target in state.ema_params1.items()
    }

    log_for_0(f"Loaded {loaded_from} checkpoint weights only")
    return state
