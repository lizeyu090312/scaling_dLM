import yaml
import os
import warnings


EMA_SCALAR_WARNING = (
    "WARNING: ema_decay1 was provided as a single value instead of a list. "
    "Please use list syntax, e.g. ema_decay1: [0.9999]. "
    "Scalar ema_decay1 is accepted only for legacy configs."
)

REMOVED_CONFIG_FIELDS = {
    "repa_align_condition_tokens",
    "repa_self_flow_target_depth",
    "repa_self_flow_t_max",
    "repa_self_flow_ema_decoder_true",
    "repa_warmup_steps",
}


class SamplingConfig:
    """Sampling configuration for generation."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        fields = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        for k in self.__class__.__annotations__:
            if k not in fields:
                fields[k] = getattr(self, k, None)
        items = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"SamplingConfig({items})"

    sampling_method: str = "ode"
    num_sampling_steps: list = [50]
    cfgs: list = [1]
    self_cond_cfg_scales: list = [1.0]
    time_schedule: str = "logit_normal"  # 'logit_normal' or 'uniform'
    sde_gamma: float = 0.0  # Per-step SDE churn fraction; 0.0 -> pure ODE. Used when sampling_method == "sde".


# ============================================
# Configuration
# ============================================
class Config:
    # Dataset
    data_path: str = None
    eval_data_path: str | list = None
    latent_cache_config_path: str = None
    max_length: int = 128
    max_input_length: int = None  # Max length for conditioning input (e.g., prompt or encoder input); None = no limit
    pad_token: str = "pad"  # "pad" or "eos" - which token to use for padding

    # Tokenizer
    tokenizer_name: str = None  # Defaults to encoder_model_name if not set

    # Encoder
    encoder_model_name: str = "t5-small"
    encoder_checkpoint: str = None
    encoder_dim: int = None
    encoder_layer: int = None
    latent_mean: float = 0.0
    latent_std: float = 1.0

    # Model architecture
    model: str = "ELF-B"
    bottleneck_dim: int = 128  # Bottleneck dimension for text projection
    num_time_tokens: int = 4  # Number of in-context time conditioning tokens
    num_self_cond_cfg_tokens: int = 4  # Number of in-context self-cond CFG tokens
    num_model_mode_tokens: int = 4  # If > 0, prepend learnable model-mode tokens that signal decoding mode
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    # Denoiser objective
    denoiser_p_mean: float = 0.8
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 1.0
    t_eps: float = 5e-2
    time_schedule: str = "logit_normal"  # 'logit_normal' or 'uniform'

    # Decoder objective
    decoder_prob: float = 0.5  # Probability of decoder (CE) step vs denoiser (L2) step
    decoder_noise_scale: float = 1.0  # Scale of noise in logit-normal-noised latent for CE branch
    decoder_p_mean: float = 0.8  # Mean for logit-normal noise schedule in decoder objective
    decoder_p_std: float = 0.8  # Std for logit-normal noise schedule in decoder objective

    # Conditioning / CFG
    label_drop_prob: float = 0.0
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    # Training (optimizer + schedule)
    epochs: int = 200
    warmup_epochs: float = None
    warmup_steps: int = 5000
    batch_size: int = None
    global_batch_size: int = 512
    lr: float = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    optimizer: str = "muon"  # "adamw" or "muon"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1  # Inferred in train.py from global_batch_size, batch_size, and world size.
    group_by_length: bool = False
    use_bf16: bool = True  # Use CUDA BF16 autocast for training/eval forward passes.
    use_compile: bool = False  # Wrap the eval/sampling model in torch.compile.
    use_model_attention_mask: bool = False  # Pass valid-token masks into ELF attention; default preserves legacy behavior.
    gradient_checkpointing: bool = False  # Save activation memory by recomputing ELF blocks during backward.

    # REPA auxiliary alignment
    repa_enabled: bool = False
    repa_depth: int = 4
    repa_t_min: float = 0.0
    repa_t_max: float = 1.0
    repa_projector_dim: int = 2048
    repa_projector_layers: int = 3
    repa_projector_type: str = "mlp"
    repa_adapter_layers: int = 1
    repa_adapter_attention_mode: str = "full"
    repa_mask_ratio: float = 0.0
    repa_strength: float = 1.0
    repa_end_fraction: float = 1.0
    repa_align_decoder_rows: bool = True
    repa_loss_type: str = "cos_sim"
    repa_teacher_model_name: str = "t5-small"
    repa_teacher_layer: int = None
    repa_teacher_dim: int = None
    repa_teacher_input_format: str = "raw"
    repa_prompt_loss_weight: float = 0.0
    repa_response_loss_weight: float = 1.0
    repa_shift: int = 0
    repa_reg_target_source: str = "repa"  # "repa" or "reg"

    # REG semantic-token denoising
    olt_enabled: bool = False
    reg_enabled: bool = False
    reg_projection_topology: str = "separate"  # "separate" or "shared_text"
    reg_loss_weight: float = 0.03
    reg_teacher_layer: int = None
    reg_teacher_dim: int = None
    reg_teacher_pooling: str = "eot"

    # EMA
    ema_decay1: list = [0.9999]
    ema_warmup_updates: int = 10_000

    # Sampling
    sampling_configs_path: str = None
    training_generation_ema: str = None
    # Sampling configs sweep (list of SamplingConfig objects, loaded from YAML)
    sampling_configs: list = [SamplingConfig()]
    generation_batch_size: int = 32
    num_samples: int = 100
    truncate_generation: bool = True  # Limit conditional outputs to max_length - max_input_length.

    # PPL Evaluation
    online_eval: bool = True  # Enable PPL evaluation for generated samples
    conditional_eval_metric: str = "bleu_rouge"  # Also: mmlu_accuracy, mmlu_rationale_accuracy, gsm8k_accuracy, math_accuracy
    eval_ppl_model: str = "gpt2-large"  # Model for PPL evaluation
    eval_ppl_batch_size: int = 64  # Batch size for PPL evaluation (adjusted to be divisible by device count)
    eval_ppl_max_length: int = 1024  # Max sequence length for PPL evaluation

    # Logging & Checkpointing
    log_freq: int = 100
    eval_freq: float = 10.0
    save_freq: float = 100  # Can be fractional (e.g., 0.1 for saving every 0.1 epoch)

    # Output
    output_dir: str = "./output_dir"
    resume: str = None
    resume_only_weights: str = None
    resume_only_weights_ema: float = None  # Source EMA for the model and all target EMAs.

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "ELF"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_tag: str = None
    wandb_resume: str = "allow"

    # Misc
    seed: int = 0
    num_workers: int = 8


def load_config_from_yaml(path: str) -> Config:
    """Load a YAML config and override defaults in Config."""
    config = Config()
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration not found: {path}")

    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}

    removed_keys = sorted(set(cfg_dict) & REMOVED_CONFIG_FIELDS)
    if removed_keys:
        raise ValueError(f"Removed config field(s): {', '.join(removed_keys)}")

    for key, value in cfg_dict.items():
        if key == "sampling_configs":
            continue  # handled below
        if hasattr(config, key):
            setattr(config, key, value)

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    config.ema_decay1 = normalize_ema_decay1(config.ema_decay1)
    return config


def config_field_is_explicit(path: str, overrides: list, field_name: str) -> bool:
    config_values = {}
    if path and os.path.isfile(path):
        with open(path, "r") as handle:
            config_values = yaml.safe_load(handle) or {}
    override_fields = {
        value.split("=", 1)[0].strip()
        for value in (overrides or [])
        if "=" in value
    }
    return field_name in config_values or field_name in override_fields


def normalize_ema_decay1(value) -> list:
    if isinstance(value, str):
        value = value.strip()
        if "," in value and not value.startswith("["):
            values = [v.strip() for v in value.split(",")]
        else:
            loaded = yaml.safe_load(value)
            if isinstance(loaded, list):
                values = loaded
            else:
                warnings.warn(EMA_SCALAR_WARNING, UserWarning, stacklevel=2)
                values = [loaded]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        warnings.warn(EMA_SCALAR_WARNING, UserWarning, stacklevel=2)
        values = [value]
    return [float(v) for v in values]


def apply_config_overrides(config: Config, overrides: list) -> Config:
    """Apply command-line config overrides to a Config object.

    Args:
        config: Config object to modify
        overrides: List of strings in format "field_name=value"

    Returns:
        Modified config object
    """
    if not overrides:
        return config

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'field_name=value'")

        field_name, value_str = override.split("=", 1)
        field_name = field_name.strip()
        value_str = value_str.strip()

        if not hasattr(config, field_name):
            raise ValueError(f"Config has no field named '{field_name}'")

        if field_name == "ema_decay1":
            setattr(config, field_name, normalize_ema_decay1(value_str))
            continue

        original_value = getattr(config, field_name)
        original_type = type(original_value)

        # Allow setting a field back to None
        if value_str.lower() == "none":
            setattr(config, field_name, None)
            continue

        if original_value is None:
            # Use type annotation to infer the intended type
            annotated_type = config.__annotations__.get(field_name)
            if annotated_type == int:
                converted_value = int(value_str)
            elif annotated_type == float:
                converted_value = float(value_str)
            elif annotated_type == bool:
                converted_value = value_str.lower() in ("true", "1", "yes")
            else:
                converted_value = value_str
        elif original_type == bool:
            converted_value = value_str.lower() in ("true", "1", "yes")
        elif original_type == int:
            converted_value = int(value_str)
        elif original_type == float:
            converted_value = float(value_str)
        elif original_type == str:
            converted_value = value_str
        else:
            converted_value = value_str

        setattr(config, field_name, converted_value)

    return config


def load_sampling_configs(sampling_configs_path: str):
    """Return sampling configs, loading from sampling_configs_path if set."""
    with open(sampling_configs_path, "r") as f:
        entries = yaml.safe_load(f)
    return [SamplingConfig(**entry) for entry in entries]
