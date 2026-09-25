"""Frozen token-level text encoders used by ELF."""

from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from modules.t5_encoder import _register_t5_legacy_3d_mask_hooks
from utils.logging_utils import log_for_0


def feature_standardize(hidden: torch.Tensor) -> torch.Tensor:
    """Standardize each token or pooled vector across its feature dimension."""
    hidden = hidden.float()
    mean = hidden.mean(dim=-1, keepdim=True)
    var = hidden.var(dim=-1, keepdim=True)
    return (hidden - mean) / torch.sqrt(var + 1e-6)


def encoder_family_from_model_type(model_type: str) -> str:
    model_type = str(model_type).lower()
    if "t5" in model_type:
        return "t5"
    if model_type.startswith("qwen"):
        return model_type
    return model_type


def _hidden_size(hf_config) -> int:
    for attr in ("hidden_size", "d_model"):
        value = int(getattr(hf_config, attr, 0) or 0)
        if value > 0:
            return value
    raise ValueError("Could not determine text encoder hidden size")


def _selected_dim(raw_dim: int, dim: Optional[int]) -> int:
    if dim is None:
        return int(raw_dim)
    dim = int(dim)
    if dim <= 0 or dim > int(raw_dim):
        raise ValueError(f"encoder_dim must be in [1, {raw_dim}], got {dim}")
    return dim


def _canonical_layer(layer: Optional[int]) -> int:
    return -1 if layer is None else int(layer)


def build_causal_segment_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a (B, L, L) segment mask into a 4D additive causal mask."""
    if attention_mask.dim() != 3:
        raise ValueError(f"Expected a 3D attention mask, got shape {tuple(attention_mask.shape)}")
    if attention_mask.shape[1] != attention_mask.shape[2]:
        raise ValueError(f"Expected a square attention mask, got shape {tuple(attention_mask.shape)}")

    batch, length, _ = attention_mask.shape
    allowed = attention_mask.to(torch.bool)
    causal = torch.ones((length, length), dtype=torch.bool, device=attention_mask.device).tril()
    allowed = allowed & causal.unsqueeze(0)
    allowed = allowed[:, None, :, :]
    blocked = torch.full((batch, 1, length, length), torch.finfo(dtype).min, dtype=dtype, device=attention_mask.device)
    return torch.where(allowed, torch.zeros_like(blocked), blocked)


def bool_mask_to_additive(attention_mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if attention_mask.dtype != torch.bool:
        return attention_mask
    blocked = torch.full(attention_mask.shape, torch.finfo(dtype).min, dtype=dtype, device=attention_mask.device)
    return torch.where(attention_mask, torch.zeros_like(blocked), blocked)


class TextEncoderConfig:
    """Resolved text-encoder metadata after optional dimension selection."""

    def __init__(
        self,
        model_name: str,
        dtype: Any,
        *,
        model_type: str,
        vocab_size: int,
        raw_d_model: int,
        d_model: int,
        num_layers: int,
        encoder_layer: Optional[int],
        family: str,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.model_type = model_type
        self.vocab_size = int(vocab_size)
        self.raw_d_model = int(raw_d_model)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.encoder_layer = encoder_layer
        self.family = family
        self.applies_own_normalization = family != "t5"

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        dtype: Any = torch.float32,
        *,
        encoder_dim: Optional[int] = None,
        encoder_layer: Optional[int] = None,
    ) -> "TextEncoderConfig":
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name)
        return cls.from_hf_config(
            model_name, hf_config, dtype=dtype,
            encoder_dim=encoder_dim, encoder_layer=encoder_layer,
        )

    @classmethod
    def from_hf_config(
        cls,
        model_name: str,
        hf_config,
        *,
        dtype: Any = torch.float32,
        encoder_dim: Optional[int] = None,
        encoder_layer: Optional[int] = None,
    ) -> "TextEncoderConfig":
        model_type = str(getattr(hf_config, "model_type", ""))
        family = encoder_family_from_model_type(model_type)
        if family not in {"t5", "qwen3"}:
            raise ValueError(
                f"Unsupported text encoder family {family!r}; only T5 and Qwen3 are supported"
            )
        if family == "t5" and encoder_layer is not None:
            raise ValueError("encoder_layer is not supported for T5 encoders")

        raw_d_model = _hidden_size(hf_config)
        return cls(
            model_name,
            dtype,
            model_type=model_type,
            vocab_size=int(getattr(hf_config, "vocab_size", 0) or 0),
            raw_d_model=raw_d_model,
            d_model=_selected_dim(raw_d_model, encoder_dim),
            num_layers=int(getattr(hf_config, "num_hidden_layers", getattr(hf_config, "num_layers", 0)) or 0),
            encoder_layer=None if encoder_layer is None else int(encoder_layer),
            family=family,
        )

    def clone_with(
        self,
        *,
        encoder_dim: Optional[int] = None,
        encoder_layer: Optional[int] = None,
    ) -> "TextEncoderConfig":
        if self.family == "t5" and encoder_layer is not None:
            raise ValueError("encoder_layer is not supported for T5 encoders")
        return TextEncoderConfig(
            self.model_name,
            self.dtype,
            model_type=self.model_type,
            vocab_size=self.vocab_size,
            raw_d_model=self.raw_d_model,
            d_model=_selected_dim(self.raw_d_model, encoder_dim),
            num_layers=self.num_layers,
            encoder_layer=None if encoder_layer is None else int(encoder_layer),
            family=self.family,
        )


class FrozenTextEncoder(nn.Module):
    """A frozen token-level encoder with repo-standard masking and normalization."""

    def __init__(self, config: TextEncoderConfig, *, pretrained: bool = True, model: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        if model is not None:
            self.model = model
            return

        if config.family == "t5":
            from transformers import T5Config, T5EncoderModel

            if pretrained:
                self.model = T5EncoderModel.from_pretrained(config.model_name)
            else:
                self.model = T5EncoderModel(T5Config.from_pretrained(config.model_name))
            _register_t5_legacy_3d_mask_hooks(self.model)
        else:
            from transformers import AutoConfig, AutoModel

            if pretrained:
                self.model = AutoModel.from_pretrained(config.model_name)
            else:
                self.model = AutoModel.from_config(AutoConfig.from_pretrained(config.model_name))

    def make_view(
        self,
        *,
        encoder_dim: Optional[int] = None,
        encoder_layer: Optional[int] = None,
    ) -> "FrozenTextEncoder":
        return FrozenTextEncoder(
            self.config.clone_with(encoder_dim=encoder_dim, encoder_layer=encoder_layer),
            model=self.model,
        )

    def _mask_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _prepare_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if attention_mask is None or self.config.family == "t5":
            return attention_mask
        if attention_mask.dim() == 3:
            return build_causal_segment_attention_mask(attention_mask, dtype=self._mask_dtype())
        if attention_mask.dim() == 4:
            return bool_mask_to_additive(attention_mask, dtype=self._mask_dtype())
        return attention_mask

    def _select_hidden(
        self, out, *, encoder_dim: Optional[int], encoder_layer: Optional[int],
        normalize: bool,
    ) -> torch.Tensor:
        layer = _canonical_layer(encoder_layer)
        truncated_layer = getattr(self.model, "_qwen3_truncated_layer", None)
        if truncated_layer is not None and layer == -1:
            raise ValueError("A truncated Qwen3 model cannot provide the final-layer view")
        if layer == -1 or layer == truncated_layer:
            hidden = out.last_hidden_state
        else:
            hidden_states = getattr(out, "hidden_states", None)
            if hidden_states is None:
                raise ValueError("Encoder did not return hidden_states")
            try:
                hidden = hidden_states[layer]
            except IndexError as exc:
                raise ValueError(f"encoder_layer={layer} is out of range for {len(hidden_states)} hidden states") from exc
        selected_dim = _selected_dim(self.config.raw_d_model, encoder_dim)
        hidden = hidden[..., :selected_dim]
        if normalize and self.config.applies_own_normalization:
            hidden = feature_standardize(hidden)
        return hidden

    def encode_views(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        views: Mapping[str, Tuple[Optional[int], Optional[int]]],
        *,
        deterministic: bool = True,
        normalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Encode once and select multiple `(dimension, layer)` views."""
        if not views:
            raise ValueError("At least one encoder view is required")
        if self.config.family == "t5":
            explicit_layers = [layer for _, layer in views.values() if layer is not None]
            if explicit_layers:
                raise ValueError("encoder_layer is not supported for T5 encoders")

        prepared_mask = self._prepare_attention_mask(attention_mask)
        truncated_layer = getattr(self.model, "_qwen3_truncated_layer", None)
        canonical_layers = [_canonical_layer(layer) for _, layer in views.values()]
        if truncated_layer is not None:
            if -1 in canonical_layers:
                raise ValueError("A truncated Qwen3 model cannot provide the final-layer view")
            if any(layer > int(truncated_layer) for layer in canonical_layers):
                raise ValueError(
                    f"Truncated Qwen3 model ends at layer {truncated_layer}, "
                    f"requested {canonical_layers}"
                )
        output_hidden_states = any(
            layer not in {-1, truncated_layer} for layer in canonical_layers
        )
        was_training = self.model.training
        if deterministic:
            self.model.eval()
        try:
            if self.config.family == "t5":
                out = self.model(input_ids=input_ids, attention_mask=prepared_mask)
            else:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=prepared_mask,
                    output_hidden_states=output_hidden_states,
                    use_cache=False,
                )
        finally:
            if not deterministic and was_training:
                self.model.train()

        return {
            name: self._select_hidden(
                out, encoder_dim=dim, encoder_layer=layer, normalize=normalize,
            )
            for name, (dim, layer) in views.items()
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        return self.encode_views(
            input_ids,
            attention_mask,
            {"default": (self.config.d_model, self.config.encoder_layer)},
            deterministic=deterministic,
        )["default"]


def get_encoder(
    model_name: str,
    dtype: Any,
    *,
    encoder_dim: Optional[int] = None,
    encoder_layer: Optional[int] = None,
    pretrained: bool = True,
):
    """Return `(config, model)`. Weights are downloaded on first use."""
    log_for_0(f"Loading Text Encoder: {model_name}...")
    config = TextEncoderConfig.from_pretrained(
        model_name, dtype=dtype, encoder_dim=encoder_dim, encoder_layer=encoder_layer,
    )
    model = FrozenTextEncoder(config, pretrained=pretrained)
    if dtype is not None:
        model = model.to(dtype)
    return config, model
