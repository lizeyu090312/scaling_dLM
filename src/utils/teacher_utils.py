"""Teacher tokenization, compatibility checks, and aligned target extraction."""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch

from modules.text_encoder import feature_standardize


EOT_TOKEN = "<|endoftext|>"
EOT_TOKEN_ID = 151643
TEACHER_INPUT_FORMATS = {"raw", "qwen3_chat"}


def _vocab_hash(vocab: dict) -> str:
    payload = json.dumps(sorted((str(token), int(idx)) for token, idx in vocab.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_hash(value) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _special_tokens(tokenizer) -> dict:
    return {
        key: [str(token) for token in value] if isinstance(value, (list, tuple)) else str(value)
        for key, value in tokenizer.special_tokens_map.items()
    }


def _post_processor_state(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    processor = getattr(backend, "post_processor", None)
    if processor is None:
        return ""
    state = processor.__getstate__()
    if isinstance(state, bytes):
        return state.decode("utf-8")
    return str(state)


def _tokenizer_model_state(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None)
    if model is None:
        return ""
    state = model.__getstate__()
    if isinstance(state, bytes):
        return state.decode("utf-8")
    return str(state)


def _embedding_rows(encoder) -> int:
    model = getattr(encoder, "model", encoder)
    embeddings = model.get_input_embeddings()
    rows = int(getattr(embeddings, "num_embeddings", 0) or 0)
    if rows <= 0:
        raise ValueError("Text encoder has no valid input embedding table")
    return rows


def _mapping_mismatches(left: dict, right: dict, limit: int = 10) -> list:
    mismatches = []
    for token in sorted(set(left) | set(right)):
        left_id, right_id = left.get(token), right.get(token)
        if left_id != right_id:
            mismatches.append({"token": token, "left_id": left_id, "right_id": right_id})
            if len(mismatches) == limit:
                break
    return mismatches


def _base_vocab(tokenizer) -> dict:
    added = tokenizer.get_added_vocab()
    return {
        token: token_id for token, token_id in tokenizer.get_vocab().items()
        if token not in added
    }


def _consumer_compatibility(dataset_tokenizer, consumer_tokenizer, consumer, family: str) -> dict:
    dataset_vocab = dataset_tokenizer.get_vocab()
    consumer_vocab = consumer_tokenizer.get_vocab()
    dataset_added = dataset_tokenizer.get_added_vocab()
    consumer_added = consumer_tokenizer.get_added_vocab()
    exact_mapping = dataset_vocab == consumer_vocab and dataset_added == consumer_added
    rows = _embedding_rows(consumer)
    consumer_ids_fit = max(consumer_vocab.values(), default=-1) < rows

    incompatible_ids = sorted({
        int(token_id) for token, token_id in dataset_added.items()
        if consumer_vocab.get(token) != token_id
    })
    incompatible_set = set(incompatible_ids)
    compatible_dataset_ids = [
        int(token_id) for token, token_id in dataset_vocab.items()
        if int(token_id) not in incompatible_set and consumer_vocab.get(token) == token_id
    ]
    dataset_ids_fit = max(compatible_dataset_ids, default=-1) < rows

    dataset_model_state = _tokenizer_model_state(dataset_tokenizer)
    consumer_model_state = _tokenizer_model_state(consumer_tokenizer)
    base_vocab_equal = _base_vocab(dataset_tokenizer) == _base_vocab(consumer_tokenizer)
    bpe_model_equal = bool(dataset_model_state) and dataset_model_state == consumer_model_state
    qwen_special_ids_valid = (
        dataset_tokenizer.convert_tokens_to_ids(EOT_TOKEN) == EOT_TOKEN_ID
        and consumer_tokenizer.convert_tokens_to_ids(EOT_TOKEN) == EOT_TOKEN_ID
        and dataset_tokenizer.pad_token_id == EOT_TOKEN_ID
        and consumer_tokenizer.pad_token_id == EOT_TOKEN_ID
    )

    if family == "qwen3":
        compatible = (
            base_vocab_equal and bpe_model_equal and qwen_special_ids_valid
            and consumer_ids_fit and dataset_ids_fit
        )
    else:
        compatible = exact_mapping and consumer_ids_fit and dataset_ids_fit

    return {
        "compatible": compatible,
        "exact_mapping": exact_mapping,
        "base_vocab_equal": base_vocab_equal,
        "bpe_model_equal": bpe_model_equal,
        "dataset_bpe_model_sha256": _text_hash(dataset_model_state),
        "consumer_bpe_model_sha256": _text_hash(consumer_model_state),
        "qwen_special_ids_valid": qwen_special_ids_valid,
        "consumer_embedding_rows": rows,
        "consumer_ids_fit_embeddings": consumer_ids_fit,
        "compatible_dataset_ids_fit_embeddings": dataset_ids_fit,
        "incompatible_dataset_added_token_ids": incompatible_ids,
    }


def validate_input_ids_compatibility(
    input_ids: torch.Tensor, incompatible_token_ids, consumer_name: str,
) -> None:
    incompatible_token_ids = tuple(int(token_id) for token_id in incompatible_token_ids)
    if not incompatible_token_ids or not input_ids.numel():
        return
    candidates = torch.tensor(
        incompatible_token_ids, dtype=input_ids.dtype, device=input_ids.device,
    )
    bad = torch.isin(input_ids, candidates)
    if torch.any(bad):
        bad_ids = sorted(set(input_ids[bad].detach().cpu().tolist()))
        raise ValueError(
            f"Stored token IDs are incompatible with {consumer_name}: {bad_ids}"
        )


def build_auxiliary_content_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_ids=(),
) -> torch.Tensor:
    """Return valid source positions that are not tokenizer-declared specials."""
    content_mask = attention_mask.to(torch.bool)
    if not special_token_ids:
        return content_mask
    special_ids = torch.tensor(
        tuple(special_token_ids), dtype=input_ids.dtype, device=input_ids.device,
    )
    return content_mask & ~torch.isin(input_ids, special_ids)


def validate_tokenizer_compatibility(
    dataset_tokenizer,
    encoder_tokenizer,
    teacher_tokenizer,
    encoder,
    teacher,
    *,
    dataset_name: str,
    encoder_name: str,
    teacher_name: Optional[str],
    encoder_family: str,
    teacher_family: Optional[str],
    report_path: Optional[str] = None,
) -> dict:
    """Validate that stored token IDs retain their meaning for encoder and teacher."""
    dataset_vocab = dataset_tokenizer.get_vocab()
    encoder_vocab = encoder_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab() if teacher_tokenizer is not None else {}
    dataset_added = dataset_tokenizer.get_added_vocab()
    encoder_added = encoder_tokenizer.get_added_vocab()
    teacher_added = teacher_tokenizer.get_added_vocab() if teacher_tokenizer is not None else {}

    encoder_check = _consumer_compatibility(
        dataset_tokenizer, encoder_tokenizer, encoder, encoder_family,
    )
    teacher_check = None
    if teacher is not None:
        teacher_check = _consumer_compatibility(
            dataset_tokenizer, teacher_tokenizer, teacher, teacher_family,
        )
    encoder_rows = encoder_check["consumer_embedding_rows"]
    teacher_rows = teacher_check["consumer_embedding_rows"] if teacher_check is not None else None
    max_dataset_id = max(dataset_vocab.values(), default=-1)
    max_encoder_id = max(encoder_vocab.values(), default=-1)
    max_teacher_id = max(teacher_vocab.values(), default=-1) if teacher_check is not None else None
    dataset_encoder_vocab_equal = dataset_vocab == encoder_vocab
    dataset_encoder_added_equal = dataset_added == encoder_added
    dataset_encoder_equal = dataset_encoder_vocab_equal and dataset_encoder_added_equal
    if teacher_check is None:
        dataset_teacher_vocab_equal = None
        dataset_teacher_added_equal = None
        encoder_teacher_vocab_equal = None
        encoder_teacher_added_equal = None
        dataset_teacher_equal = None
        encoder_teacher_equal = None
    else:
        dataset_teacher_vocab_equal = dataset_vocab == teacher_vocab
        dataset_teacher_added_equal = dataset_added == teacher_added
        encoder_teacher_vocab_equal = encoder_vocab == teacher_vocab
        encoder_teacher_added_equal = encoder_added == teacher_added
        dataset_teacher_equal = dataset_teacher_vocab_equal and dataset_teacher_added_equal
        encoder_teacher_equal = encoder_teacher_vocab_equal and encoder_teacher_added_equal
    family_equal = teacher_check is None or encoder_family == teacher_family
    supported_family = (
        encoder_family in {"t5", "qwen3"}
        and (teacher_check is None or (family_equal and teacher_family in {"t5", "qwen3"}))
    )
    ids_fit_encoder = encoder_check["consumer_ids_fit_embeddings"]
    ids_fit_teacher = (
        teacher_check["consumer_ids_fit_embeddings"] if teacher_check is not None else None
    )
    compatible = (
        encoder_check["compatible"] and supported_family
        and (teacher_check is None or teacher_check["compatible"])
    )

    dataset_post_processor = _post_processor_state(dataset_tokenizer)
    encoder_post_processor = _post_processor_state(encoder_tokenizer)
    teacher_post_processor = _post_processor_state(teacher_tokenizer) if teacher_tokenizer is not None else ""
    dataset_chat_template = getattr(dataset_tokenizer, "chat_template", None)
    encoder_chat_template = getattr(encoder_tokenizer, "chat_template", None)
    teacher_chat_template = getattr(teacher_tokenizer, "chat_template", None) if teacher_tokenizer is not None else None
    dataset_special_tokens = _special_tokens(dataset_tokenizer)
    encoder_special_tokens = _special_tokens(encoder_tokenizer)
    teacher_special_tokens = _special_tokens(teacher_tokenizer) if teacher_tokenizer is not None else {}

    report = {
        "compatible": compatible,
        "dataset_tokenizer": dataset_name,
        "encoder_tokenizer": encoder_name,
        "teacher_tokenizer": teacher_name,
        "encoder_family": encoder_family,
        "teacher_family": teacher_family,
        "family_equal": family_equal,
        "supported_family": supported_family,
        "dataset_vocab_size": len(dataset_vocab),
        "encoder_vocab_size": len(encoder_vocab),
        "teacher_vocab_size": len(teacher_vocab) if teacher_check is not None else None,
        "dataset_tokenizer_length": len(dataset_tokenizer),
        "encoder_tokenizer_length": len(encoder_tokenizer),
        "teacher_tokenizer_length": len(teacher_tokenizer) if teacher_tokenizer is not None else None,
        "dataset_vocab_sha256": _vocab_hash(dataset_vocab),
        "encoder_vocab_sha256": _vocab_hash(encoder_vocab),
        "teacher_vocab_sha256": _vocab_hash(teacher_vocab) if teacher_check is not None else None,
        "dataset_added_vocab_sha256": _vocab_hash(dataset_added),
        "encoder_added_vocab_sha256": _vocab_hash(encoder_added),
        "teacher_added_vocab_sha256": _vocab_hash(teacher_added) if teacher_check is not None else None,
        "dataset_encoder_vocab_equal": dataset_encoder_vocab_equal,
        "dataset_teacher_vocab_equal": dataset_teacher_vocab_equal,
        "dataset_encoder_added_vocab_equal": dataset_encoder_added_equal,
        "dataset_teacher_added_vocab_equal": dataset_teacher_added_equal,
        "encoder_teacher_vocab_equal": encoder_teacher_vocab_equal,
        "encoder_teacher_added_vocab_equal": encoder_teacher_added_equal,
        "dataset_encoder_mapping_equal": dataset_encoder_equal,
        "dataset_teacher_mapping_equal": dataset_teacher_equal,
        "encoder_teacher_mapping_equal": encoder_teacher_equal,
        "dataset_encoder_mismatches": _mapping_mismatches(dataset_vocab, encoder_vocab),
        "dataset_teacher_mismatches": (
            _mapping_mismatches(dataset_vocab, teacher_vocab) if teacher_check is not None else []
        ),
        "dataset_encoder_added_mismatches": _mapping_mismatches(dataset_added, encoder_added),
        "dataset_teacher_added_mismatches": (
            _mapping_mismatches(dataset_added, teacher_added) if teacher_check is not None else []
        ),
        "encoder_teacher_mismatches": (
            _mapping_mismatches(encoder_vocab, teacher_vocab) if teacher_check is not None else []
        ),
        "encoder_teacher_added_mismatches": (
            _mapping_mismatches(encoder_added, teacher_added) if teacher_check is not None else []
        ),
        "encoder_embedding_rows": encoder_rows,
        "teacher_embedding_rows": teacher_rows,
        "encoder_config_vocab_size": int(getattr(encoder.model.config, "vocab_size", 0) or 0),
        "teacher_config_vocab_size": (
            int(getattr(teacher.model.config, "vocab_size", 0) or 0)
            if teacher is not None else None
        ),
        "max_dataset_token_id": max_dataset_id,
        "max_encoder_token_id": max_encoder_id,
        "max_teacher_token_id": max_teacher_id,
        "ids_fit_encoder_embeddings": ids_fit_encoder,
        "ids_fit_teacher_embeddings": ids_fit_teacher,
        "dataset_encoder_base_vocab_equal": encoder_check["base_vocab_equal"],
        "dataset_teacher_base_vocab_equal": (
            teacher_check["base_vocab_equal"] if teacher_check is not None else None
        ),
        "dataset_encoder_bpe_model_equal": encoder_check["bpe_model_equal"],
        "dataset_teacher_bpe_model_equal": (
            teacher_check["bpe_model_equal"] if teacher_check is not None else None
        ),
        "dataset_bpe_model_sha256": encoder_check["dataset_bpe_model_sha256"],
        "encoder_bpe_model_sha256": encoder_check["consumer_bpe_model_sha256"],
        "teacher_bpe_model_sha256": (
            teacher_check["consumer_bpe_model_sha256"] if teacher_check is not None else None
        ),
        "dataset_encoder_qwen_special_ids_valid": encoder_check["qwen_special_ids_valid"],
        "dataset_teacher_qwen_special_ids_valid": (
            teacher_check["qwen_special_ids_valid"] if teacher_check is not None else None
        ),
        "dataset_encoder_compatible_ids_fit_embeddings": (
            encoder_check["compatible_dataset_ids_fit_embeddings"]
        ),
        "dataset_teacher_compatible_ids_fit_embeddings": (
            teacher_check["compatible_dataset_ids_fit_embeddings"]
            if teacher_check is not None else None
        ),
        "dataset_encoder_incompatible_added_token_ids": (
            encoder_check["incompatible_dataset_added_token_ids"]
        ),
        "dataset_teacher_incompatible_added_token_ids": (
            teacher_check["incompatible_dataset_added_token_ids"]
            if teacher_check is not None else []
        ),
        "dataset_special_tokens": dataset_special_tokens,
        "encoder_special_tokens": encoder_special_tokens,
        "teacher_special_tokens": teacher_special_tokens,
        "dataset_encoder_special_tokens_equal": dataset_special_tokens == encoder_special_tokens,
        "dataset_teacher_special_tokens_equal": (
            dataset_special_tokens == teacher_special_tokens if teacher_check is not None else None
        ),
        "dataset_post_processor_sha256": _text_hash(dataset_post_processor),
        "encoder_post_processor_sha256": _text_hash(encoder_post_processor),
        "teacher_post_processor_sha256": (
            _text_hash(teacher_post_processor) if teacher_check is not None else None
        ),
        "dataset_encoder_post_processor_equal": dataset_post_processor == encoder_post_processor,
        "dataset_teacher_post_processor_equal": (
            dataset_post_processor == teacher_post_processor if teacher_check is not None else None
        ),
        "dataset_chat_template_sha256": _text_hash(dataset_chat_template),
        "encoder_chat_template_sha256": _text_hash(encoder_chat_template),
        "teacher_chat_template_sha256": (
            _text_hash(teacher_chat_template) if teacher_check is not None else None
        ),
        "dataset_encoder_chat_template_equal": dataset_chat_template == encoder_chat_template,
        "dataset_teacher_chat_template_equal": (
            dataset_chat_template == teacher_chat_template if teacher_check is not None else None
        ),
    }
    if report_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    if not compatible:
        suffix = f"; see {report_path} for details" if report_path is not None else ""
        raise ValueError(f"Tokenizer compatibility check failed{suffix}")
    return report


def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence:
        raise ValueError("Chat-template sentinel tokenization is empty")
    matches = [
        idx for idx in range(len(sequence) - len(subsequence) + 1)
        if sequence[idx:idx + len(subsequence)] == subsequence
    ]
    if len(matches) != 1:
        raise ValueError("Could not uniquely locate chat-template sentinel")
    return matches[0]


@dataclass(frozen=True)
class TeacherScaffold:
    user_prefix: Tuple[int, ...] = ()
    user_suffix: Tuple[int, ...] = ()
    assistant_prefix: Tuple[int, ...] = ()
    assistant_suffix: Tuple[int, ...] = ()

    @classmethod
    def qwen3_chat(cls, tokenizer) -> "TeacherScaffold":
        user_text = "ELF_USER_SENTINEL_8d31"
        assistant_text = "ELF_ASSISTANT_SENTINEL_4a72"
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        user_ids = tokenizer(user_text, add_special_tokens=False)["input_ids"]
        assistant_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
        user_start = _find_subsequence(full_ids, user_ids)
        assistant_start = _find_subsequence(full_ids, assistant_ids)
        if assistant_start <= user_start + len(user_ids):
            raise ValueError("Unexpected Qwen3 chat-template message order")

        prefix = full_ids[:user_start]
        bridge = full_ids[user_start + len(user_ids):assistant_start]
        suffix = full_ids[assistant_start + len(assistant_ids):]
        assistant_marker = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        marker_start = _find_subsequence(bridge, assistant_marker)
        scaffold = cls(
            user_prefix=tuple(prefix),
            user_suffix=tuple(bridge[:marker_start]),
            assistant_prefix=tuple(bridge[marker_start:]),
            assistant_suffix=tuple(suffix),
        )
        reconstructed = (
            list(scaffold.user_prefix) + user_ids + list(scaffold.user_suffix)
            + list(scaffold.assistant_prefix) + assistant_ids + list(scaffold.assistant_suffix)
        )
        if reconstructed != full_ids:
            raise ValueError("Unsupported Qwen3 chat template: scaffold reconstruction failed")
        return scaffold


@dataclass
class TeacherBatch:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    text_positions: torch.Tensor
    text_valid: torch.Tensor
    reg_positions: Optional[torch.Tensor]


@dataclass
class TeacherTargets:
    text: dict[str, torch.Tensor]
    reg: dict[str, torch.Tensor]


class TeacherTargetProvider:
    """Build teacher sequences and gather aligned token/REG representations."""

    def __init__(
        self, encoder, tokenizer, input_format: str, *, require_reg: bool,
        reg_teacher_pooling: str = "eot",
        incompatible_token_ids=(), consumer_name: str = "teacher",
        source_special_token_ids=None,
    ):
        if input_format not in TEACHER_INPUT_FORMATS:
            raise ValueError(
                f"repa_teacher_input_format must be one of {sorted(TEACHER_INPUT_FORMATS)}, got {input_format!r}"
            )
        if input_format == "qwen3_chat" and getattr(encoder.config, "family", None) != "qwen3":
            raise ValueError("qwen3_chat teacher formatting requires a Qwen3 teacher")
        if tokenizer.pad_token_id is None:
            raise ValueError("Teacher tokenizer must define pad_token_id")
        if reg_teacher_pooling not in {"eot", "last", "mean"}:
            raise ValueError(f"reg_teacher_pooling must be one of ['eot', 'last', 'mean'], got {reg_teacher_pooling!r}")

        self.encoder = encoder
        self.tokenizer = tokenizer
        self.input_format = input_format
        self.reg_teacher_pooling = reg_teacher_pooling
        self.incompatible_token_ids = tuple(int(token_id) for token_id in incompatible_token_ids)
        self.consumer_name = str(consumer_name)
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.scaffold = (
            TeacherScaffold.qwen3_chat(tokenizer)
            if input_format == "qwen3_chat" else TeacherScaffold()
        )
        self.family = getattr(encoder.config, "family", None)
        if self.family == "qwen3":
            special_ids = (
                tokenizer.all_special_ids
                if source_special_token_ids is None else source_special_token_ids
            )
            self.source_special_token_ids = tuple(sorted({
                int(token_id) for token_id in special_ids
            }))
        else:
            self.source_special_token_ids = ()
        self.eot_token_id = None
        if self.family == "qwen3" or require_reg:
            self.eot_token_id = int(tokenizer.convert_tokens_to_ids(EOT_TOKEN))
            if self.eot_token_id != EOT_TOKEN_ID:
                feature = "REG" if require_reg else "Qwen teacher inputs"
                raise ValueError(
                    f"{feature} requires {EOT_TOKEN} to resolve to token ID "
                    f"{EOT_TOKEN_ID}, got {self.eot_token_id}"
                )

        model_config = getattr(encoder.model, "config", None)
        self.max_position_embeddings = int(getattr(model_config, "max_position_embeddings", 0) or 0)
        self.embedding_rows = _embedding_rows(encoder)

    @staticmethod
    def _scatter_piece(
        output: torch.Tensor,
        is_valid: torch.Tensor,
        is_condition: torch.Tensor,
        starts: torch.Tensor,
        piece: Tuple[int, ...],
        *,
        condition: bool,
    ) -> None:
        if not piece:
            return
        batch = output.shape[0]
        offsets = torch.arange(len(piece), device=output.device).view(1, -1)
        positions = starts.view(-1, 1) + offsets
        values = torch.tensor(piece, dtype=output.dtype, device=output.device).view(1, -1).expand(batch, -1)
        output.scatter_(1, positions, values)
        is_valid.scatter_(1, positions, torch.ones_like(values, dtype=torch.bool))
        if condition:
            is_condition.scatter_(1, positions, torch.ones_like(values, dtype=torch.bool))

    def build_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cond_seq_mask: torch.Tensor,
        label_drop_mask: Optional[torch.Tensor],
        *,
        include_reg: bool,
    ) -> TeacherBatch:
        batch, text_length = input_ids.shape
        source_valid = attention_mask.to(torch.bool)
        condition_mask = cond_seq_mask.to(torch.bool)
        if torch.any(condition_mask & ~source_valid):
            raise ValueError("Condition tokens must also be valid text tokens")
        source_condition_valid = condition_mask & source_valid
        source_cond_lens = source_condition_valid.sum(dim=1).to(torch.long)
        source_total_lens = source_valid.sum(dim=1).to(torch.long)
        source_pos = torch.arange(text_length, device=input_ids.device).view(1, -1)
        if not torch.equal(source_valid, source_pos < source_total_lens.view(-1, 1)):
            raise ValueError("Teacher formatting requires valid text tokens to form a contiguous prefix")
        if not torch.equal(
            source_condition_valid, source_pos < source_cond_lens.view(-1, 1),
        ):
            raise ValueError("Teacher formatting requires condition tokens to form a contiguous prefix")

        text_valid = source_valid
        if self.input_format == "raw" and self.family == "qwen3":
            text_valid = build_auxiliary_content_mask(
                input_ids, attention_mask, self.source_special_token_ids,
            )
            if torch.any(text_valid.sum(dim=1) == 0):
                raise ValueError("Raw teacher input contains no content tokens")
        condition_valid = condition_mask & text_valid
        cond_lens = condition_valid.sum(dim=1).to(torch.long)
        total_lens = text_valid.sum(dim=1).to(torch.long)

        up = len(self.scaffold.user_prefix)
        us = len(self.scaffold.user_suffix)
        ap = len(self.scaffold.assistant_prefix)
        ass = len(self.scaffold.assistant_suffix)
        if self.input_format == "raw" and self.family == "qwen3":
            append_eot = include_reg and self.reg_teacher_pooling == "eot"
        else:
            append_eot = self.eot_token_id is not None
        if append_eot and self.eot_token_id is None:
            raise ValueError("TeacherTargetProvider was not configured for EOT REG pooling")
        teacher_lens = up + us + ap + ass + total_lens + int(append_eot)
        max_teacher_length = int(teacher_lens.max().item())
        if self.max_position_embeddings and max_teacher_length > self.max_position_embeddings:
            raise ValueError(
                f"Teacher input length {max_teacher_length} exceeds max_position_embeddings={self.max_position_embeddings}"
            )

        teacher_ids = torch.full(
            (batch, max_teacher_length), self.pad_token_id,
            dtype=input_ids.dtype, device=input_ids.device,
        )
        is_valid = torch.zeros((batch, max_teacher_length), dtype=torch.bool, device=input_ids.device)
        is_condition = torch.zeros_like(is_valid)
        zeros = torch.zeros((batch,), dtype=torch.long, device=input_ids.device)

        self._scatter_piece(
            teacher_ids, is_valid, is_condition, zeros,
            self.scaffold.user_prefix, condition=True,
        )

        source_pos = source_pos.expand(batch, -1)
        content_offsets = text_valid.to(torch.long).cumsum(dim=1) - 1
        prompt_positions = up + content_offsets
        response_positions = up + us + ap + content_offsets
        text_positions = torch.where(condition_valid, prompt_positions, response_positions)
        text_positions = torch.where(text_valid, text_positions, torch.full_like(text_positions, -1))
        source_rows, source_cols = torch.where(text_valid)
        target_cols = text_positions[source_rows, source_cols]
        teacher_ids[source_rows, target_cols] = input_ids[source_rows, source_cols]
        is_valid[source_rows, target_cols] = True
        is_condition[source_rows, target_cols] = condition_valid[source_rows, source_cols]

        user_suffix_start = torch.full_like(cond_lens, up) + cond_lens
        self._scatter_piece(
            teacher_ids, is_valid, is_condition, user_suffix_start,
            self.scaffold.user_suffix, condition=True,
        )
        assistant_prefix_start = user_suffix_start + us
        self._scatter_piece(
            teacher_ids, is_valid, is_condition, assistant_prefix_start,
            self.scaffold.assistant_prefix, condition=False,
        )
        assistant_suffix_start = up + us + ap + total_lens
        self._scatter_piece(
            teacher_ids, is_valid, is_condition, assistant_suffix_start,
            self.scaffold.assistant_suffix, condition=False,
        )

        reg_positions = None
        if append_eot:
            reg_positions = assistant_suffix_start + ass
            teacher_ids[torch.arange(batch, device=input_ids.device), reg_positions] = self.eot_token_id
            is_valid[torch.arange(batch, device=input_ids.device), reg_positions] = True

        # Without label dropping, causal Qwen gives every consumed text/EOT
        # state the same context; the only remaining tokens are later padding.
        teacher_attention_mask = None
        if self.family != "qwen3" or label_drop_mask is not None:
            allowed = (
                (is_condition[:, :, None] & is_condition[:, None, :])
                | (~is_condition[:, :, None] & is_valid[:, None, :])
            )
            allowed = allowed & is_valid[:, :, None]
            if label_drop_mask is not None:
                drop = label_drop_mask.to(torch.bool).view(-1, 1, 1)
                blocked = (~is_condition[:, :, None]) & is_condition[:, None, :]
                allowed = allowed & ~(drop & blocked)
            if reg_positions is not None:
                row = torch.arange(batch, device=input_ids.device)
                allowed[row, reg_positions] = is_valid
            teacher_attention_mask = allowed.to(torch.float32)

        return TeacherBatch(
            input_ids=teacher_ids,
            attention_mask=teacher_attention_mask,
            text_positions=text_positions,
            text_valid=text_valid,
            reg_positions=reg_positions,
        )

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cond_seq_mask: torch.Tensor,
        label_drop_mask: Optional[torch.Tensor],
        views: Mapping[str, Tuple[Optional[int], Optional[int]]],
        *,
        include_reg: bool,
        use_bf16: bool,
    ) -> TeacherTargets:
        validate_input_ids_compatibility(
            input_ids, self.incompatible_token_ids, self.consumer_name,
        )
        teacher_batch = self.build_batch(
            input_ids, attention_mask, cond_seq_mask, label_drop_mask,
            include_reg=include_reg,
        )
        if teacher_batch.input_ids.numel():
            min_id = int(teacher_batch.input_ids.min().item())
            max_id = int(teacher_batch.input_ids.max().item())
            if min_id < 0 or max_id >= self.embedding_rows:
                raise ValueError(
                    f"Teacher input IDs must be in [0, {self.embedding_rows}), got [{min_id}, {max_id}]"
                )

        autocast_enabled = bool(use_bf16) and input_ids.is_cuda
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            hidden_views = self.encoder.encode_views(
                teacher_batch.input_ids,
                teacher_batch.attention_mask,
                views,
                deterministic=True,
                normalize=False,
            )

        text_targets = {}
        reg_targets = {}
        for name, hidden in hidden_views.items():
            gather_pos = teacher_batch.text_positions.clamp_min(0)
            gather_pos = gather_pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            text = hidden.gather(1, gather_pos)
            text_targets[name] = text * teacher_batch.text_valid.unsqueeze(-1).to(text.dtype)
            if include_reg:
                if self.reg_teacher_pooling == "eot":
                    row = torch.arange(hidden.shape[0], device=hidden.device)
                    pooled = hidden[row, teacher_batch.reg_positions].unsqueeze(1)
                elif self.reg_teacher_pooling == "last":
                    positions = torch.arange(text.shape[1], device=text.device)
                    positions = positions.expand(text.shape[0], -1)
                    last = positions.masked_fill(~teacher_batch.text_valid, -1).max(dim=1).values
                    if torch.any(last < 0):
                        raise ValueError("Last REG pooling requires at least one valid text token")
                    row = torch.arange(text.shape[0], device=text.device)
                    pooled = text_targets[name][row, last].unsqueeze(1)
                else:
                    counts = teacher_batch.text_valid.sum(dim=1, keepdim=True)
                    if torch.any(counts == 0):
                        raise ValueError("Mean REG pooling requires at least one valid text token")
                    pooled = text_targets[name].sum(dim=1, keepdim=True)
                    pooled = pooled / counts.unsqueeze(-1).to(pooled.dtype)
                reg_targets[name] = feature_standardize(pooled)
        return TeacherTargets(text=text_targets, reg=reg_targets)
