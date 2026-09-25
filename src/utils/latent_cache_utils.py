"""Read row-aligned compressed latent caches without loading their source models."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
from torch.utils.data import Dataset

from utils.data_utils import get_pad_token_id


CACHE_SCHEMA_VERSION = 4
STAGED_CACHE_ROOT_ENV = "ELF_LATENT_CACHE_ROOT"


@dataclass(frozen=True)
class LatentCacheSplitConfig:
    text_cache_path: str
    teacher_cache_path: Optional[str] = None


@dataclass(frozen=True)
class LatentCacheConfig:
    cache_root: str
    destination_path: Optional[str]
    train: LatentCacheSplitConfig
    eval: dict[str, LatentCacheSplitConfig]

    @property
    def runtime_root(self) -> str:
        return os.path.abspath(os.path.expanduser(
            os.environ.get(STAGED_CACHE_ROOT_ENV, self.cache_root),
        ))

    def referenced_paths(self):
        yield self.train.text_cache_path, "encoder"
        if self.train.teacher_cache_path is not None:
            yield self.train.teacher_cache_path, "teacher"
        for split in self.eval.values():
            yield split.text_cache_path, "encoder"


@dataclass(frozen=True)
class LatentCacheInfo:
    text_dim: int
    teacher_dim: Optional[int]
    cache_config: LatentCacheConfig


@dataclass(frozen=True)
class _Shard:
    path: str
    row_start: int
    row_end: int
    code_vectors: int


def _strict_keys(value, allowed, context):
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unknown {context} field(s): {', '.join(unknown)}",
        )


def _relative_cache_path(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty relative path")
    value = os.path.normpath(value.strip())
    if os.path.isabs(value) or value == ".." or value.startswith(".." + os.sep):
        raise ValueError(f"{field_name} must stay beneath cache_root")
    return value


def _split_config(value, context, *, allow_teacher):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    allowed = {"text_cache_path", "teacher_cache_path"} if allow_teacher else {
        "text_cache_path",
    }
    _strict_keys(value, allowed, context)
    if "text_cache_path" not in value:
        raise ValueError(f"{context}.text_cache_path is required")
    teacher = value.get("teacher_cache_path")
    return LatentCacheSplitConfig(
        text_cache_path=_relative_cache_path(
            value["text_cache_path"], f"{context}.text_cache_path",
        ),
        teacher_cache_path=(
            _relative_cache_path(teacher, f"{context}.teacher_cache_path")
            if teacher is not None else None
        ),
    )


def load_latent_cache_config(path: str) -> LatentCacheConfig:
    if not path or not os.path.isfile(path):
        raise ValueError(f"Latent cache config does not exist: {path!r}")
    with open(path, encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Latent cache config must be a mapping")
    _strict_keys(value, {"cache_root", "destination_path", "train", "eval"}, "latent cache config")
    for field in ("cache_root", "train", "eval"):
        if field not in value:
            raise ValueError(f"Latent cache config requires {field}")

    cache_root = value["cache_root"]
    if not isinstance(cache_root, str) or not cache_root.strip():
        raise ValueError("cache_root must be a non-empty path")
    cache_root = os.path.abspath(os.path.expanduser(cache_root))

    destination = value.get("destination_path")
    if destination is not None:
        if not isinstance(destination, str) or not destination.strip():
            raise ValueError("destination_path must be a path or null")
        destination = os.path.abspath(os.path.expanduser(destination))

    eval_value = value["eval"]
    if not isinstance(eval_value, dict):
        raise ValueError("eval must be a mapping of dataset names")
    eval_splits = {
        str(name): _split_config(spec, f"eval.{name}", allow_teacher=False)
        for name, spec in eval_value.items()
    }
    return LatentCacheConfig(
        cache_root=cache_root,
        destination_path=destination,
        train=_split_config(value["train"], "train", allow_teacher=True),
        eval=eval_splits,
    )


def _json_load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _vocab_hash(vocab):
    payload = json.dumps(
        sorted((str(token), int(index)) for token, index in vocab.items()),
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _special_tokens(tokenizer):
    return {
        key: [str(token) for token in item] if isinstance(item, (list, tuple)) else str(item)
        for key, item in tokenizer.special_tokens_map.items()
    }


def _tokenizer_identity(tokenizer):
    return {
        "dataset_tokenizer_length": len(tokenizer),
        "dataset_vocab_sha256": _vocab_hash(tokenizer.get_vocab()),
        "dataset_added_vocab_sha256": _vocab_hash(tokenizer.get_added_vocab()),
        "dataset_special_tokens": _special_tokens(tokenizer),
    }


def _canonical_layer(value):
    return -1 if value is None else int(value)


class CompressedLatentCache:
    """Validated, lazily memory-mapped compressed latent shards."""

    def __init__(
        self, path, *, source, dataset, dataset_name, config, tokenizer,
        tokenizer_identity,
    ):
        self.path = os.path.abspath(path)
        self.source = source
        self.manifest = self._validate_manifest(
            dataset, dataset_name, config, tokenizer, tokenizer_identity,
        )
        self.dimension = int(self.manifest["dimension"])
        self.max_length = int(self.manifest["artifact_identity"]["max_length"])
        self.shards = self._validate_shards()
        self._row_ends = [shard.row_end for shard in self.shards]
        self._open_arrays = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_open_arrays"] = {}
        return state

    def _validate_manifest(
        self, dataset, dataset_name, config, tokenizer, tokenizer_identity,
    ):
        manifest_path = os.path.join(self.path, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise ValueError(f"Latent cache has no manifest: {self.path}")
        manifest = _json_load(manifest_path)
        if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported latent cache schema: {self.path}")
        if manifest.get("status") != "complete":
            raise ValueError(f"Latent cache is not complete: {self.path}")
        if manifest.get("dtype") != "float16":
            raise ValueError(f"Latent cache must contain float16 codes: {self.path}")
        if int(manifest.get("dimension", 0)) <= 0:
            raise ValueError(f"Latent cache has an invalid dimension: {self.path}")

        expected_layout = (
            "dense_rows" if self.source == "encoder"
            else "repa_text_then_repa_eot_then_reg_last_then_reg_mean_then_reg_eot"
        )
        if manifest.get("layout") != expected_layout:
            raise ValueError(f"Latent cache layout does not match {self.source}: {self.path}")

        rows = len(dataset)
        fingerprint = getattr(dataset, "_fingerprint", None)
        dataset_identity = manifest.get("dataset", {})
        if (
            int(manifest.get("rows", -1)) != rows
            or int(dataset_identity.get("rows", -1)) != rows
            or str(dataset_identity.get("fingerprint")) != str(fingerprint)
            or dataset_identity.get("name") != dataset_name
        ):
            raise ValueError(
                f"Latent cache dataset identity does not match {dataset_name}: {self.path}",
            )

        identity = manifest.get("artifact_identity", {})
        if identity.get("source") != self.source:
            raise ValueError(f"Latent cache source mismatch: {self.path}")
        tokenizer_name = config.tokenizer_name or config.encoder_model_name
        if identity.get("tokenizer_name") != tokenizer_name:
            raise ValueError(f"Latent cache tokenizer name mismatch: {self.path}")
        cached_tokenizer = identity.get("tokenizer_identity", {})
        for key, expected in tokenizer_identity.items():
            if cached_tokenizer.get(key) != expected:
                raise ValueError(f"Latent cache tokenizer identity mismatch: {self.path}")
        if (
            int(identity.get("max_length", -1)) != int(config.max_length)
            or identity.get("max_input_length") != config.max_input_length
            or identity.get("pad_token") != config.pad_token
            or int(identity.get("pad_token_id", -1))
            != int(get_pad_token_id(tokenizer, config.pad_token))
            or identity.get("hidden_state_preprocessing") != "raw"
        ):
            raise ValueError(f"Latent cache sequence configuration mismatch: {self.path}")

        specs = identity.get("target_specs", [])
        if self.source == "encoder":
            matches = [spec for spec in specs if "clean" in spec.get("roles", [])]
            if len(matches) != 1 or (
                matches[0].get("model_name") != config.encoder_model_name
                or int(matches[0].get("layer", -2)) != _canonical_layer(config.encoder_layer)
            ):
                raise ValueError(f"Latent cache clean-encoder identity mismatch: {self.path}")
        else:
            if identity.get("teacher_input_format") != config.repa_teacher_input_format:
                raise ValueError(f"Latent cache teacher format mismatch: {self.path}")
            for role, layer in (
                ("repa", config.repa_teacher_layer),
                ("reg", config.reg_teacher_layer),
            ):
                matches = [spec for spec in specs if role in spec.get("roles", [])]
                if len(matches) != 1 or (
                    matches[0].get("model_name") != config.repa_teacher_model_name
                    or int(matches[0].get("layer", -2)) != _canonical_layer(layer)
                ):
                    raise ValueError(f"Latent cache {role.upper()} identity mismatch: {self.path}")
            if config.reg_teacher_pooling not in identity.get("supported_reg_teacher_pooling", []):
                raise ValueError(f"Latent cache does not support configured REG pooling: {self.path}")
            if config.repa_reg_target_source not in identity.get(
                "supported_repa_reg_target_source", [],
            ):
                raise ValueError(f"Latent cache does not support configured REPA prefix: {self.path}")
        return manifest

    def _validate_shards(self):
        manifest_shards = self.manifest.get("shards", [])
        if len(manifest_shards) != int(self.manifest.get("num_shards", -1)):
            raise ValueError(f"Latent cache shard count mismatch: {self.path}")
        shards = []
        next_row = 0
        for index, entry in enumerate(manifest_shards):
            start, end = int(entry.get("row_start", -1)), int(entry.get("row_end", -1))
            if int(entry.get("index", -1)) != index or start != next_row or end < start:
                raise ValueError(f"Latent cache shard ranges are invalid: {self.path}")
            shard_path = os.path.join(self.path, entry["name"])
            metadata_path = os.path.join(shard_path, "shard.json")
            if not os.path.isdir(shard_path) or not os.path.isfile(metadata_path):
                raise ValueError(f"Latent cache shard is missing: {shard_path}")
            metadata = _json_load(metadata_path)
            code_vectors = int(metadata.get("code_vectors", -1))
            if (
                int(metadata.get("index", -1)) != index
                or int(metadata.get("row_start", -1)) != start
                or int(metadata.get("row_end", -1)) != end
                or int(metadata.get("dimension", -1)) != self.dimension
                or metadata.get("dtype") != "float16"
            ):
                raise ValueError(f"Latent cache shard metadata mismatch: {shard_path}")

            codes = np.load(os.path.join(shard_path, "codes.npy"), mmap_mode="r")
            rows = end - start
            if self.source == "encoder":
                expected_shape = (rows, self.max_length, self.dimension)
                expected_vectors = rows * self.max_length
            else:
                offsets = np.load(os.path.join(shard_path, "offsets.npy"), mmap_mode="r")
                content = np.load(
                    os.path.join(shard_path, "content_lengths.npy"), mmap_mode="r",
                )
                prompt = np.load(
                    os.path.join(shard_path, "prompt_lengths.npy"), mmap_mode="r",
                )
                if (
                    offsets.dtype != np.int64 or offsets.shape != (rows + 1,)
                    or content.dtype != np.int64 or content.shape != (rows,)
                    or prompt.dtype != np.int64 or prompt.shape != (rows,)
                    or int(offsets[0]) != 0
                    or np.any(content <= 0) or np.any(content >= self.max_length)
                    or np.any(prompt < 0) or np.any(prompt > content)
                    or not np.array_equal(np.diff(offsets), content + 4)
                ):
                    raise ValueError(f"Latent teacher sidecars are invalid: {shard_path}")
                expected_vectors = int(offsets[-1])
                expected_shape = (expected_vectors, self.dimension)
            if (
                codes.dtype != np.float16
                or tuple(codes.shape) != expected_shape
                or code_vectors != expected_vectors
            ):
                raise ValueError(f"Latent cache code tensor mismatch: {shard_path}")
            del codes
            shards.append(_Shard(shard_path, start, end, code_vectors))
            next_row = end
        if next_row != int(self.manifest["rows"]):
            raise ValueError(f"Latent cache shards do not cover all rows: {self.path}")
        return shards

    def _shard_index(self, row):
        row = int(row)
        if row < 0:
            row += int(self.manifest["rows"])
        if not 0 <= row < int(self.manifest["rows"]):
            raise IndexError(row)
        return bisect.bisect_right(self._row_ends, row), row

    def _arrays(self, shard_index):
        arrays = self._open_arrays.get(shard_index)
        if arrays is not None:
            return arrays
        shard = self.shards[shard_index]
        arrays = {"codes": np.load(os.path.join(shard.path, "codes.npy"), mmap_mode="r")}
        if self.source == "teacher":
            for name in ("offsets", "content_lengths", "prompt_lengths"):
                arrays[name] = np.load(os.path.join(shard.path, name + ".npy"), mmap_mode="r")
        self._open_arrays[shard_index] = arrays
        return arrays

    def text_row(self, row):
        if self.source != "encoder":
            raise ValueError("text_row requires an encoder cache")
        shard_index, row = self._shard_index(row)
        local = row - self.shards[shard_index].row_start
        return self._arrays(shard_index)["codes"][local]

    def teacher_row(self, row):
        if self.source != "teacher":
            raise ValueError("teacher_row requires a teacher cache")
        shard_index, row = self._shard_index(row)
        arrays = self._arrays(shard_index)
        local = row - self.shards[shard_index].row_start
        start, end = int(arrays["offsets"][local]), int(arrays["offsets"][local + 1])
        content_length = int(arrays["content_lengths"][local])
        values = arrays["codes"][start:end]
        return {
            "repa_text": values[:content_length],
            "repa_eot": values[content_length],
            "reg_last": values[content_length + 1],
            "reg_mean": values[content_length + 2],
            "reg_eot": values[content_length + 3],
            "content_length": content_length,
            "prompt_length": int(arrays["prompt_lengths"][local]),
        }


def _validate_cache_pair(text_cache, teacher_cache):
    text_identity = text_cache.manifest["artifact_identity"]
    teacher_identity = teacher_cache.manifest["artifact_identity"]
    if text_cache.manifest["dataset"] != teacher_cache.manifest["dataset"]:
        raise ValueError("Text and teacher caches refer to different datasets")
    for key in (
        "tokenizer_name", "tokenizer_identity", "max_length", "max_input_length",
        "pad_token", "pad_token_id",
    ):
        if text_identity.get(key) != teacher_identity.get(key):
            raise ValueError(f"Text and teacher cache identities differ for {key}")


class LatentCachedDataset(Dataset):
    """Attach compressed latent rows to the original token dataset."""

    def __init__(
        self, dataset, text_cache, teacher_cache, *, config, special_token_ids,
        eos_token_id,
    ):
        self.dataset = dataset
        self.text_cache = text_cache
        self.teacher_cache = teacher_cache
        self.max_length = int(config.max_length)
        self.max_input_length = config.max_input_length
        self.repa_enabled = bool(config.repa_enabled)
        self.reg_enabled = bool(config.reg_enabled)
        self.reg_teacher_pooling = config.reg_teacher_pooling
        self.repa_reg_target_source = config.repa_reg_target_source
        self.special_token_ids = np.asarray(
            tuple(int(item) for item in special_token_ids), dtype=np.int64,
        )
        self.eos_token_id = int(eos_token_id)

    def __len__(self):
        return len(self.dataset)

    def _validate_teacher_lengths(self, item, cached):
        condition = np.asarray(item.get("condition_input_ids", ()), dtype=np.int64)
        if self.max_input_length is not None:
            condition = condition[: int(self.max_input_length)]
        response = np.asarray(item["input_ids"], dtype=np.int64)
        sequence = np.concatenate((condition, response))[: self.max_length]
        if (
            not len(sequence)
            or int(sequence[-1]) != self.eos_token_id
            or np.isin(sequence[:-1], self.special_token_ids).any()
            or np.isin(condition, self.special_token_ids).any()
        ):
            raise ValueError("Token row no longer satisfies the teacher cache layout")
        content_length = len(sequence) - 1
        prompt_length = len(condition)
        if (
            content_length != cached["content_length"]
            or prompt_length != cached["prompt_length"]
        ):
            raise ValueError("Teacher cache row is not aligned with its token row")

    @staticmethod
    def _pooled(cached, source, pooling):
        if source == "reg":
            return cached[f"reg_{pooling}"]
        if pooling == "eot":
            return cached["repa_eot"]
        if pooling == "last":
            return cached["repa_text"][-1]
        return cached["repa_text"].astype(np.float32).mean(axis=0).astype(np.float16)

    def __getitem__(self, row):
        item = dict(self.dataset[int(row)])
        item["cached_text_latents"] = self.text_cache.text_row(row)
        if self.teacher_cache is None:
            return item

        cached = self.teacher_cache.teacher_row(row)
        self._validate_teacher_lengths(item, cached)
        if self.repa_enabled:
            item["cached_repa_latents"] = cached["repa_text"]
        if self.reg_enabled:
            item["cached_reg_latent"] = self._pooled(
                cached, "reg", self.reg_teacher_pooling,
            )
            if self.repa_enabled:
                item["cached_repa_reg_latent"] = self._pooled(
                    cached, self.repa_reg_target_source, self.reg_teacher_pooling,
                )
        return item


def _join(root, relative):
    return os.path.join(root, relative)


def attach_latent_caches(
    config, tokenizer, *, train_dataset=None, eval_dataset=None,
):
    """Wrap loaded Arrow datasets and return their cache-derived dimensions."""
    cache_config = load_latent_cache_config(config.latent_cache_config_path)
    if float(config.label_drop_prob) != 0.0:
        raise ValueError("Cached latents require label_drop_prob=0")
    tokenizer_identity = _tokenizer_identity(tokenizer)
    special_token_ids = tuple(int(item) for item in tokenizer.all_special_ids)
    root = cache_config.runtime_root
    text_dim = None
    teacher_dim = None

    def open_cache(dataset, name, split, source):
        relative = (
            split.text_cache_path if source == "encoder"
            else split.teacher_cache_path
        )
        return CompressedLatentCache(
            _join(root, relative), source=source, dataset=dataset,
            dataset_name=name, config=config, tokenizer=tokenizer,
            tokenizer_identity=tokenizer_identity,
        )

    if train_dataset is not None:
        text_cache = open_cache(train_dataset, "train", cache_config.train, "encoder")
        text_dim = text_cache.dimension
        needs_teacher = bool(config.repa_enabled or config.reg_enabled)
        if needs_teacher != (cache_config.train.teacher_cache_path is not None):
            required = "required" if needs_teacher else "not used"
            raise ValueError(f"train.teacher_cache_path is {required} for this model")
        teacher_cache = None
        if needs_teacher:
            teacher_cache = open_cache(
                train_dataset, "train", cache_config.train, "teacher",
            )
            _validate_cache_pair(text_cache, teacher_cache)
            teacher_dim = teacher_cache.dimension
        train_dataset = LatentCachedDataset(
            train_dataset, text_cache, teacher_cache,
            config=config, special_token_ids=special_token_ids,
            eos_token_id=tokenizer.eos_token_id,
        )

    expected_eval_names = []
    if isinstance(config.eval_data_path, list):
        expected_eval_names = [str(spec["name"]) for spec in config.eval_data_path]
        datasets = list(eval_dataset or [])
    elif config.eval_data_path is not None:
        expected_eval_names = ["eval"]
        datasets = [eval_dataset]
    else:
        datasets = []
    if set(cache_config.eval) != set(expected_eval_names):
        raise ValueError("Latent cache eval datasets do not match eval_data_path")
    if len(datasets) != len(expected_eval_names):
        raise ValueError("Latent cache eval dataset count mismatch")
    wrapped_eval = []
    for name, dataset in zip(expected_eval_names, datasets):
        split = cache_config.eval[name]
        text_cache = open_cache(dataset, name, split, "encoder")
        if text_dim is None:
            text_dim = text_cache.dimension
        elif text_cache.dimension != text_dim:
            raise ValueError("All text latent caches must use one dimension")
        wrapped_eval.append(LatentCachedDataset(
            dataset, text_cache, None,
            config=config, special_token_ids=special_token_ids,
            eos_token_id=tokenizer.eos_token_id,
        ))
    if not isinstance(config.eval_data_path, list):
        wrapped_eval = wrapped_eval[0] if wrapped_eval else None
    if text_dim is None:
        raise ValueError("Latent cache config did not resolve a text dimension")

    config.encoder_dim = int(text_dim)
    if teacher_dim is not None:
        if config.repa_enabled:
            config.repa_teacher_dim = int(teacher_dim)
        if config.reg_enabled:
            config.reg_teacher_dim = int(teacher_dim)
    return train_dataset, wrapped_eval, LatentCacheInfo(
        text_dim=int(text_dim), teacher_dim=teacher_dim,
        cache_config=cache_config,
    )


def _validate_stage_source(path, source):
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise ValueError(f"Latent cache has no manifest: {path}")
    manifest = _json_load(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"Refusing to stage incomplete latent cache: {path}")
    if manifest.get("artifact_identity", {}).get("source") != source:
        raise ValueError(f"Latent cache source mismatch: {path}")


def stage_plan(training_config_path, overrides):
    from configs.config import apply_config_overrides, load_config_from_yaml

    config = load_config_from_yaml(training_config_path)
    if overrides:
        config = apply_config_overrides(config, overrides)
    if not config.latent_cache_config_path:
        return None
    cache_config = load_latent_cache_config(config.latent_cache_config_path)
    if cache_config.destination_path is None:
        return None
    references = []
    seen = set()
    validate_source = not os.path.exists(cache_config.destination_path)
    for relative, source in cache_config.referenced_paths():
        if relative in seen:
            continue
        seen.add(relative)
        if validate_source:
            _validate_stage_source(_join(cache_config.cache_root, relative), source)
        references.append(relative)
    return cache_config.destination_path, cache_config.cache_root, references


def _main():
    parser = argparse.ArgumentParser(description="Resolve latent-cache staging paths")
    parser.add_argument("--stage-plan", metavar="TRAINING_CONFIG")
    parser.add_argument("--config_override", action="append", default=[])
    args, _ = parser.parse_known_args()
    if not args.stage_plan:
        parser.error("--stage-plan is required")
    plan = stage_plan(args.stage_plan, args.config_override)
    if plan is not None:
        destination, source_root, references = plan
        print(destination)
        print(source_root)
        for relative in references:
            print(relative)


if __name__ == "__main__":
    _main()
