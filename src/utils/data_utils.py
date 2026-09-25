import json
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from utils.encoder_utils import build_self_attn_cond_masks
from utils.logging_utils import log_for_0


def _process_count() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def _process_index() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def get_pad_token_id(tokenizer, pad_token: str = "pad") -> int:
    """Resolve the token id used for padding, optionally using EOS as pad."""
    token_id = tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    if token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
    return token_id


def prepare_batch(batch: Dict, config, generator: torch.Generator) -> Dict:
    """Convert numpy batch to torch tensors and sample label-drop decisions."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v)
        elif isinstance(v, torch.Tensor):
            result[k] = v
        else:
            result[k] = v

    batch_size = result["input_ids"].shape[0]
    label_drop_mask = torch.zeros((batch_size,), dtype=torch.bool)
    if config.label_drop_prob > 0:
        u = torch.rand((batch_size,), generator=generator)
        label_drop_mask = u < config.label_drop_prob
    result["label_drop_mask"] = label_drop_mask
    return result


def pad_and_truncate(ids_list, target_len, pad_token_id):
    """Pad or truncate sequences to target_len, return stacked array and lengths."""
    padded, lengths = [], []
    for ids in ids_list:
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            ids = np.concatenate([ids, np.full(target_len - orig_len, pad_token_id, dtype=ids.dtype)])
        padded.append(ids)
        lengths.append(orig_len)
    return np.stack(padded), np.array(lengths)


class _SkipFirstSamplesSampler:
    def __init__(self, sampler, skip_first_samples: int):
        self.sampler = sampler
        self.skip_first_samples = skip_first_samples

    def __iter__(self):
        skip = self.skip_first_samples
        self.skip_first_samples = 0
        for i, idx in enumerate(self.sampler):
            if i < skip:
                continue
            yield idx

    def __len__(self):
        return max(0, len(self.sampler) - self.skip_first_samples)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)


def _effective_sequence_lengths(dataset, max_seq_length, max_input_seq_length):
    base_dataset = getattr(dataset, "dataset", dataset)
    table = getattr(base_dataset, "data", None)
    column_names = set(getattr(table, "column_names", ()))
    if "input_ids" in column_names:
        import pyarrow.compute as pc

        input_lengths = pc.list_value_length(table.column("input_ids")).to_numpy(zero_copy_only=False,)
        if "condition_input_ids" in column_names:
            condition_lengths = pc.list_value_length(
                table.column("condition_input_ids"),
            ).to_numpy(zero_copy_only=False)
            if max_input_seq_length is not None:
                condition_lengths = np.minimum(condition_lengths, max_input_seq_length,)
            input_lengths = input_lengths + condition_lengths
        return np.minimum(input_lengths, max_seq_length).astype(np.int32)

    lengths = np.empty(len(dataset), dtype=np.int32)
    for index in range(len(dataset)):
        item = dataset[index]
        length = len(item["input_ids"])
        if "condition_input_ids" in item:
            condition_length = len(item["condition_input_ids"])
            if max_input_seq_length is not None:
                condition_length = min(condition_length, max_input_seq_length)
            length += condition_length
        lengths[index] = min(length, max_seq_length)
    return lengths


class _LengthGroupedSampler(Sampler):
    """Shuffle similar-length global microbatches without ordering the epoch by length."""

    def __init__(
        self, lengths, batch_size, num_replicas, rank, seed,
    ):
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.global_batch_size = self.batch_size * self.num_replicas
        self.total_size = (len(self.lengths) // self.global_batch_size) * self.global_batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.lengths), generator=generator).tolist()
        indices = indices[:self.total_size]

        window_size = 50 * self.global_batch_size
        batches = []
        for start in range(0, self.total_size, window_size):
            window = sorted(indices[start:start + window_size], key=lambda index: self.lengths[index], reverse=True,)
            batches.extend(window[offset:offset + self.global_batch_size] for offset in range(0, len(window), self.global_batch_size))

        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        rank_indices = []
        for batch_index in batch_order:
            rank_indices.extend(batches[batch_index][self.rank::self.num_replicas])
        return iter(rank_indices)

    def __len__(self):
        return self.total_size // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def get_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    max_seq_length: int = 512,
    pad_token_id: int = 0,
    max_input_seq_length: Optional[int] = None,
    distributed: bool = True,
    seed: Optional[int] = None,
    skip_first_batches: int = 0,
    group_by_length: bool = False,
):
    """Create a DataLoader."""
    if skip_first_batches < 0:
        raise ValueError("skip_first_batches must be non-negative")

    def collate_fn(batch_list):
        input_ids_list = [np.array(item["input_ids"]) for item in batch_list]

        if "condition_input_ids" in batch_list[0]:
            seq_list, cond_lens = [], []
            for item in batch_list:
                cond = np.array(item["condition_input_ids"])[:max_input_seq_length]
                inp = np.array(item["input_ids"])
                seq_list.append(np.concatenate([cond, inp]))
                cond_lens.append(len(cond))
            cond_lens = np.array(cond_lens)
        else:
            seq_list = input_ids_list
            cond_lens = np.zeros(len(input_ids_list), dtype=np.int32)

        ids, total_lens = pad_and_truncate(seq_list, max_seq_length, pad_token_id)
        pos = np.arange(max_seq_length)[None, :]
        is_cond = pos < cond_lens[:, None]
        is_valid = pos < total_lens[:, None]
        encoder_attn, attn, pred = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
        result = {
            "input_ids": ids,
            "encoder_attention_mask": encoder_attn,
            "attention_mask": attn,
            "cond_seq_mask": pred,
        }
        for key in ("index", "input", "target", "task_id"):
            if key in batch_list[0]:
                result[key] = [item[key] for item in batch_list]

        if "cached_text_latents" in batch_list[0]:
            text_latents = [np.asarray(item["cached_text_latents"]) for item in batch_list]
            if any(
                value.ndim != 2 or value.shape[0] != max_seq_length
                for value in text_latents
            ):
                raise ValueError("Cached text latents do not match max_seq_length")
            result["cached_text_latents"] = np.stack(text_latents)

        if "cached_repa_latents" in batch_list[0]:
            repa_latents = [np.asarray(item["cached_repa_latents"]) for item in batch_list]
            target_dim = repa_latents[0].shape[-1]
            padded_repa = np.zeros(
                (len(batch_list), max_seq_length, target_dim), dtype=np.float16,
            )
            for row, value in enumerate(repa_latents):
                if value.ndim != 2 or value.shape[0] > max_seq_length or value.shape[1] != target_dim:
                    raise ValueError("Cached REPA latents have inconsistent shapes")
                padded_repa[row, :value.shape[0]] = value
            result["cached_repa_latents"] = padded_repa

        for key in ("cached_reg_latent", "cached_repa_reg_latent"):
            if key in batch_list[0]:
                values = np.stack([np.asarray(item[key]) for item in batch_list])
                if values.ndim != 2:
                    raise ValueError(f"{key} must contain one vector per row")
                result[key] = values[:, None, :]
        return result

    common = dict(
        batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
        drop_last=drop_last, persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    skip_first_samples = skip_first_batches * batch_size
    if group_by_length:
        if not shuffle or not drop_last:
            raise ValueError("group_by_length requires shuffle=True and drop_last=True")
        num_replicas = _process_count() if distributed else 1
        rank = _process_index() if distributed else 0
        sampler = _LengthGroupedSampler(
            _effective_sequence_lengths(dataset, max_seq_length, max_input_seq_length,),
            batch_size=batch_size, num_replicas=num_replicas, rank=rank, seed=0 if seed is None else int(seed),
        )
        if skip_first_samples:
            sampler = _SkipFirstSamplesSampler(sampler, skip_first_samples)
        return DataLoader(dataset, sampler=sampler, **common)
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=_process_count(), rank=_process_index(),
            shuffle=shuffle, seed=0 if seed is None else int(seed), drop_last=drop_last,
        )
        if skip_first_samples:
            sampler = _SkipFirstSamplesSampler(sampler, skip_first_samples)
        return DataLoader(dataset, sampler=sampler, **common)
    if shuffle and seed is not None:
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=int(seed), drop_last=drop_last,)
        if skip_first_samples:
            sampler = _SkipFirstSamplesSampler(sampler, skip_first_samples)
        return DataLoader(dataset, sampler=sampler, **common)
    if skip_first_samples:
        raise ValueError("skip_first_batches requires a sampler-backed dataloader")
    return DataLoader(dataset, shuffle=shuffle, **common)


def load_jsonl_dataset(path, tokenizer, input_key="input", output_key="output"):
    """Load a JSONL eval set (one `{input, output}` example per line)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append({
                "index": i,
                "input": data[input_key],
                "target": data[output_key],
                "condition_input_ids": tokenizer(data[input_key], add_special_tokens=False)["input_ids"],
                "input_ids": tokenizer(data[output_key], add_special_tokens=False)["input_ids"],
            })
    return examples


# ============================================
# Dataset loading
# ============================================



def load_dataset_split(path: str, dataset_cache_dir=None):
    """Load a prepared local Arrow split produced by the preparation scripts."""
    from datasets import DatasetDict, load_from_disk
    ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        if len(ds) != 1:
            raise ValueError(f"Expected one dataset split at {path!r}, got {list(ds)}")
        ds = ds[next(iter(ds))]
    ds.set_format(type="numpy", columns=ds.column_names)
    return ds


def load_dataset(config, dataset_cache_dir=None):
    """Resolve config.data_path / config.eval_data_path into train/eval datasets."""
    log_for_0(f"Loading dataset from {config.data_path}...")
    train_dataset = load_dataset_split(config.data_path, dataset_cache_dir)
    log_for_0(f"Train size: {len(train_dataset)}")

    eval_dataset = None
    if isinstance(config.eval_data_path, list):
        eval_dataset = []
        for spec in config.eval_data_path:
            dataset = load_dataset_split(spec["path"], dataset_cache_dir)
            eval_dataset.append(dataset)
            log_for_0(f"Eval dataset {spec['name']} size: {len(dataset)}")
    elif config.eval_data_path:
        eval_dataset = load_dataset_split(config.eval_data_path, dataset_cache_dir)
        log_for_0(f"Eval size: {len(eval_dataset)}")
    else:
        log_for_0("No eval dataset")
    return train_dataset, eval_dataset
