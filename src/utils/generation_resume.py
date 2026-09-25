import hashlib
import json
import os
from typing import Dict, Iterable, List, Tuple


def rank_range(num_samples: int, world: int, rank: int) -> Tuple[int, int]:
    per_rank = (num_samples + world - 1) // world
    start = rank * per_rank
    end = min(start + per_rank, num_samples)
    return start, max(start, end)


def stable_eval_seed(*parts) -> int:
    data = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)


def shard_path(final_path: str, rank: int) -> str:
    root, ext = os.path.splitext(final_path)
    return f"{root}.rank{rank}.tmp{ext}"


def metadata_path(path: str) -> str:
    return f"{path}.meta.json"


def final_tmp_path(final_path: str) -> str:
    root, ext = os.path.splitext(final_path)
    return f"{root}.final.tmp{ext}"


def write_metadata(path: str, metadata: Dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def validate_metadata(path: str, expected: Dict, ignored_keys=()):
    with open(path, "r", encoding="utf-8") as f:
        found = json.load(f)
    if (
        found.get("version") == 1
        and expected.get("version") == 1
        and expected.get("reg_enabled") is False
    ):
        found.setdefault("reg_enabled", False)
        found.setdefault("reg_teacher_dim", None)
    if ignored_keys:
        found = {k: v for k, v in found.items() if k not in ignored_keys}
        expected = {k: v for k, v in expected.items() if k not in ignored_keys}
    if found != expected:
        raise ValueError(f"metadata mismatch for {path}")


def _read_jsonl(path: str, allow_truncated_last: bool = False) -> Tuple[List[Dict], bool]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rows = []
    truncated = False
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            if allow_truncated_last and idx == len(lines) - 1:
                truncated = True
                break
            raise ValueError(f"blank JSONL line in {path}:{idx + 1}")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_truncated_last and idx == len(lines) - 1:
                truncated = True
                break
            raise
    return rows, truncated


def write_jsonl(path: str, rows: Iterable[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_rows(handle, rows: Iterable[Dict]):
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def _validate_sequential(rows: List[Dict], start_id: int, end_id: int):
    expected_id = start_id
    for row in rows:
        if row.get("id") != expected_id:
            raise ValueError(f"expected id {expected_id}, found {row.get('id')}")
        if "generated" not in row:
            raise ValueError(f"missing generated text for id {expected_id}")
        expected_id += 1
    if expected_id > end_id:
        raise ValueError(f"too many rows: expected ids below {end_id}, found {expected_id}")


def prepare_rank_shard(
    path: str,
    metadata: Dict,
    start_id: int,
    end_id: int,
    batch_size: int,
) -> int:
    del batch_size
    meta_path = metadata_path(path)
    if os.path.exists(meta_path):
        validate_metadata(meta_path, metadata, ignored_keys=("batch_size",))
    elif os.path.exists(path):
        raise ValueError(f"missing metadata for existing shard {path}")
    else:
        write_metadata(meta_path, metadata)

    if not os.path.exists(path):
        return start_id

    rows, truncated = _read_jsonl(path, allow_truncated_last=True)
    _validate_sequential(rows, start_id, end_id)

    if truncated:
        write_jsonl(path, rows)
    return start_id + len(rows)


def final_output_complete(path: str, metadata: Dict, num_samples: int) -> bool:
    if not os.path.exists(path):
        return False
    meta_path = metadata_path(path)
    if not os.path.exists(meta_path):
        raise ValueError(f"missing metadata for existing output {path}")
    validate_metadata(meta_path, metadata, ignored_keys=("batch_size",))
    rows, _ = _read_jsonl(path, allow_truncated_last=False)
    if len(rows) != num_samples:
        raise ValueError(f"{path} has {len(rows)} rows, expected {num_samples}")
    _validate_sequential(rows, 0, num_samples)
    return True


def read_generated_rows(path: str) -> List[Tuple[int, str]]:
    rows, _ = _read_jsonl(path, allow_truncated_last=False)
    return [(int(row["id"]), row["generated"]) for row in rows]


def read_generated_records(path: str) -> List[Dict]:
    rows, _ = _read_jsonl(path, allow_truncated_last=False)
    return rows


def merge_rank_shards(final_path: str, metadata: Dict, num_samples: int, world: int):
    meta_path = metadata_path(final_path)
    if os.path.exists(meta_path):
        validate_metadata(meta_path, metadata, ignored_keys=("batch_size",))
    else:
        write_metadata(meta_path, metadata)

    tmp_path = final_tmp_path(final_path)
    expected_id = 0
    with open(tmp_path, "w", encoding="utf-8") as out:
        for rank in range(world):
            start_id, end_id = rank_range(num_samples, world, rank)
            if start_id == end_id:
                continue
            path = shard_path(final_path, rank)
            if not os.path.exists(path):
                raise ValueError(f"missing completed shard {path}")
            rows, _ = _read_jsonl(path, allow_truncated_last=False)
            if len(rows) != end_id - start_id:
                raise ValueError(f"{path} has {len(rows)} rows, expected {end_id - start_id}")
            _validate_sequential(rows, start_id, end_id)
            for row in rows:
                if row["id"] != expected_id:
                    raise ValueError(f"expected merged id {expected_id}, found {row['id']}")
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                expected_id += 1
    if expected_id != num_samples:
        raise ValueError(f"merged {expected_id} rows, expected {num_samples}")
    os.replace(tmp_path, final_path)


def cleanup_rank_shards(final_path: str, world: int):
    for rank in range(world):
        path = shard_path(final_path, rank)
        for stale in (path, metadata_path(path)):
            if os.path.exists(stale):
                os.remove(stale)


def metrics_has_entry(
    path: str, epoch: int, step: int, metric: str = None,
) -> bool:
    if not os.path.exists(path):
        return False
    rows, truncated = _read_jsonl(path, allow_truncated_last=True)
    if truncated:
        write_jsonl(path, rows)
    return any(
        row.get("epoch") == epoch
        and row.get("step") == step
        and (metric is None or row.get("metric") == metric)
        for row in rows
    )


def append_metrics_once(path: str, row: Dict) -> bool:
    if metrics_has_entry(
        path, int(row["epoch"]), int(row["step"]), metric=row.get("metric"),
    ):
        return False
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
    return True
