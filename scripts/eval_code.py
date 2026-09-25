#!/usr/bin/env python3
"""Evaluate complete Python solutions with pinned EvalPlus benchmarks."""

import argparse
import ast
import json
import os
import platform
from collections import Counter


EVALPLUS_COMMIT = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
HUMANEVAL_PLUS_VERSION = "v0.1.10"
MBPP_PLUS_VERSION = "v0.2.0"

BENCHMARKS = {
    "humaneval": ("humanevalplus", HUMANEVAL_PLUS_VERSION),
    "mbpp": ("mbppplus", MBPP_PLUS_VERSION),
}


def _load_evalplus():
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.evaluate import evaluate
    from evalplus.sanitize import sanitize

    return get_human_eval_plus, get_mbpp_plus, sanitize, evaluate


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_generations(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line in {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object in {path}:{line_number}")
            if not isinstance(row.get("task_id"), str):
                raise ValueError(f"missing task_id in {path}:{line_number}")
            if not isinstance(row.get("generated"), str):
                raise ValueError(f"missing generated text in {path}:{line_number}")
            if not isinstance(row.get("eos_emitted"), bool):
                raise ValueError(f"missing eos_emitted in {path}:{line_number}")
            rows.append(row)
    return rows


def _infer_benchmark(rows, problems_by_benchmark):
    task_ids = [row["task_id"] for row in rows]
    duplicates = sorted(
        task_id for task_id, count in Counter(task_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate task IDs: {duplicates}")

    observed = set(task_ids)
    expected = {
        benchmark: set(problems)
        for benchmark, problems in problems_by_benchmark.items()
    }
    for benchmark, expected_ids in expected.items():
        if observed == expected_ids:
            return benchmark

    unknown = observed - set().union(*expected.values())
    if unknown:
        raise ValueError(f"unknown task IDs: {sorted(unknown)}")

    for benchmark, expected_ids in expected.items():
        if observed <= expected_ids:
            raise ValueError(
                f"missing {benchmark} task IDs: {sorted(expected_ids - observed)}"
            )
    raise ValueError("task IDs mix benchmarks instead of matching one complete benchmark")


def _syntax_valid(code):
    if not code.strip():
        return False
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def _rate(count, total):
    return count / total if total else 0.0


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _execution_statuses(native_result):
    base = Counter()
    plus = Counter()
    for task_results in native_result["eval"].values():
        for result in task_results:
            base[result["base_status"]] += 1
            plus[result["plus_status"]] += 1
    return {"base": dict(sorted(base.items())), "plus": dict(sorted(plus.items()))}


def evaluate_generations(
    json_file,
    metadata_file,
    output_file,
    *,
    source_path=None,
    dependencies=None,
):
    generation_metadata = _read_json(metadata_file)
    rows = _read_generations(json_file)
    if dependencies is None:
        dependencies = _load_evalplus()
    get_humaneval, get_mbpp, sanitize, evaluate = dependencies

    problems_by_benchmark = {
        "humaneval": get_humaneval(version=HUMANEVAL_PLUS_VERSION),
        "mbpp": get_mbpp(version=MBPP_PLUS_VERSION),
    }
    benchmark = _infer_benchmark(rows, problems_by_benchmark)
    problems = problems_by_benchmark[benchmark]

    samples = []
    raw_syntax = 0
    sanitized_syntax = 0
    sanitizer_changes = 0
    eos_count = 0
    for row in rows:
        raw = row["generated"]
        sanitized = sanitize(raw, entrypoint=problems[row["task_id"]]["entry_point"])
        if not isinstance(sanitized, str):
            raise TypeError("EvalPlus sanitizer returned a non-string value")
        raw_syntax += _syntax_valid(raw)
        sanitized_syntax += _syntax_valid(sanitized)
        sanitizer_changes += sanitized != raw
        eos_count += row["eos_emitted"]
        samples.append({"task_id": row["task_id"], "solution": sanitized})

    work_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(work_dir, exist_ok=True)
    samples_path = os.path.join(work_dir, "evalplus_samples.jsonl")
    native_path = os.path.join(work_dir, "evalplus_native.json")
    _write_jsonl(samples_path, samples)

    _, benchmark_version = BENCHMARKS[benchmark]
    evaluate(
        dataset=benchmark,
        samples=samples_path,
        base_only=False,
        parallel=2,
        version=benchmark_version,
        output_file=native_path,
    )
    native_result = _read_json(native_path)

    total = len(rows)
    benchmark_name, _ = BENCHMARKS[benchmark]
    result = {
        "source_path": source_path or os.path.abspath(json_file),
        "generation_metadata": generation_metadata,
        "benchmark": benchmark_name,
        "versions": {
            "evalplus_commit": EVALPLUS_COMMIT,
            "humanevalplus": HUMANEVAL_PLUS_VERSION,
            "mbppplus": MBPP_PLUS_VERSION,
            "python": platform.python_version(),
        },
        "diagnostics": {
            "total": total,
            "raw_syntax": {"valid": raw_syntax, "rate": _rate(raw_syntax, total)},
            "sanitized_syntax": {
                "valid": sanitized_syntax,
                "rate": _rate(sanitized_syntax, total),
            },
            "sanitizer_changes": {
                "count": sanitizer_changes,
                "rate": _rate(sanitizer_changes, total),
            },
            "eos": {"count": eos_count, "rate": _rate(eos_count, total)},
            "execution_statuses": _execution_statuses(native_result),
        },
        "evalplus_result": native_result,
    }

    temporary_output = output_file + ".tmp"
    with open(temporary_output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_output, output_file)

    pass_at_k = native_result["pass_at_k"]
    base = pass_at_k["base"]["pass@1"]
    plus = pass_at_k["plus"]["pass@1"]
    print(
        f"{benchmark_name}: base pass@1={base:.4f}, plus pass@1={plus:.4f}, "
        f"raw syntax={_rate(raw_syntax, total):.4f}, "
        f"sanitized syntax={_rate(sanitized_syntax, total):.4f}"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--metadata_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--source_path")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_generations(
        args.json_file,
        args.metadata_file,
        args.output_file,
        source_path=args.source_path,
    )


if __name__ == "__main__":
    main()
