#!/usr/bin/env python
"""Evaluate original MBPP-500 assertions inside the existing EvalPlus container."""

import argparse
import json
import math
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import eval_code

from eval_code_variants import SourceTransformer


MODE = "sanitize_alias_single"
SETTINGS = dict(workers=2, max_memory_bytes_per_program=6 * 1024**3, program_timeout_seconds=20.0)
VERSIONS = dict(evalplus_commit=eval_code.EVALPLUS_COMMIT, mbpp500_evaluator="1")


def load_tasks(path):
    tasks = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if ([r["task_id"] for r in tasks] != [f"Mbpp/{i}" for i in range(11, 511)]
            or [r["index"] for r in tasks] != list(range(500))):
        raise ValueError("expected all 500 ordered MBPP task IDs")
    for task in tasks:
        if (task["source_id"] != int(task["task_id"].split("/")[1]) or len(task["test_list"]) != 3
                or not task["entry_point"].isidentifier()
                or not all(isinstance(t, str) for t in task["test_list"])
                or not isinstance(task["test_setup_code"], str)):
            raise ValueError("invalid MBPP-500 evaluation task")
    return tasks


def assemble(solution, task):
    program = solution + "\n" + task["test_setup_code"] + "\n" + "\n".join(task["test_list"])
    # The guarded callable contains all untrusted execution, including module setup.
    # EvalPlus converts caught timer exceptions into ordinary failures. Exiting the
    # guarded child leaves its status unset, which its parent reports as timeout.
    return ("def __mbpp500_check__():\n"
            "    from evalplus.eval.utils import TimeoutException\n"
            "    import os\n"
            "    try:\n"
            f"        exec({program!r}, {{}})\n"
            "    except TimeoutException:\n"
            "        os._exit(124)\n"
            "    return True\n")


def score_one(item, *, dependencies=None):
    row, task = item
    if dependencies is None:
        from evalplus.eval import untrusted_check
        from evalplus.sanitize import sanitize
    else:
        sanitize, untrusted_check = dependencies
    os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = str(SETTINGS["max_memory_bytes_per_program"])
    transformer = SourceTransformer(MODE, sanitize)
    solution = transformer(row["generated"], entrypoint=task["entry_point"])
    status, _ = untrusted_check(dataset="mbpp500", code=assemble(solution, task), inputs=[()],
                               entry_point="__mbpp500_check__", expected=[True], atol=0,
                               ref_time=[0.0], min_time_limit=SETTINGS["program_timeout_seconds"],
                               gt_time_limit_factor=1.0)
    if status not in ("pass", "fail", "timeout"):
        raise ValueError(f"unexpected execution status: {status}")
    return dict(task_id=task["task_id"], base_status=status, solution=solution,
                transformation_counts=dict(transformer.counters))


def outcomes(result, saved, task_ids):
    if (result["generation_metadata"] != saved or result["benchmark"] != "mbpp500"
            or result["evaluation_mode"] != MODE or result["evaluation_settings"] != SETTINGS
            or any(result["versions"].get(k) != v for k, v in VERSIONS.items())):
        raise ValueError("incompatible MBPP-500 result")
    records = result["eval"]
    if [r["task_id"] for r in records] != task_ids:
        raise ValueError("missing, duplicate, or reordered MBPP-500 outcomes")
    if any(r["base_status"] not in ("pass", "fail", "timeout") for r in records):
        raise ValueError("unknown MBPP-500 execution status")
    scores = {r["task_id"]: dict(base_status=r["base_status"], base_correct=int(r["base_status"] == "pass"))
              for r in records}
    accuracy = sum(r["base_correct"] for r in scores.values()) / len(task_ids)
    if not math.isclose(accuracy, result["pass_at_1"], abs_tol=1e-12):
        raise ValueError("MBPP-500 pass@1 disagrees with outcomes")
    return scores


def evaluate(json_file, metadata_file, tasks_file, output_file):
    rows = eval_code._read_generations(json_file)
    saved = json.loads(Path(metadata_file).read_text())
    tasks = load_tasks(tasks_file)
    if (saved["benchmark"] != "mbpp500" or saved["questions"] != 500
            or [r["task_id"] for r in rows] != [r["task_id"] for r in tasks]):
        raise ValueError("MBPP-500 generation or tasks differ from metadata")
    if any(row["id"] != i or any(row.get(k) != v for k, v in saved.items() if k != "generation_sha256")
           for i, row in enumerate(rows)):
        raise ValueError("invalid MBPP-500 generation provenance")
    with ProcessPoolExecutor(max_workers=SETTINGS["workers"]) as pool:
        records = list(pool.map(score_one, zip(rows, tasks)))
    result = dict(benchmark="mbpp500", generation_metadata=saved, evaluation_mode=MODE,
                  evaluation_settings=SETTINGS, versions=dict(VERSIONS, python=platform.python_version()),
                  eval=records, pass_at_1=sum(r["base_status"] == "pass" for r in records) / len(tasks))
    outcomes(result, saved, [r["task_id"] for r in tasks])
    output = Path(output_file)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(f"MBPP-500 pass@1={result['pass_at_1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--metadata_file", required=True)
    parser.add_argument("--tasks_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    evaluate(args.json_file, args.metadata_file, args.tasks_file, args.output_file)
