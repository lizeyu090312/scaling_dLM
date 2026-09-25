#!/usr/bin/env python
"""Prepare the pinned MBPP full/test split for headline code evaluation; never alter training data."""

import argparse
import ast
import io
import json
import tempfile
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoTokenizer

import prepare_opencodeinstruct_python_qwen3 as code


SOURCE = "google-research-datasets/mbpp"
REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
SOURCE_FILE = "full/test-00000-of-00001.parquet"
OUTPUT = Path(code.OUTPUT_DIR) / "mbpp500_test"


def read_source(raw):
    rows = pq.read_table(io.BytesIO(raw)).to_pylist()
    if [r["task_id"] for r in rows] != list(range(11, 511)):
        raise ValueError("expected exactly ordered MBPP task IDs 11–510")
    return rows


def entry_point(row):
    definitions = {n.name for n in ast.parse(row["code"]).body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    calls = {n.func.id for test in row["test_list"] for n in ast.walk(ast.parse(test))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    names = definitions & calls
    if len(names) != 1:
        raise ValueError(f"ambiguous entry point for MBPP task {row['task_id']}: {names}")
    return names.pop()


def prompt_for(row, tokenizer):
    if len(row["test_list"]) != 3:
        raise ValueError("expected three public assertions")
    for count in range(3, -1, -1):
        task = row["text"]
        if count:
            task += "\n\nYour code should pass these tests:\n" + "\n".join(row["test_list"][:count])
        prompt = code.build_code_prompt(task)
        if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) <= code.MAX_INPUT_LENGTH:
            return prompt, count
    raise ValueError(f"statement exceeds input limit: {row['task_id']}")


def prepare_rows(rows, tokenizer):
    examples, tasks = [], []
    for index, row in enumerate(rows):
        prompt, count = prompt_for(row, tokenizer)
        target = row["code"].replace("\r\n", "\n").strip()
        condition, response = code.tokenize_code_examples([prompt], [target], tokenizer)[0]
        reason = code.length_filter_reason(condition, response)
        if reason:
            raise ValueError(f"MBPP task {row['task_id']}: {reason}")
        task_id = f"Mbpp/{row['task_id']}"
        examples.append(dict(index=index, source="mbpp500", source_index=index, task_id=task_id,
                             input=prompt, target=target, condition_input_ids=condition, input_ids=response))
        tasks.append(dict(index=index, source_id=row["task_id"], task_id=task_id, text=row["text"],
                          prompt=prompt, prompt_assertions=count, entry_point=entry_point(row),
                          test_list=row["test_list"], test_setup_code=row["test_setup_code"]))
    return Dataset.from_list(examples, features=code.BENCHMARK_FEATURES), tasks


def prepare(destination, source_path=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if source_path:
        raw = Path(source_path).read_bytes()
    else:
        url = f"https://huggingface.co/datasets/{SOURCE}/resolve/{REVISION}/{SOURCE_FILE}"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
    rows = read_source(raw)
    tokenizer = AutoTokenizer.from_pretrained(code.TOKENIZER_NAME, revision=code.TOKENIZER_REVISION)
    dataset, tasks = prepare_rows(rows, tokenizer)
    shortened = {r["source_id"]: r["prompt_assertions"] for r in tasks if r["prompt_assertions"] != 3}
    if shortened != {380: 2, 462: 1, 493: 0}:
        raise ValueError(f"unexpected prompt example counts: {shortened}")
    metadata = dict(version="mbpp500_v1", source=SOURCE, revision=REVISION, source_file=SOURCE_FILE,
                    tokenizer=code.TOKENIZER_NAME, tokenizer_revision=code.TOKENIZER_REVISION, questions=len(tasks),
                    task_ids=[r["task_id"] for r in tasks], max_input_length=code.MAX_INPUT_LENGTH,
                    max_length=code.MAX_LENGTH, shortened_prompt_assertions=shortened)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mbpp500-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "dataset"
        dataset.save_to_disk(str(staged))
        (staged / "source.parquet").write_bytes(raw)
        (staged / "evaluation_tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tasks))
        (staged / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        staged.rename(destination)
    print(f"Prepared {len(dataset)} MBPP-500 tasks: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--source-parquet", type=Path)
    args = parser.parse_args()
    prepare(args.output_dir, args.source_parquet)
