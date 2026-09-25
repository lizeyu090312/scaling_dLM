#!/usr/bin/env python
"""Prepare OpenCodeInstruct Python data and pinned EvalPlus benchmarks."""

import argparse
import ast
import gzip
import json
import os
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation

import numpy as np
from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer


OPENCODE_SOURCE = "nvidia/OpenCodeInstruct"
OPENCODE_CONFIG = "train"
OPENCODE_SPLIT = "train"
OPENCODE_REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
EXPECTED_OPENCODE_ROWS = 5_000_000

TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"
TOKENIZER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

HUMANEVAL_PLUS_VERSION = "v0.1.10"
MBPP_PLUS_VERSION = "v0.2.0"
HUMANEVAL_PLUS_URL = (
    "https://github.com/evalplus/humanevalplus_release/releases/download/"
    f"{HUMANEVAL_PLUS_VERSION}/HumanEvalPlus.jsonl.gz"
)
MBPP_PLUS_URL = (
    "https://github.com/evalplus/mbppplus_release/releases/download/"
    f"{MBPP_PLUS_VERSION}/MbppPlus.jsonl.gz"
)
EXPECTED_HUMANEVAL_PLUS_ROWS = 164
EXPECTED_MBPP_PLUS_ROWS = 378

OUTPUT_DIR = (
    "data/opencodeinstruct-python-qwen3-embedding-0.6b-"
    "input512-max1024-v1"
)
MAX_INPUT_LENGTH = 512
MAX_LENGTH = 1_024
WORD_8GRAM_JACCARD_THRESHOLD = 0.5
TOKENIZE_BATCH_SIZE = 1_000
DEFAULT_NUM_PROC = min(16, os.cpu_count() or 1)

PROMPT_TEMPLATE = (
    "Write a complete Python solution. Return only code.\n\n"
    "{task}\n\n"
    "Python solution:\n"
)
JUDGE_FIELDS = (
    "requirement_conformance",
    "logical_correctness",
    "edge_case_consideration",
)
EXPECTED_DOMAINS = frozenset({"generic", "algorithmic"})
EXPECTED_GENERATION_ALGORITHMS = frozenset({"self-instruct", "evol-instruct"})
EXPECTED_SOURCE_COLUMNS = frozenset({
    "id",
    "input",
    "output",
    "domain",
    "generation_algorithm",
    "llm_judgement",
    "unit_tests",
    "tests_execution_status",
    "average_test_score",
})

_OUTER_PYTHON_FENCE = re.compile(
    r"\A[ \t\n]*```[ \t]*python[ \t]*\n(?P<code>.*)\n```[ \t\n]*\Z",
    re.IGNORECASE | re.DOTALL,
)

INTERMEDIATE_FEATURES = Features({
    "source": Value("string"),
    "source_index": Value("int32"),
    "source_id": Value("string"),
    "source_subtype": Value("string"),
    "generation_algorithm": Value("string"),
    "average_test_score": Value("float32"),
    "judge_scores": Sequence(Value("int8")),
    "normalized_task": Value("string"),
    "input": Value("string"),
    "target": Value("string"),
    "condition_input_ids": Sequence(Value("int32")),
    "input_ids": Sequence(Value("int32")),
    "filter_reason": Value("string"),
    "contamination_task_id": Value("string"),
    "contamination_evidence": Value("string"),
})

TRAIN_FEATURES = Features({
    "index": Value("int32"),
    "source": Value("string"),
    "source_index": Value("int32"),
    "source_id": Value("string"),
    "source_subtype": Value("string"),
    "generation_algorithm": Value("string"),
    "average_test_score": Value("float32"),
    "input": Value("string"),
    "target": Value("string"),
    "condition_input_ids": Sequence(Value("int32")),
    "input_ids": Sequence(Value("int32")),
})

BENCHMARK_FEATURES = Features({
    "index": Value("int32"),
    "source": Value("string"),
    "source_index": Value("int32"),
    "task_id": Value("string"),
    "input": Value("string"),
    "target": Value("string"),
    "condition_input_ids": Sequence(Value("int32")),
    "input_ids": Sequence(Value("int32")),
})


def _normalized_newlines(value):
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _required_text(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field}_must_be_string")
    value = _normalized_newlines(value).strip()
    if not value:
        raise ValueError(f"missing_{field}")
    return value


def build_code_prompt(task):
    return PROMPT_TEMPLATE.format(task=_required_text(task, "task"))


def normalize_match_text(value):
    value = _required_text(value, "comparison_text")
    return " ".join(unicodedata.normalize("NFC", value).lower().split())


def word_ngrams(value, size=8):
    words = value.split()
    return [
        " ".join(words[index:index + size])
        for index in range(len(words) - size + 1)
    ]


def canonical_python_ast(code_or_tree):
    tree = code_or_tree if isinstance(code_or_tree, ast.AST) else ast.parse(code_or_tree)
    return ast.dump(tree, include_attributes=False)


def _parse_python_output(value):
    if not isinstance(value, str):
        raise ValueError("output_must_be_string")
    value = _normalized_newlines(value)
    match = _OUTER_PYTHON_FENCE.fullmatch(value)
    if match:
        code = match.group("code").strip()
    else:
        code = value.strip()
    if not code:
        raise ValueError("missing_code")
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        if "```" in code:
            raise ValueError("invalid_code_fence") from error
        raise ValueError("invalid_python") from error
    return code, tree


def normalize_python_output(value):
    return _parse_python_output(value)[0]


def parse_quality_metadata(
    llm_judgement, unit_tests, tests_execution_status, average_test_score,
):
    try:
        judgement = json.loads(llm_judgement)
        tests = json.loads(unit_tests)
        statuses = json.loads(tests_execution_status)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid_serialized_metadata") from error

    if not isinstance(judgement, dict) or not set(JUDGE_FIELDS) <= set(judgement):
        raise ValueError("invalid_judgement_fields")
    scores = []
    for field in JUDGE_FIELDS:
        item = judgement[field]
        if not isinstance(item, dict) or set(item) != {"score", "justification"}:
            raise ValueError("invalid_judgement_item")
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError("invalid_judgement_score")
        if not isinstance(item["justification"], str) or not item["justification"].strip():
            raise ValueError("invalid_judgement_justification")
        scores.append(score)

    if (
        not isinstance(tests, list)
        or len(tests) != 10
        or any(not isinstance(test, str) for test in tests)
    ):
        raise ValueError("invalid_unit_tests")
    if (
        not isinstance(statuses, list)
        or len(statuses) != 10
        or any(status not in {"pass", "fail"} for status in statuses)
    ):
        raise ValueError("invalid_test_statuses")
    try:
        test_score = Decimal(str(average_test_score))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("invalid_average_test_score") from error
    pass_count = statuses.count("pass")
    if test_score < 0 or test_score > 1 or test_score * 10 != pass_count:
        raise ValueError("inconsistent_average_test_score")
    return scores, float(test_score)


def tokenize_code_examples(prompts, targets, tokenizer):
    if len(prompts) != len(targets):
        raise ValueError("prompts and targets must have the same length")
    kwargs = {
        "add_special_tokens": False,
        "return_attention_mask": False,
        "return_token_type_ids": False,
    }
    condition_rows = tokenizer(prompts, **kwargs)["input_ids"]
    response_rows = tokenizer(targets, **kwargs)["input_ids"]
    joint_rows = tokenizer(
        [prompt + target for prompt, target in zip(prompts, targets)], **kwargs,
    )["input_ids"]
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    rows = []
    for condition_ids, response_ids, joint_ids in zip(
        condition_rows, response_rows, joint_rows,
    ):
        if condition_ids + response_ids != joint_ids:
            raise ValueError(
                "separate prompt/response tokenization does not match joint tokenization"
            )
        rows.append((
            list(condition_ids),
            list(response_ids) + [int(tokenizer.eos_token_id)],
        ))
    return rows


def length_filter_reason(
    condition_ids, response_ids, max_input_length=MAX_INPUT_LENGTH,
    max_length=MAX_LENGTH,
):
    if len(condition_ids) > max_input_length:
        return "input_too_long"
    if len(condition_ids) + len(response_ids) > max_length:
        return "sequence_too_long"
    return ""


def _download_release(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, path)
    return path


def load_gzip_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_evalplus_releases(cache_dir):
    release_dir = os.path.join(cache_dir, "evalplus_releases")
    humaneval_path = _download_release(
        HUMANEVAL_PLUS_URL,
        os.path.join(release_dir, f"HumanEvalPlus-{HUMANEVAL_PLUS_VERSION}.jsonl.gz"),
    )
    mbpp_path = _download_release(
        MBPP_PLUS_URL,
        os.path.join(release_dir, f"MbppPlus-{MBPP_PLUS_VERSION}.jsonl.gz"),
    )
    humaneval = load_gzip_jsonl(humaneval_path)
    mbpp = load_gzip_jsonl(mbpp_path)
    _validate_benchmark_rows(
        humaneval, EXPECTED_HUMANEVAL_PLUS_ROWS, "HumanEval/", "HumanEval+",
    )
    _validate_benchmark_rows(
        mbpp, EXPECTED_MBPP_PLUS_ROWS, "Mbpp/", "MBPP+",
    )
    return humaneval, mbpp


def _validate_benchmark_rows(rows, expected_count, task_prefix, name):
    task_ids = [row.get("task_id") for row in rows]
    if len(rows) != expected_count:
        raise ValueError(f"{name} has {len(rows)} rows, expected {expected_count}")
    if len(set(task_ids)) != expected_count:
        raise ValueError(f"{name} task IDs are not unique")
    if any(not isinstance(task_id, str) or not task_id.startswith(task_prefix) for task_id in task_ids):
        raise ValueError(f"{name} has invalid task IDs")
    required = {"task_id", "prompt", "canonical_solution", "entry_point"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"{name} rows are missing required fields")


def full_benchmark_solution(row, source):
    if source == "humanevalplus":
        code = row["prompt"] + row["canonical_solution"]
    elif source == "mbppplus":
        code = row["canonical_solution"]
    else:
        raise ValueError(f"unsupported benchmark source: {source}")
    code = _normalized_newlines(_required_text(code, "canonical_solution"))
    try:
        ast.parse(code)
    except SyntaxError as error:
        raise ValueError(f"invalid {source} canonical solution") from error
    return code


def _benchmark_text_variants(prompt):
    prompt = _required_text(prompt, "benchmark_prompt")
    variants = {prompt}
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return variants
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node, clean=False)
            if docstring and docstring.strip():
                variants.add(docstring.strip())
                without_assertions = "\n".join(
                    line for line in docstring.splitlines()
                    if not line.lstrip().startswith("assert ")
                ).strip()
                if without_assertions:
                    variants.add(without_assertions)
    return variants


def build_contamination_index(humaneval_rows, mbpp_rows):
    exact_prompts = defaultdict(set)
    word_8grams = defaultdict(set)
    prompt_gram_sets = defaultdict(list)
    solution_asts = defaultdict(set)
    for source, rows in (
        ("humanevalplus", humaneval_rows),
        ("mbppplus", mbpp_rows),
    ):
        for row in rows:
            task_id = row["task_id"]
            for variant in _benchmark_text_variants(row["prompt"]):
                normalized = normalize_match_text(variant)
                exact_prompts[normalized].add(task_id)
                grams = frozenset(word_ngrams(normalized))
                if grams and grams not in prompt_gram_sets[task_id]:
                    prompt_gram_sets[task_id].append(grams)
                for gram in grams:
                    word_8grams[gram].add(task_id)
            solution_asts[canonical_python_ast(
                full_benchmark_solution(row, source)
            )].add(task_id)
    return {
        "exact_prompts": {
            key: sorted(value) for key, value in exact_prompts.items()
        },
        "word_8grams": {
            key: sorted(value) for key, value in word_8grams.items()
        },
        "prompt_gram_sets": dict(prompt_gram_sets),
        "solution_asts": {
            key: sorted(value) for key, value in solution_asts.items()
        },
    }


def find_contamination(task, code, contamination_index, solution_ast=None):
    normalized = normalize_match_text(task)
    matches = contamination_index["exact_prompts"].get(normalized)
    if matches:
        return "benchmark_prompt_exact", matches[0], normalized
    candidate_grams = set(word_ngrams(normalized))
    proposed_task_ids = set()
    for gram in candidate_grams:
        proposed_task_ids.update(contamination_index["word_8grams"].get(gram, ()))
    for task_id in sorted(proposed_task_ids):
        for reference_grams in contamination_index["prompt_gram_sets"][task_id]:
            shared_grams = candidate_grams & reference_grams
            overlap = len(shared_grams) / len(candidate_grams | reference_grams)
            if overlap >= WORD_8GRAM_JACCARD_THRESHOLD:
                return (
                    "benchmark_prompt_word_8gram_jaccard",
                    task_id,
                    f"jaccard={overlap:.3f}; matched_8gram={min(shared_grams)}",
                )
    solution_ast = solution_ast or canonical_python_ast(code)
    matches = contamination_index["solution_asts"].get(solution_ast)
    if matches:
        return "benchmark_solution_ast_exact", matches[0], "exact AST"
    return "", "", ""


def _empty_intermediate_columns():
    return {name: [] for name in INTERMEDIATE_FEATURES}


def prepare_training_batch(
    batch, indices, tokenizer, contamination_index,
    max_input_length=MAX_INPUT_LENGTH, max_length=MAX_LENGTH,
):
    output = _empty_intermediate_columns()
    pending = []
    prompts = []
    targets = []
    columns = list(EXPECTED_SOURCE_COLUMNS)
    for row_number, values in enumerate(zip(*(batch[name] for name in columns))):
        row = dict(zip(columns, values))
        source_index = int(indices[row_number])
        try:
            source_id = _required_text(row["id"], "id")
            task = _required_text(row["input"], "input")
            if row["domain"] not in EXPECTED_DOMAINS:
                raise ValueError("invalid_domain")
            if row["generation_algorithm"] not in EXPECTED_GENERATION_ALGORITHMS:
                raise ValueError("invalid_generation_algorithm")
            scores, test_score = parse_quality_metadata(
                row["llm_judgement"],
                row["unit_tests"],
                row["tests_execution_status"],
                row["average_test_score"],
            )
        except ValueError as error:
            raise ValueError(f"source row {source_index}: {error}") from error

        reason = ""
        code = ""
        solution_ast = ""
        contamination_task_id = ""
        contamination_evidence = ""
        if min(scores) < 4:
            reason = "judge_score_below_4"
        else:
            try:
                code, code_tree = _parse_python_output(row["output"])
                solution_ast = canonical_python_ast(code_tree)
            except ValueError as error:
                reason = str(error)
        if not reason:
            (
                reason,
                contamination_task_id,
                contamination_evidence,
            ) = find_contamination(
                task, code, contamination_index, solution_ast=solution_ast,
            )

        record_index = len(output["source_index"])
        output["source"].append("opencodeinstruct")
        output["source_index"].append(source_index)
        output["source_id"].append(source_id)
        output["source_subtype"].append(row["domain"])
        output["generation_algorithm"].append(row["generation_algorithm"])
        output["average_test_score"].append(test_score)
        output["judge_scores"].append(scores)
        output["normalized_task"].append(normalize_match_text(task))
        output["input"].append("")
        output["target"].append("")
        output["condition_input_ids"].append([])
        output["input_ids"].append([])
        output["filter_reason"].append(reason)
        output["contamination_task_id"].append(contamination_task_id)
        output["contamination_evidence"].append(contamination_evidence)
        if not reason:
            prompt = build_code_prompt(task)
            pending.append(record_index)
            prompts.append(prompt)
            targets.append(code)

    if pending:
        tokenized = tokenize_code_examples(prompts, targets, tokenizer)
        for record_index, prompt, target, (condition_ids, response_ids) in zip(
            pending, prompts, targets, tokenized,
        ):
            reason = length_filter_reason(
                condition_ids, response_ids, max_input_length, max_length,
            )
            output["filter_reason"][record_index] = reason
            output["input"][record_index] = prompt
            output["target"][record_index] = target
            output["condition_input_ids"][record_index] = condition_ids
            output["input_ids"][record_index] = response_ids
    return output


def prepare_benchmark_dataset(
    rows, source, tokenizer, max_input_length=MAX_INPUT_LENGTH,
    max_length=MAX_LENGTH,
):
    output = {name: [] for name in BENCHMARK_FEATURES}
    for source_index, row in enumerate(rows):
        prompt = build_code_prompt(row["prompt"])
        target = full_benchmark_solution(row, source)
        condition_ids, response_ids = tokenize_code_examples(
            [prompt], [target], tokenizer,
        )[0]
        reason = length_filter_reason(
            condition_ids, response_ids, max_input_length, max_length,
        )
        if reason:
            raise ValueError(f"{source} task {row['task_id']} is over length: {reason}")
        output["index"].append(source_index)
        output["source"].append(source)
        output["source_index"].append(source_index)
        output["task_id"].append(row["task_id"])
        output["input"].append(prompt)
        output["target"].append(target)
        output["condition_input_ids"].append(condition_ids)
        output["input_ids"].append(response_ids)
    return Dataset.from_dict(output, features=BENCHMARK_FEATURES)


def deduplicate_training_rows(dataset):
    seen = set()
    kept_indices = []
    offset = 0
    for batch in dataset.select_columns(["normalized_task"]).iter(batch_size=10_000):
        for index, key in enumerate(batch["normalized_task"]):
            if key not in seen:
                seen.add(key)
                kept_indices.append(offset + index)
        offset += len(batch["normalized_task"])
    return dataset.select(kept_indices), len(dataset) - len(kept_indices)


def finalize_training_dataset(dataset):
    columns = [name for name in TRAIN_FEATURES if name != "index"]
    missing = set(columns) - set(dataset.column_names)
    if missing:
        raise ValueError(f"training dataset is missing columns: {sorted(missing)}")
    dataset = dataset.select_columns(columns)
    dataset = dataset.add_column("index", list(range(len(dataset))))
    return dataset.select_columns(list(TRAIN_FEATURES)).cast(TRAIN_FEATURES)


def _reason_counts(dataset):
    return dict(sorted(Counter(dataset["filter_reason"]).items()))


def _value_counts(dataset, column):
    return {
        str(key): count
        for key, count in sorted(Counter(dataset[column]).items())
    }


def _judge_score_counts(dataset):
    counts = Counter()
    for batch in dataset.select_columns(["judge_scores"]).iter(batch_size=10_000):
        counts.update(tuple(scores) for scores in batch["judge_scores"])
    return {
        "/".join(map(str, scores)): count
        for scores, count in sorted(counts.items())
    }


def _test_score_counts(dataset):
    return {
        f"{float(score):.1f}": count
        for score, count in sorted(Counter(dataset["average_test_score"]).items())
    }


def token_length_statistics(dataset):
    prompt_lengths = []
    response_lengths = []
    for batch in dataset.select_columns(
        ["condition_input_ids", "input_ids"]
    ).iter(batch_size=10_000):
        prompt_lengths.extend(map(len, batch["condition_input_ids"]))
        response_lengths.extend(map(len, batch["input_ids"]))
    combined_lengths = [
        prompt + response
        for prompt, response in zip(prompt_lengths, response_lengths)
    ]

    def summarize(values):
        if not values:
            return {key: None for key in ("mean", "p50", "p90", "p95", "p99", "max")}
        array = np.asarray(values, dtype=np.int32)
        return {
            "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": int(array.max()),
        }

    return {
        "prompt": summarize(prompt_lengths),
        "response_including_eos": summarize(response_lengths),
        "combined": summarize(combined_lengths),
    }


def write_json(path, value):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_contamination_report(path, dataset):
    temporary = f"{path}.tmp"
    fields = (
        "source_index",
        "source_id",
        "filter_reason",
        "contamination_task_id",
        "contamination_evidence",
    )
    with open(temporary, "w", encoding="utf-8") as handle:
        selected = dataset.select_columns(fields)
        for batch in selected.iter(batch_size=1_000):
            for values in zip(*(batch[field] for field in fields)):
                row = dict(zip(fields, values))
                if row["contamination_task_id"]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--num_proc", type=int, default=DEFAULT_NUM_PROC)
    parser.add_argument("--max_input_length", type=int, default=MAX_INPUT_LENGTH)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    if args.num_proc <= 0:
        parser.error("--num_proc must be positive")
    if args.max_input_length <= 0 or args.max_length <= 0:
        parser.error("length limits must be positive")
    if args.max_input_length > args.max_length:
        parser.error("--max_input_length cannot exceed --max_length")
    return args


def main():
    args = parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    cache_dir = args.cache_dir or os.path.expanduser("~/.cache/elf_code_generation")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        revision=TOKENIZER_REVISION,
        cache_dir=cache_dir,
    )
    humaneval_rows, mbpp_rows = load_evalplus_releases(cache_dir)
    contamination_index = build_contamination_index(humaneval_rows, mbpp_rows)

    raw = load_dataset(
        OPENCODE_SOURCE,
        OPENCODE_CONFIG,
        split=OPENCODE_SPLIT,
        revision=OPENCODE_REVISION,
        cache_dir=cache_dir,
    )
    if set(raw.column_names) != EXPECTED_SOURCE_COLUMNS:
        raise ValueError(
            f"unexpected OpenCodeInstruct columns: {sorted(raw.column_names)}"
        )
    if len(raw) != EXPECTED_OPENCODE_ROWS:
        raise ValueError(
            f"OpenCodeInstruct has {len(raw)} rows, expected {EXPECTED_OPENCODE_ROWS}"
        )

    prepared = raw.map(
        lambda batch, indices: prepare_training_batch(
            batch,
            indices,
            tokenizer,
            contamination_index,
            args.max_input_length,
            args.max_length,
        ),
        batched=True,
        batch_size=TOKENIZE_BATCH_SIZE,
        with_indices=True,
        num_proc=None if args.num_proc == 1 else args.num_proc,
        remove_columns=raw.column_names,
        features=INTERMEDIATE_FEATURES,
        desc="Validating and tokenizing OpenCodeInstruct",
    )
    filter_counts = _reason_counts(prepared)
    accepted_before_deduplication = filter_counts.pop("", 0)
    accepted = prepared.filter(
        lambda reason: reason == "",
        input_columns=["filter_reason"],
        num_proc=None if args.num_proc == 1 else args.num_proc,
        desc="Keeping accepted OpenCodeInstruct rows",
    )
    deduplicated, duplicate_rows = deduplicate_training_rows(accepted)
    train = finalize_training_dataset(deduplicated)
    mbpp_test = prepare_benchmark_dataset(
        mbpp_rows, "mbppplus", tokenizer, args.max_input_length, args.max_length,
    )
    humaneval_test = prepare_benchmark_dataset(
        humaneval_rows,
        "humanevalplus",
        tokenizer,
        args.max_input_length,
        args.max_length,
    )

    os.makedirs(args.output_dir)
    benchmark_dir = os.path.join(args.output_dir, "benchmark_tests")
    os.makedirs(benchmark_dir)
    for name, rows in [(f"HumanEvalPlus-{HUMANEVAL_PLUS_VERSION}.jsonl", humaneval_rows),
                       (f"MbppPlus-{MBPP_PLUS_VERSION}.jsonl", mbpp_rows)]:
        with open(os.path.join(benchmark_dir, name), "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    train.save_to_disk(os.path.join(args.output_dir, "train"))
    mbpp_test.save_to_disk(os.path.join(args.output_dir, "mbppplus_test"))
    humaneval_test.save_to_disk(os.path.join(args.output_dir, "humanevalplus_test"))
    write_contamination_report(
        os.path.join(args.output_dir, "benchmark_contamination_matches.jsonl"),
        prepared,
    )

    metadata = {
        "version": 1,
        "sources": {
            "opencodeinstruct": {
                "dataset": OPENCODE_SOURCE,
                "revision": OPENCODE_REVISION,
                "raw_rows": len(raw),
            },
            "humanevalplus": {
                "version": HUMANEVAL_PLUS_VERSION,
                "rows": len(humaneval_test),
            },
            "mbppplus": {
                "version": MBPP_PLUS_VERSION,
                "rows": len(mbpp_test),
            },
        },
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "revision": TOKENIZER_REVISION,
            "eos_token_id": int(tokenizer.eos_token_id),
        },
        "prompt_template": PROMPT_TEMPLATE,
        "target_format": "complete bare Python source",
        "quality_filter": {
            "required_judge_fields": list(JUDGE_FIELDS),
            "minimum_score_per_field": 4,
            "test_score_used_as_filter": False,
        },
        "near_match_filter": {
            "word_ngram_size": 8,
            "jaccard_threshold": WORD_8GRAM_JACCARD_THRESHOLD,
        },
        "length_limits": {
            "max_input_length": args.max_input_length,
            "max_combined_length": args.max_length,
            "over_length_action": "reject",
        },
        "filter_counts": filter_counts,
        "accepted_before_exact_deduplication": accepted_before_deduplication,
        "exact_duplicate_rows_removed": duplicate_rows,
        "outputs": {
            "train": len(train),
            "mbppplus_test": len(mbpp_test),
            "humanevalplus_test": len(humaneval_test),
        },
        "training_distributions": {
            "domain": _value_counts(train, "source_subtype"),
            "generation_algorithm": _value_counts(train, "generation_algorithm"),
            "judge_scores": _judge_score_counts(deduplicated),
            "average_test_score": _test_score_counts(train),
        },
        "training_token_lengths": token_length_statistics(train),
        "benchmark_token_lengths": {
            "mbppplus": token_length_statistics(mbpp_test),
            "humanevalplus": token_length_statistics(humaneval_test),
        },
    }
    write_json(
        os.path.join(args.output_dir, "preprocessing_metadata.json"), metadata,
    )
    print(
        f"Saved {len(train):,} training rows, {len(mbpp_test)} MBPP+ rows, "
        f"and {len(humaneval_test)} HumanEval+ rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
