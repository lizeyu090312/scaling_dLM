#!/usr/bin/env python
"""Prepare OpenMathInstruct-2 MATH data and the MATH-500 benchmark."""

import argparse
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations

from datasets import Features, Sequence, Value, load_dataset
from datasketch import MinHash, MinHashLSH
from transformers import AutoTokenizer


OPENMATH_SOURCE = "nvidia/OpenMathInstruct-2"
OPENMATH_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"
MATH500_SOURCE = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
TOKENIZER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"
OUTPUT_DIR = "data/openmath-math-qwen3-embedding-0.6b-v1"

OPENMATH_SUBTYPES = frozenset({"math", "augmented_math"})
OPENMATH_SPLIT_COUNTS = {
    "train": {"rows": 13_972_791, "math_rows": 11_402_286},
    "train_5M": {"rows": 5_000_000, "math_rows": 4_194_072},
}
EXPECTED_MATH500_ROWS = 500

MAX_INPUT_LENGTH = 800
MAX_RESPONSE_LENGTH = 800
MAX_LENGTH = 1_600
TOKENIZE_BATCH_SIZE = 1_000
DEFAULT_NUM_PROC = min(10, os.cpu_count() or 1)

SPLIT_SEED = 20_260_702
VALIDATION_FRACTION = 0.10
VALIDATION_SUBSET_SEED = 20_260_705
VALIDATION_SUBSET_SIZE = 1_024

TEST_SUBSTRING_THRESHOLD = 0.85
SPLIT_SUBSTRING_THRESHOLD = 0.90
WORD_5GRAM_THRESHOLD = 0.55
WORD_8GRAM_THRESHOLD = 0.40
MINHASH_NUM_PERM = 128
MINHASH_SEED = 20_260_702
EXHAUSTIVE_PAIR_LIMIT = 100_000

# LSH proposes candidates only. question_similarity makes the final decision.
LSH_PROPOSAL_THRESHOLDS = {
    "tokens": 0.50,
    "characters": 0.50,
    "word_5grams": 0.40,
    "word_8grams": 0.30,
}

ALIGNMENT_ENVIRONMENTS = frozenset({
    "pmatrix",
    "bmatrix",
    "array",
    "cases",
    "matrix",
    "vmatrix",
})
_BEGIN_END_RE = re.compile(r"\\(begin|end)\s*\{([^{}]+)\}")

FINAL_FEATURES = Features({
    "index": Value("int32"),
    "source": Value("string"),
    "source_index": Value("int32"),
    "source_subtype": Value("string"),
    "problem_group_id": Value("string"),
    "gold_answer": Value("string"),
    "input": Value("string"),
    "target": Value("string"),
    "condition_input_ids": Sequence(Value("int32")),
    "input_ids": Sequence(Value("int32")),
})

PREPARED_FEATURES = Features({
    "source": Value("string"),
    "source_index": Value("int32"),
    "source_subtype": Value("string"),
    "question_key": Value("string"),
    "gold_answer": Value("string"),
    "raw_answer": Value("string"),
    "input": Value("string"),
    "target": Value("string"),
    "filter_reason": Value("string"),
})


def _normalize_nonempty_text(value, field_name):
    if not isinstance(value, str):
        raise ValueError(f"missing_{field_name}")
    value = value.strip()
    if not value:
        raise ValueError(f"missing_{field_name}")
    return value


def normalize_question(question):
    """Return a case- and operator-preserving exact comparison key."""
    question = _normalize_nonempty_text(question, "question")
    return " ".join(unicodedata.normalize("NFC", question).split())


def build_math_prompt(question):
    question = _normalize_nonempty_text(question, "question")
    return f"Question: {question}\nStep-by-Step Answer:"


def _is_escaped(text, index):
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _validate_braces_and_specials(answer):
    depth = 0
    special_reasons = {
        "$": "unescaped_dollar",
        "%": "unescaped_percent",
        "#": "unescaped_hash",
    }
    for index, character in enumerate(answer):
        if _is_escaped(answer, index):
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced_braces")
        elif character in special_reasons:
            raise ValueError(special_reasons[character])
    if depth:
        raise ValueError("unbalanced_braces")


def _validate_ampersands(answer):
    if not any(
        character == "&" and not _is_escaped(answer, index)
        for index, character in enumerate(answer)
    ):
        return

    events = {
        match.start(): match
        for match in _BEGIN_END_RE.finditer(answer)
        if not _is_escaped(answer, match.start())
    }
    stack = []
    index = 0
    while index < len(answer):
        match = events.get(index)
        if match is not None:
            kind, environment = match.group(1), match.group(2).strip()
            if environment in ALIGNMENT_ENVIRONMENTS:
                if kind == "begin":
                    stack.append(environment)
                elif not stack or stack[-1] != environment:
                    raise ValueError("invalid_alignment_environment")
                else:
                    stack.pop()
            index = match.end()
            continue
        if answer[index] == "&" and not _is_escaped(answer, index) and not stack:
            raise ValueError("unescaped_ampersand")
        index += 1
    if stack:
        raise ValueError("invalid_alignment_environment")


def validate_expected_answer(value):
    """Strip only outer whitespace and reject structurally unsafe TeX."""
    answer = _normalize_nonempty_text(value, "answer")
    _validate_braces_and_specials(answer)
    _validate_ampersands(answer)
    return answer


def authoritative_answer_suffix(gold_answer):
    return "\nFinal answer: $\\boxed{" + gold_answer + "}$"


def canonicalize_response(solution, expected_answer):
    solution = _normalize_nonempty_text(solution, "solution")
    gold_answer = validate_expected_answer(expected_answer)
    return solution + authoritative_answer_suffix(gold_answer), gold_answer


def _prepared_columns():
    return {name: [] for name in PREPARED_FEATURES}


def _append_prepared(
    output,
    source,
    source_index,
    source_subtype,
    question,
    solution,
    expected_answer,
):
    question_key = ""
    gold_answer = ""
    prompt = ""
    target = ""
    reason = ""
    try:
        question_key = normalize_question(question)
        prompt = build_math_prompt(question)
        target, gold_answer = canonicalize_response(solution, expected_answer)
    except ValueError as error:
        reason = str(error)

    output["source"].append(source)
    output["source_index"].append(int(source_index))
    output["source_subtype"].append(str(source_subtype))
    output["question_key"].append(question_key)
    output["gold_answer"].append(gold_answer)
    output["raw_answer"].append(
        "" if expected_answer is None else str(expected_answer)
    )
    output["input"].append(prompt)
    output["target"].append(target)
    output["filter_reason"].append(reason)


def prepare_openmath_batch(batch, indices):
    output = _prepared_columns()
    for source_index, problem, solution, answer, subtype in zip(
        indices,
        batch["problem"],
        batch["generated_solution"],
        batch["expected_answer"],
        batch["problem_source"],
    ):
        if subtype not in OPENMATH_SUBTYPES:
            continue
        _append_prepared(
            output,
            "openmath",
            source_index,
            subtype,
            problem,
            solution,
            answer,
        )
    return output


def prepare_math500_batch(batch, indices):
    output = _prepared_columns()
    for source_index, problem, solution, answer, subject in zip(
        indices,
        batch["problem"],
        batch["solution"],
        batch["answer"],
        batch["subject"],
    ):
        _append_prepared(
            output,
            "math500",
            source_index,
            subject,
            problem,
            solution,
            answer,
        )
    return output


def tokenize_math_examples(prompts, targets, tokenizer):
    if len(prompts) != len(targets):
        raise ValueError("prompts and targets must have the same length")
    response_texts = [" " + target for target in targets]
    tokenizer_kwargs = {
        "add_special_tokens": False,
        "return_attention_mask": False,
        "return_token_type_ids": False,
    }
    condition_rows = tokenizer(prompts, **tokenizer_kwargs)["input_ids"]
    response_rows = tokenizer(response_texts, **tokenizer_kwargs)["input_ids"]
    joint_rows = tokenizer(
        [prompt + response for prompt, response in zip(prompts, response_texts)],
        **tokenizer_kwargs,
    )["input_ids"]
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")

    tokenized = []
    for condition_ids, response_ids, joint_ids in zip(
        condition_rows, response_rows, joint_rows,
    ):
        if condition_ids + response_ids != joint_ids:
            raise ValueError(
                "separate prompt/response tokenization does not match joint tokenization"
            )
        tokenized.append((
            list(condition_ids),
            list(response_ids) + [int(tokenizer.eos_token_id)],
        ))
    return tokenized


def tokenize_math_example(prompt, target, tokenizer):
    return tokenize_math_examples([prompt], [target], tokenizer)[0]


def training_length_filter_reason(
    condition_ids, response_ids, max_input_length=MAX_INPUT_LENGTH,
    max_response_length=MAX_RESPONSE_LENGTH, max_length=MAX_LENGTH,
):
    if len(condition_ids) > max_input_length:
        return "input_too_long"
    if max_response_length is not None and len(response_ids) > max_response_length:
        return "response_too_long"
    if len(condition_ids) + len(response_ids) > max_length:
        return "sequence_too_long"
    return ""


def math500_length_filter_reason(condition_ids, max_input_length=MAX_INPUT_LENGTH,):
    return "input_too_long" if len(condition_ids) > max_input_length else ""


def tokenize_candidate_batch(
    batch, tokenizer, prompt_only=False, max_input_length=MAX_INPUT_LENGTH,
    max_response_length=MAX_RESPONSE_LENGTH, max_length=MAX_LENGTH, truncate_prompts=False,
):
    tokenized = tokenize_math_examples(batch["input"], batch["target"], tokenizer)
    output = {
        "condition_input_ids": [],
        "input_ids": [],
        "filter_reason": [],
        "prompt_truncated": [],
    }
    for condition_ids, response_ids in tokenized:
        prompt_truncated = prompt_only and truncate_prompts and len(condition_ids) > max_input_length
        if prompt_truncated:
            condition_ids = condition_ids[:max_input_length]
        output["condition_input_ids"].append(condition_ids)
        output["input_ids"].append(response_ids)
        reason = (
            math500_length_filter_reason(condition_ids, max_input_length) if prompt_only else
            training_length_filter_reason(condition_ids, response_ids, max_input_length, max_response_length, max_length,)
        )
        output["filter_reason"].append(reason)
        output["prompt_truncated"].append(prompt_truncated)
    return output


def _word_ngrams(words, size):
    if len(words) < size:
        return set()
    return {
        tuple(words[index:index + size])
        for index in range(len(words) - size + 1)
    }


def _jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def question_similarity(
    left,
    right,
    substring_threshold=SPLIT_SUBSTRING_THRESHOLD,
    word_5gram_threshold=WORD_5GRAM_THRESHOLD,
    word_8gram_threshold=WORD_8GRAM_THRESHOLD,
):
    """Return verified similarity evidence on operator-preserving keys."""
    left_words = left.split()
    right_words = right.split()
    substring_coverage = 0.0
    if left == right:
        reason = "exact"
        substring_coverage = 1.0
    else:
        shorter, longer = sorted((left, right), key=len)
        if shorter in longer:
            substring_coverage = len(shorter) / len(longer)
        reason = "substring" if substring_coverage >= substring_threshold else ""

    five_jaccard = _jaccard(
        _word_ngrams(left_words, 5), _word_ngrams(right_words, 5),
    )
    eight_jaccard = _jaccard(
        _word_ngrams(left_words, 8), _word_ngrams(right_words, 8),
    )
    if not reason and five_jaccard >= word_5gram_threshold:
        reason = "word_5gram_jaccard"
    if not reason and eight_jaccard >= word_8gram_threshold:
        reason = "word_8gram_jaccard"
    if not reason:
        return None
    return {
        "reason": reason,
        "substring_coverage": substring_coverage,
        "word_5gram_jaccard": five_jaccard,
        "word_8gram_jaccard": eight_jaccard,
    }


def _character_ngrams(text, size=4):
    grams = (
        [text] if len(text) < size else
        [text[index:index + size] for index in range(len(text) - size + 1)]
    )
    counts = Counter(grams)
    return {
        f"{gram}\x1e{occurrence}"
        for gram, count in counts.items()
        for occurrence in range(count)
    }


def _minhash(values, num_perm=MINHASH_NUM_PERM, seed=MINHASH_SEED):
    signature = MinHash(num_perm=num_perm, seed=seed)
    signature.update_batch([
        value.encode("utf-8")
        if isinstance(value, str) else
        "\x1f".join(value).encode("utf-8")
        for value in sorted(values)
    ])
    return signature


def _question_signatures(text, num_perm=MINHASH_NUM_PERM):
    words = text.split()
    features = {
        "tokens": set(words),
        "characters": _character_ngrams(text),
        "word_5grams": _word_ngrams(words, 5),
        "word_8grams": _word_ngrams(words, 8),
    }
    return {
        name: _minhash(values, num_perm=num_perm)
        for name, values in features.items() if values
    }


def _lsh_indexes(num_perm=MINHASH_NUM_PERM):
    return {
        name: MinHashLSH(threshold=threshold, num_perm=num_perm)
        for name, threshold in LSH_PROPOSAL_THRESHOLDS.items()
    }


def find_similar_question_pairs(
    questions,
    substring_threshold=SPLIT_SUBSTRING_THRESHOLD,
    num_perm=MINHASH_NUM_PERM,
):
    questions = sorted(set(questions))
    pairs = []
    if len(questions) * (len(questions) - 1) // 2 <= EXHAUSTIVE_PAIR_LIMIT:
        candidates = combinations(range(len(questions)), 2)
        for left_index, right_index in candidates:
            evidence = question_similarity(
                questions[left_index], questions[right_index], substring_threshold,
            )
            if evidence:
                pairs.append({
                    "left": questions[left_index],
                    "right": questions[right_index],
                    **evidence,
                })
        return pairs

    indexes = _lsh_indexes(num_perm)
    for right_index, right in enumerate(questions):
        signatures = _question_signatures(right, num_perm)
        proposed = set()
        for name, signature in signatures.items():
            proposed.update(int(key) for key in indexes[name].query(signature))
        for left_index in sorted(proposed):
            evidence = question_similarity(
                questions[left_index], right, substring_threshold,
            )
            if evidence:
                pairs.append({
                    "left": questions[left_index],
                    "right": right,
                    **evidence,
                })
        key = str(right_index)
        for name, signature in signatures.items():
            indexes[name].insert(key, signature)
    return pairs


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def cluster_question_texts(questions):
    questions = sorted(set(questions))
    union_find = UnionFind(questions)
    similar_pairs = find_similar_question_pairs(questions)
    for pair in similar_pairs:
        union_find.union(pair["left"], pair["right"])

    members_by_root = defaultdict(list)
    for question in questions:
        members_by_root[union_find.find(question)].append(question)
    groups = sorted(members_by_root.values(), key=min)
    question_to_group = {}
    for index, members in enumerate(groups):
        group_id = f"problem_{index:08d}"
        for question in members:
            question_to_group[question] = group_id
    return question_to_group, similar_pairs


def find_contamination_matches(
    candidate_questions,
    reference_questions,
    substring_threshold=TEST_SUBSTRING_THRESHOLD,
    num_perm=MINHASH_NUM_PERM,
):
    candidates = sorted(set(candidate_questions))
    references = sorted(set(reference_questions))
    matches = []
    if len(candidates) * len(references) <= EXHAUSTIVE_PAIR_LIMIT:
        proposed_pairs = (
            (candidate, reference)
            for candidate in candidates for reference in references
        )
        for candidate, reference in proposed_pairs:
            evidence = question_similarity(
                candidate, reference, substring_threshold,
            )
            if evidence:
                matches.append({
                    "candidate": candidate,
                    "reference": reference,
                    **evidence,
                })
        return matches

    indexes = _lsh_indexes(num_perm)
    exact_references = defaultdict(list)
    for reference_index, reference in enumerate(references):
        exact_references[reference].append(reference_index)
        key = str(reference_index)
        for name, signature in _question_signatures(reference, num_perm).items():
            indexes[name].insert(key, signature)

    for candidate in candidates:
        proposed = set(exact_references.get(candidate, ()))
        for name, signature in _question_signatures(candidate, num_perm).items():
            proposed.update(int(key) for key in indexes[name].query(signature))
        for reference_index in sorted(proposed):
            reference = references[reference_index]
            evidence = question_similarity(
                candidate, reference, substring_threshold,
            )
            if evidence:
                matches.append({
                    "candidate": candidate,
                    "reference": reference,
                    **evidence,
                })
    return matches


def find_conflicting_answer_groups(dataset):
    records = {}
    answer_sets = {}
    columns = dataset.select_columns(["question_key", "gold_answer"])
    for batch in columns.iter(batch_size=10_000):
        for question, answer in zip(batch["question_key"], batch["gold_answer"]):
            record = records.get(question)
            if record is None:
                records[question] = [answer, 1]
                continue
            record[1] += 1
            if answer != record[0]:
                answer_sets.setdefault(question, {record[0]}).add(answer)
    return [
        {
            "question_key": question,
            "answers": sorted(answers),
            "rows": records[question][1],
        }
        for question, answers in sorted(answer_sets.items())
    ]


def select_validation_groups(
    problem_group_ids,
    fraction=VALIDATION_FRACTION,
    seed=SPLIT_SEED,
):
    groups = sorted(set(problem_group_ids))
    count = int(len(groups) * fraction + 0.5)
    if not groups or count <= 0 or count >= len(groups):
        raise ValueError("training and validation must both contain problem groups")
    return set(random.Random(seed).sample(groups, count))


def attach_problem_groups(dataset, question_to_group):
    return dataset.map(
        lambda batch: {
            "problem_group_id": [
                question_to_group[question] for question in batch["question_key"]
            ]
        },
        batched=True,
        batch_size=10_000,
        desc="Attaching problem groups",
    )


def split_by_problem_group(dataset, validation_groups):
    train = dataset.filter(
        lambda group_id: group_id not in validation_groups,
        input_columns=["problem_group_id"],
        desc="Selecting training groups",
    )
    validation = dataset.filter(
        lambda group_id: group_id in validation_groups,
        input_columns=["problem_group_id"],
        desc="Selecting validation groups",
    )
    if not len(train) or not len(validation):
        raise ValueError("training and validation splits must both be non-empty")
    if set(train.unique("problem_group_id")) & set(
        validation.unique("problem_group_id")
    ):
        raise AssertionError("a problem group appears in both splits")
    return train, validation


def select_one_row_per_group(dataset):
    seen = set()
    indices = []
    for index, group_id in enumerate(dataset["problem_group_id"]):
        if group_id not in seen:
            seen.add(group_id)
            indices.append(index)
    return dataset.select(indices)


def build_validation_views(
    validation_full,
    subset_size=VALIDATION_SUBSET_SIZE,
    seed=VALIDATION_SUBSET_SEED,
):
    validation_unique = select_one_row_per_group(validation_full)
    if len(validation_unique) < subset_size:
        raise ValueError(
            f"validation has {len(validation_unique)} problem groups, fewer than "
            f"the requested subset of {subset_size}"
        )
    indices = sorted(
        random.Random(seed).sample(range(len(validation_unique)), subset_size)
    )
    return validation_unique, validation_unique.select(indices)


def _add_contiguous_indices(batch, indices):
    del batch
    return {"index": [int(index) for index in indices]}


def finalize_dataset(dataset):
    required = set(FINAL_FEATURES) - {"index"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"dataset is missing final columns: {sorted(missing)}")
    for batch in dataset.select_columns(["target", "gold_answer"]).iter(
        batch_size=10_000,
    ):
        for target, answer in zip(batch["target"], batch["gold_answer"]):
            if not target.endswith(authoritative_answer_suffix(answer)):
                raise ValueError("target is missing its authoritative answer suffix")
    if "index" in dataset.column_names:
        dataset = dataset.remove_columns("index")
    indexed_features = Features(dict(dataset.features))
    indexed_features["index"] = Value("int32")
    dataset = dataset.map(
        _add_contiguous_indices,
        batched=True,
        with_indices=True,
        batch_size=10_000,
        features=indexed_features,
        desc="Assigning contiguous row indices",
    )
    return dataset.select_columns(list(FINAL_FEATURES)).cast(FINAL_FEATURES)


def _map_source(raw, prepare_batch, num_proc, description):
    return raw.map(
        prepare_batch,
        batched=True,
        batch_size=TOKENIZE_BATCH_SIZE,
        num_proc=None if num_proc == 1 else num_proc,
        remove_columns=raw.column_names,
        with_indices=True,
        features=PREPARED_FEATURES,
        desc=description,
    )


def _tokenize_dataset(
    dataset, tokenizer, num_proc, description, prompt_only=False,
    max_input_length=MAX_INPUT_LENGTH, max_response_length=MAX_RESPONSE_LENGTH,
    max_length=MAX_LENGTH, truncate_prompts=False,
):
    features = Features(dict(dataset.features))
    features.update({
        "condition_input_ids": Sequence(Value("int32")),
        "input_ids": Sequence(Value("int32")),
        "filter_reason": Value("string"),
        "prompt_truncated": Value("bool"),
    })
    return dataset.map(
        lambda batch: tokenize_candidate_batch(batch, tokenizer, prompt_only, max_input_length, max_response_length, max_length, truncate_prompts,),
        batched=True,
        batch_size=TOKENIZE_BATCH_SIZE,
        num_proc=None if num_proc == 1 else num_proc,
        features=features,
        desc=description,
    )


def _accepted_rows(dataset, num_proc, description):
    return dataset.filter(
        lambda reason: reason == "",
        input_columns=["filter_reason"],
        num_proc=None if num_proc == 1 else num_proc,
        desc=description,
    ).remove_columns("filter_reason")


def _rejected_rows(dataset, num_proc, description):
    return dataset.filter(
        lambda reason: reason != "",
        input_columns=["filter_reason"],
        num_proc=None if num_proc == 1 else num_proc,
        desc=description,
    )


def _set_filter_reason(dataset, reason):
    return dataset.map(
        lambda batch: {"filter_reason": [reason] * len(batch["source"])},
        batched=True,
        batch_size=10_000,
        desc=f"Marking {reason} rows",
    )


def _filter_problem_groups(dataset, excluded_groups, keep, description):
    return dataset.filter(
        lambda group_id: (group_id in excluded_groups) == keep,
        input_columns=["problem_group_id"],
        desc=description,
    )


def enrich_contamination_matches(matches, question_to_group, grouped_dataset):
    affected_groups = {
        question_to_group[match["candidate"]] for match in matches
    }
    row_counts = Counter()
    if affected_groups:
        for batch in grouped_dataset.select_columns(
            ["problem_group_id"]
        ).iter(batch_size=10_000):
            for group_id in batch["problem_group_id"]:
                if group_id in affected_groups:
                    row_counts[group_id] += 1

    enriched = []
    for match in matches:
        group_id = question_to_group[match["candidate"]]
        enriched.append({
            **match,
            "benchmark_question": match["reference"],
            "problem_group_id": group_id,
            "affected_row_count": row_counts[group_id],
        })
    return enriched


def _reason_counts(dataset):
    return dict(sorted(Counter(dataset["filter_reason"]).items()))


def _value_counts(dataset, column):
    return dict(sorted(Counter(dataset[column]).items()))


def token_length_statistics(dataset):
    count = 0
    prompt_sum = 0
    response_sum = 0
    prompt_min = None
    response_min = None
    prompt_max = 0
    response_max = 0
    for batch in dataset.select_columns(
        ["condition_input_ids", "input_ids"]
    ).iter(batch_size=10_000):
        for condition_ids, response_ids in zip(
            batch["condition_input_ids"], batch["input_ids"],
        ):
            prompt_length = len(condition_ids)
            response_length = len(response_ids)
            count += 1
            prompt_sum += prompt_length
            response_sum += response_length
            prompt_min = (
                prompt_length if prompt_min is None else
                min(prompt_min, prompt_length)
            )
            response_min = (
                response_length if response_min is None else
                min(response_min, response_length)
            )
            prompt_max = max(prompt_max, prompt_length)
            response_max = max(response_max, response_length)

    return {
        "prompt_tokens": {
            "min": prompt_min,
            "max": prompt_max if count else None,
            "mean": prompt_sum / count if count else None,
        },
        "response_tokens_including_eos": {
            "min": response_min,
            "max": response_max if count else None,
            "mean": response_sum / count if count else None,
        },
    }


def _iter_rejection_records(datasets_with_stage):
    fields = (
        "source",
        "source_index",
        "source_subtype",
        "question_key",
        "gold_answer",
        "raw_answer",
        "input",
        "filter_reason",
    )
    for dataset, stage in datasets_with_stage:
        for batch in dataset.select_columns(fields).iter(batch_size=1_000):
            for values in zip(*(batch[field] for field in fields)):
                record = dict(zip(fields, values))
                record["stage"] = stage
                yield record


def write_json(path, value):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_jsonl(path, rows):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _check_columns(dataset, required, source):
    missing = set(required) - set(dataset.column_names)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def _dataset_fingerprint(dataset):
    return getattr(dataset, "_fingerprint", None)


def _optional_positive_int(value):
    if value.lower() == "none":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer_name", default=TOKENIZER_NAME)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--openmath_split",
        choices=tuple(OPENMATH_SPLIT_COUNTS),
        default="train",
    )
    parser.add_argument("--num_proc", type=int, default=DEFAULT_NUM_PROC)
    parser.add_argument("--max_input_length", type=int, default=MAX_INPUT_LENGTH, help="maximum prompt tokens",)
    parser.add_argument("--max_response_length", type=_optional_positive_int, default=MAX_RESPONSE_LENGTH, help="maximum response tokens including EOS, or 'none'",)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="maximum combined prompt and response tokens",)
    parser.add_argument("--truncate_math500_prompts", action="store_true", help="right-truncate over-length MATH-500 prompts instead of rejecting",)
    args = parser.parse_args()
    if args.num_proc <= 0:
        parser.error("--num_proc must be positive")
    if args.max_input_length <= 0:
        parser.error("--max_input_length must be positive")
    if args.max_length <= 0:
        parser.error("--max_length must be positive")
    if args.max_input_length > args.max_length:
        parser.error("--max_input_length cannot exceed --max_length")
    return args


def main():
    args = parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, revision=TOKENIZER_REVISION if args.tokenizer_name == TOKENIZER_NAME else None, cache_dir=args.cache_dir,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")

    openmath_raw = load_dataset(
        OPENMATH_SOURCE,
        split=args.openmath_split,
        revision=OPENMATH_REVISION,
        cache_dir=args.cache_dir,
    )
    math500_raw = load_dataset(
        MATH500_SOURCE,
        split="test",
        revision=MATH500_REVISION,
        cache_dir=args.cache_dir,
    )
    _check_columns(
        openmath_raw,
        {"problem", "generated_solution", "expected_answer", "problem_source"},
        OPENMATH_SOURCE,
    )
    _check_columns(
        math500_raw,
        {"problem", "solution", "answer", "subject"},
        MATH500_SOURCE,
    )
    expected_openmath = OPENMATH_SPLIT_COUNTS[args.openmath_split]
    if len(openmath_raw) != expected_openmath["rows"]:
        raise ValueError(
            f"expected {expected_openmath['rows']} OpenMath rows, "
            f"found {len(openmath_raw)}"
        )
    if len(math500_raw) != EXPECTED_MATH500_ROWS:
        raise ValueError(
            f"expected {EXPECTED_MATH500_ROWS} MATH-500 rows, found {len(math500_raw)}"
        )

    openmath_prepared = _map_source(
        openmath_raw,
        prepare_openmath_batch,
        args.num_proc,
        "Preparing OpenMath MATH rows",
    )
    if len(openmath_prepared) != expected_openmath["math_rows"]:
        raise ValueError(
            f"expected {expected_openmath['math_rows']} OpenMath MATH rows, "
            f"found {len(openmath_prepared)}"
        )
    math500_prepared = _map_source(
        math500_raw,
        prepare_math500_batch,
        args.num_proc,
        "Preparing MATH-500",
    )

    structural_rejected = _rejected_rows(
        openmath_prepared,
        args.num_proc,
        "Collecting structurally invalid OpenMath rows",
    )
    structurally_valid = _accepted_rows(
        openmath_prepared,
        args.num_proc,
        "Keeping structurally valid OpenMath rows",
    )

    conflict_groups = find_conflicting_answer_groups(structurally_valid)
    conflicting_keys = {group["question_key"] for group in conflict_groups}
    conflict_rejected = structurally_valid.filter(
        lambda question: question in conflicting_keys,
        input_columns=["question_key"],
        desc="Collecting conflicting-answer rows",
    )
    conflict_rejected = _set_filter_reason(
        conflict_rejected.add_column(
            "filter_reason", [""] * len(conflict_rejected)
        ),
        "conflicting_answer_group",
    )
    answer_valid = structurally_valid.filter(
        lambda question: question not in conflicting_keys,
        input_columns=["question_key"],
        desc="Removing conflicting-answer groups",
    )

    tokenized = _tokenize_dataset(
        answer_valid,
        tokenizer,
        args.num_proc,
        "Tokenizing OpenMath MATH candidates",
        max_input_length=args.max_input_length,
        max_response_length=args.max_response_length,
        max_length=args.max_length,
    )
    length_rejected = _rejected_rows(
        tokenized,
        args.num_proc,
        "Collecting over-length OpenMath rows",
    )
    length_valid = _accepted_rows(
        tokenized,
        args.num_proc,
        "Keeping OpenMath rows within token limits",
    )

    math500_rejected = _rejected_rows(
        math500_prepared,
        args.num_proc,
        "Checking MATH-500 answer structure",
    )
    if len(math500_rejected):
        raise ValueError(
            f"MATH-500 has invalid answers: {_reason_counts(math500_rejected)}"
        )
    math500 = _accepted_rows(
        math500_prepared,
        args.num_proc,
        "Keeping structurally valid MATH-500 rows",
    )
    math500_tokenized = _tokenize_dataset(
        math500,
        tokenizer,
        args.num_proc,
        "Tokenizing MATH-500",
        prompt_only=True,
        max_input_length=args.max_input_length,
        max_response_length=args.max_response_length,
        max_length=args.max_length,
        truncate_prompts=args.truncate_math500_prompts,
    )
    bad_math500_prompts = _rejected_rows(
        math500_tokenized,
        args.num_proc,
        "Checking MATH-500 prompt lengths",
    )
    if len(bad_math500_prompts):
        raise ValueError(
            f"MATH-500 has over-length prompts: "
            f"{_reason_counts(bad_math500_prompts)}"
        )
    math500_tokenized = _accepted_rows(
        math500_tokenized,
        args.num_proc,
        "Keeping all MATH-500 prompts",
    )
    if len(math500_tokenized) != EXPECTED_MATH500_ROWS:
        raise AssertionError("all 500 MATH-500 prompts must be preserved")
    math500_truncated_prompts = sum(math500_tokenized["prompt_truncated"])

    training_questions = set(length_valid.unique("question_key"))
    question_to_group, similar_pairs = cluster_question_texts(training_questions)
    for pair in similar_pairs:
        pair["problem_group_id"] = question_to_group[pair["left"]]

    contamination_matches = find_contamination_matches(
        training_questions,
        set(math500_tokenized.unique("question_key")),
    )
    grouped = attach_problem_groups(length_valid, question_to_group)
    contamination_matches = enrich_contamination_matches(
        contamination_matches, question_to_group, grouped,
    )
    contaminated_groups = {
        match["problem_group_id"] for match in contamination_matches
    }
    contaminated = _filter_problem_groups(
        grouped,
        contaminated_groups,
        True,
        "Auditing MATH-500-contaminated rows",
    )
    accepted = _filter_problem_groups(
        grouped,
        contaminated_groups,
        False,
        "Removing MATH-500-contaminated rows",
    )

    validation_groups = select_validation_groups(
        accepted.unique("problem_group_id"),
    )
    train_internal, validation_internal = split_by_problem_group(
        accepted, validation_groups,
    )
    validation_unique_internal, validation_subset_internal = build_validation_views(
        validation_internal,
    )

    math500_tokenized = math500_tokenized.add_column(
        "problem_group_id",
        [f"math500_test_{index:04d}" for index in range(len(math500_tokenized))],
    )
    train = finalize_dataset(train_internal)
    validation_full = finalize_dataset(validation_internal)
    validation_unique = finalize_dataset(validation_unique_internal)
    validation_1024 = finalize_dataset(validation_subset_internal)
    math500_test = finalize_dataset(math500_tokenized)

    outputs = {
        "train": train,
        "validation_full": validation_full,
        "validation_unique": validation_unique,
        "validation_1024": validation_1024,
        "math500_test": math500_test,
    }
    internal_outputs = {
        "train": train_internal,
        "validation_full": validation_internal,
        "validation_unique": validation_unique_internal,
        "validation_1024": validation_subset_internal,
        "math500_test": math500_tokenized,
    }
    os.makedirs(args.output_dir)
    for name, dataset in outputs.items():
        dataset.save_to_disk(os.path.join(args.output_dir, name))

    write_jsonl(
        os.path.join(args.output_dir, "rejected_rows.jsonl"),
        _iter_rejection_records((
            (structural_rejected, "structure"),
            (conflict_rejected, "answer_conflict"),
            (length_rejected, "length"),
        )),
    )
    write_jsonl(
        os.path.join(args.output_dir, "conflicting_answer_groups.jsonl"),
        conflict_groups,
    )
    write_jsonl(
        os.path.join(args.output_dir, "math500_contamination_matches.jsonl"),
        contamination_matches,
    )
    write_jsonl(
        os.path.join(args.output_dir, "verified_similar_question_pairs.jsonl"),
        similar_pairs,
    )

    metadata = {
        "version": 1,
        "sources": {
            "openmath": {
                "dataset": OPENMATH_SOURCE,
                "revision": OPENMATH_REVISION,
                "split": args.openmath_split,
                "fingerprint": _dataset_fingerprint(openmath_raw),
                "raw_rows": len(openmath_raw),
                "included_problem_sources": sorted(OPENMATH_SUBTYPES),
                "eligible_rows": len(openmath_prepared),
                "structurally_rejected_rows": len(structural_rejected),
                "structural_rejection_counts": _reason_counts(structural_rejected),
                "conflicting_answer_groups": len(conflict_groups),
                "conflicting_answer_rows": len(conflict_rejected),
                "length_rejected_rows": len(length_rejected),
                "length_rejection_counts": _reason_counts(length_rejected),
                "contaminated_rows": len(contaminated),
                "saved_rows": len(train) + len(validation_full),
            },
            "math500": {
                "dataset": MATH500_SOURCE,
                "revision": MATH500_REVISION,
                "split": "test",
                "fingerprint": _dataset_fingerprint(math500_raw),
                "rows": len(math500_test),
                "subject_counts": _value_counts(math500_test, "source_subtype"),
            },
        },
        "prompt_format": "Question: {problem}\\nStep-by-Step Answer:",
        "target_format": (
            "{solution.strip()}\\nFinal answer: "
            "$\\\\boxed{{{authoritative_answer.strip()}}}$"
        ),
        "answer_policy": {
            "authoritative_fields": {
                "openmath": "expected_answer",
                "math500": "answer",
            },
            "outer_whitespace_only": True,
            "existing_solution_boxes_preserved": True,
            "rejected_unescaped_specials": ["$", "%", "#"],
            "allowed_ampersand_environments": sorted(ALIGNMENT_ENVIRONMENTS),
            "conflicting_exact_question_groups_rejected": True,
        },
        "question_key": {
            "unicode_normalization": "NFC",
            "outer_whitespace_stripped": True,
            "internal_whitespace_collapsed": True,
            "case_preserved": True,
            "punctuation_and_operators_preserved": True,
        },
        "tokenizer": args.tokenizer_name,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_size_with_added_tokens": len(tokenizer),
        "eos_token": tokenizer.eos_token,
        "stored_eos_token_id": int(tokenizer.eos_token_id),
        "pad_token": tokenizer.pad_token,
        "stored_pad_token_id": int(tokenizer.pad_token_id),
        "token_column_dtype": "int32",
        "index_column_dtype": "int32",
        "training_length_limits": {
            "max_input_length": args.max_input_length,
            "max_response_length_including_eos": args.max_response_length,
            "max_combined_length": args.max_length,
        },
        "math500_length_policy": {
            "max_input_length": args.max_input_length,
            "response_or_combined_limit": None,
            "over_length_action": (
                "right_truncate" if args.truncate_math500_prompts else "reject"
            ),
            "truncated_rows": math500_truncated_prompts,
            "all_rows_required": EXPECTED_MATH500_ROWS,
        },
        "token_lengths_after_training_length_filter": token_length_statistics(
            length_valid
        ),
        "matching": {
            "test_substring_threshold": TEST_SUBSTRING_THRESHOLD,
            "split_substring_threshold": SPLIT_SUBSTRING_THRESHOLD,
            "word_5gram_jaccard_threshold": WORD_5GRAM_THRESHOLD,
            "word_8gram_jaccard_threshold": WORD_8GRAM_THRESHOLD,
            "minhash_num_perm": MINHASH_NUM_PERM,
            "minhash_seed": MINHASH_SEED,
            "lsh_proposal_thresholds": LSH_PROPOSAL_THRESHOLDS,
            "verified_similar_pairs": len(similar_pairs),
            "math500_contamination_matches": len(contamination_matches),
            "contaminated_problem_groups": len(contaminated_groups),
            "problem_groups_before_decontamination": len(
                set(question_to_group.values())
            ),
        },
        "split": {
            "unit": "problem_group",
            "validation_fraction": VALIDATION_FRACTION,
            "split_seed": SPLIT_SEED,
            "validation_groups": len(validation_groups),
            "validation_subset_seed": VALIDATION_SUBSET_SEED,
            "validation_subset_size": VALIDATION_SUBSET_SIZE,
        },
        "fingerprints": {
            "openmath_prepared": _dataset_fingerprint(openmath_prepared),
            "answer_valid": _dataset_fingerprint(answer_valid),
            "length_valid": _dataset_fingerprint(length_valid),
            "decontaminated": _dataset_fingerprint(accepted),
            "math500_tokenized": _dataset_fingerprint(math500_tokenized),
        },
        "outputs": {
            name: {
                "rows": len(dataset),
                "problem_groups": len(
                    internal_outputs[name].unique("problem_group_id")
                ),
                "fingerprint": _dataset_fingerprint(dataset),
            }
            for name, dataset in outputs.items()
        },
    }
    metadata["tokenizer_revision"] = TOKENIZER_REVISION if args.tokenizer_name == TOKENIZER_NAME else None
    write_json(
        os.path.join(args.output_dir, "preprocessing_metadata.json"), metadata,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
