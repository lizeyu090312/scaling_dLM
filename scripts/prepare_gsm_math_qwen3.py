#!/usr/bin/env python
"""Build a leakage-resistant GSM-style math dataset for Qwen3 ELF training."""

import argparse
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations

from datasets import (
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
)
from datasketch import MinHash, MinHashLSH
from transformers import AutoTokenizer


OPENMATH_SOURCE = "nvidia/OpenMathInstruct-2"
METAMATH_SOURCE = "meta-math/MetaMathQA"
GSM8K_SOURCE = "openai/gsm8k"
GSM8K_CONFIG = "main"
# Snapshot revisions used to prepare the released data.
OPENMATH_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"
METAMATH_REVISION = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
TOKENIZER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"
OUTPUT_DIR = "data/gsm-math-qwen3-embedding-0.6b-v1"

OPENMATH_SUBTYPES = frozenset({"gsm8k", "augmented_gsm8k"})
EXPECTED_GSM8K_TEST_ROWS = 1_319
MAX_INPUT_LENGTH = 256
MAX_LENGTH = 1_024
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

# LSH only proposes pairs. The exact thresholds above make the final decision.
LSH_PROPOSAL_THRESHOLDS = {
    "tokens": 0.50,
    "characters": 0.50,
    "word_5grams": 0.40,
    "word_8grams": 0.30,
}

_ANSWER_MARKER = "The answer is:"
_TERMINAL_HASH_ANSWER_RE = re.compile(r"\s*####\s*[^\n]*\s*$")
_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_UNSIGNED_INTEGER = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_UNSIGNED_DECIMAL = rf"(?:{_UNSIGNED_INTEGER}(?:\.\d+)?|\.\d+)"
_DECIMAL_RE = re.compile(rf"[+-]?{_UNSIGNED_DECIMAL}")
_FRACTION_RE = re.compile(
    rf"(?P<sign>[+-]?)(?P<numerator>{_UNSIGNED_INTEGER})\s*/\s*"
    rf"(?P<denominator>{_UNSIGNED_INTEGER})"
)

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
    "source_question_key": Value("string"),
    "gold_answer": Value("string"),
    "input": Value("string"),
    "target": Value("string"),
    "filter_reason": Value("string"),
})


def build_math_prompt(question):
    question = str(question).strip()
    if not question:
        raise ValueError("missing_question")
    return f"Question: {question}\nAnswer:"


def normalize_numeric_answer(value):
    """Return the supported numeric answer without thousands separators."""
    if value is None:
        raise ValueError("missing_answer")
    answer = str(value).strip()
    if not answer:
        raise ValueError("missing_answer")

    fraction = _FRACTION_RE.fullmatch(answer)
    if fraction:
        numerator = fraction.group("numerator").replace(",", "")
        denominator = fraction.group("denominator").replace(",", "")
        if int(denominator) == 0:
            raise ValueError("zero_fraction_denominator")
        return f"{fraction.group('sign')}{numerator}/{denominator}"

    if _DECIMAL_RE.fullmatch(answer):
        answer = answer.replace(",", "")
        if answer.startswith("."):
            answer = "0" + answer
        elif answer.startswith(("+.", "-.")):
            answer = answer[:1] + "0" + answer[1:]
        return answer
    raise ValueError("unsupported_answer")


def _strip_terminal_hash_answer(rationale):
    rationale = _TERMINAL_HASH_ANSWER_RE.sub("", rationale.rstrip()).rstrip()
    if "####" in rationale:
        raise ValueError("answer_marker_in_rationale")
    return rationale


def make_target(rationale, gold_answer):
    if rationale is None:
        raise ValueError("missing_solution")
    rationale = _strip_terminal_hash_answer(str(rationale).strip())
    target = f"{rationale}\n#### {gold_answer}" if rationale else f"#### {gold_answer}"
    if target.count("####") != 1 or not target.endswith(f"#### {gold_answer}"):
        raise ValueError("invalid_target")
    return target


def canonicalize_openmath_response(solution, expected_answer):
    """Use OpenMath's expected answer, never a number inferred from its solution."""
    gold_answer = normalize_numeric_answer(expected_answer)
    return make_target(solution, gold_answer), gold_answer


def canonicalize_metamath_response(response):
    if response is None:
        raise ValueError("missing_solution")
    response = str(response).rstrip()
    marker_start = response.rfind(_ANSWER_MARKER)
    if marker_start < 0:
        raise ValueError("missing_terminal_answer_marker")
    answer_text = response[marker_start + len(_ANSWER_MARKER):].strip()
    if "\n" in answer_text:
        raise ValueError("multiline_answer")
    gold_answer = normalize_numeric_answer(answer_text)
    return make_target(response[:marker_start], gold_answer), gold_answer


def canonicalize_gsm8k_response(response):
    if response is None:
        raise ValueError("missing_solution")
    response = str(response).strip()
    marker_start = response.rfind("####")
    if marker_start < 0:
        raise ValueError("missing_terminal_hash_answer")
    gold_answer = normalize_numeric_answer(response[marker_start + 4:].strip())
    return make_target(response[:marker_start], gold_answer), gold_answer


def normalize_question(question):
    """Normalize case, whitespace, and punctuation for question comparison."""
    if question is None:
        raise ValueError("missing_question")
    text = unicodedata.normalize("NFKC", str(question)).lower()
    words = _WORD_RE.findall(text)
    if not words:
        raise ValueError("missing_question")
    return " ".join(words)


def _prepared_columns():
    return {
        "source": [],
        "source_index": [],
        "source_subtype": [],
        "question_key": [],
        "source_question_key": [],
        "gold_answer": [],
        "input": [],
        "target": [],
        "filter_reason": [],
    }


def _append_prepared(
    output, source, source_index, subtype, question, source_question, response_builder,
):
    question_key = ""
    source_question_key = ""
    prompt = ""
    target = ""
    gold_answer = ""
    reason = ""
    try:
        question_key = normalize_question(question)
        source_question_key = normalize_question(source_question)
        prompt = build_math_prompt(question)
        target, gold_answer = response_builder()
    except ValueError as error:
        reason = str(error)

    output["source"].append(source)
    output["source_index"].append(int(source_index))
    output["source_subtype"].append(str(subtype))
    output["question_key"].append(question_key)
    output["source_question_key"].append(source_question_key)
    output["gold_answer"].append(gold_answer)
    output["input"].append(prompt)
    output["target"].append(target)
    output["filter_reason"].append(reason)


def prepare_openmath_batch(batch, indices):
    output = _prepared_columns()
    for source_index, problem, solution, expected_answer, subtype in zip(
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
            problem,
            lambda solution=solution, expected_answer=expected_answer: (
                canonicalize_openmath_response(solution, expected_answer)
            ),
        )
    return output


def prepare_metamath_batch(batch, indices):
    output = _prepared_columns()
    for source_index, question, response, subtype, original_question in zip(
        indices,
        batch["query"],
        batch["response"],
        batch["type"],
        batch["original_question"],
    ):
        if not str(subtype).startswith("GSM"):
            continue
        source_question = original_question
        try:
            normalize_question(source_question)
        except ValueError:
            source_question = question
        _append_prepared(
            output,
            "metamath",
            source_index,
            subtype,
            question,
            source_question,
            lambda response=response: canonicalize_metamath_response(response),
        )
    return output


def prepare_gsm8k_batch(batch, indices):
    output = _prepared_columns()
    for source_index, question, response in zip(
        indices, batch["question"], batch["answer"],
    ):
        _append_prepared(
            output,
            "gsm8k",
            source_index,
            "test",
            question,
            question,
            lambda response=response: canonicalize_gsm8k_response(response),
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


def length_filter_reason(condition_ids, response_ids):
    if len(condition_ids) > MAX_INPUT_LENGTH:
        return "input_too_long"
    if len(condition_ids) + len(response_ids) > MAX_LENGTH:
        return "sequence_too_long"
    return ""


def tokenize_candidate_batch(batch, tokenizer):
    tokenized = tokenize_math_examples(batch["input"], batch["target"], tokenizer)
    output = {
        "condition_input_ids": [],
        "input_ids": [],
        "filter_reason": [],
    }
    for condition_ids, response_ids in tokenized:
        output["condition_input_ids"].append(condition_ids)
        output["input_ids"].append(response_ids)
        output["filter_reason"].append(
            length_filter_reason(condition_ids, response_ids)
        )
    return output


def _word_ngrams(words, size):
    if len(words) < size:
        return set()
    return {tuple(words[index:index + size]) for index in range(len(words) - size + 1)}


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
    """Return exact similarity evidence when two normalized questions match."""
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
        value.encode("utf-8") if isinstance(value, str) else "\x1f".join(value).encode("utf-8")
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
    """Find verified pairs among unique, normalized question strings."""
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


def cluster_question_texts(questions, linked_pairs=()):
    """Group exact/near duplicates and caller-supplied source relationships."""
    questions = sorted(set(questions))
    union_find = UnionFind(questions)
    for left, right in linked_pairs:
        union_find.union(left, right)

    similar_pairs = find_similar_question_pairs(questions)
    for pair in similar_pairs:
        union_find.union(pair["left"], pair["right"])

    members_by_root = defaultdict(list)
    for question in questions:
        members_by_root[union_find.find(question)].append(question)
    ordered_groups = sorted(members_by_root.values(), key=lambda members: min(members))
    question_to_group = {}
    for index, members in enumerate(ordered_groups):
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
    """Find candidate questions matching the official test after exact verification."""
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


def select_validation_groups(
    source_question_keys,
    question_to_group,
    fraction=VALIDATION_FRACTION,
    seed=SPLIT_SEED,
):
    """Select source questions, then expand them to their complete global groups."""
    rng = random.Random(seed)
    selected_by_source = {}
    validation_groups = set()
    for source in sorted(source_question_keys):
        questions = sorted(source_question_keys[source])
        count = int(len(questions) * fraction + 0.5)
        selected = sorted(rng.sample(questions, count)) if count else []
        selected_by_source[source] = selected
        validation_groups.update(question_to_group[question] for question in selected)
    return validation_groups, selected_by_source


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
        raise AssertionError("a problem group appears in both training and validation")
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
    rng = random.Random(seed)
    subset_indices = sorted(rng.sample(range(len(validation_unique)), subset_size))
    return validation_unique, validation_unique.select(subset_indices)


def _add_contiguous_indices(batch, indices):
    del batch
    return {"index": [int(index) for index in indices]}


def finalize_dataset(dataset):
    """Project to the public schema and force token IDs to Arrow int32."""
    required = set(FINAL_FEATURES) - {"index"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"dataset is missing final columns: {sorted(missing)}")
    for batch in dataset.select_columns(["target"]).iter(batch_size=10_000):
        for target in batch["target"]:
            if target.count("####") != 1 or not re.search(
                r"(?:^|\n)####\s*\S+\s*$", target,
            ):
                raise ValueError(
                    "every accepted target must have one terminal #### answer"
                )
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
    dataset = dataset.select_columns(list(FINAL_FEATURES))
    return dataset.cast(FINAL_FEATURES)


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


def _tokenize_dataset(dataset, tokenizer, num_proc, description):
    features = Features(dict(dataset.features))
    features.update({
        "condition_input_ids": Sequence(Value("int32")),
        "input_ids": Sequence(Value("int32")),
        "filter_reason": Value("string"),
    })
    return dataset.map(
        lambda batch: tokenize_candidate_batch(batch, tokenizer),
        batched=True,
        batch_size=TOKENIZE_BATCH_SIZE,
        num_proc=None if num_proc == 1 else num_proc,
        features=features,
        desc=description,
    )


def _accepted_rows(prepared, num_proc, description):
    return prepared.filter(
        lambda reason: reason == "",
        input_columns=["filter_reason"],
        num_proc=None if num_proc == 1 else num_proc,
        desc=description,
    ).remove_columns("filter_reason")


def _rejected_rows(prepared, num_proc, description):
    return prepared.filter(
        lambda reason: reason != "",
        input_columns=["filter_reason"],
        num_proc=None if num_proc == 1 else num_proc,
        desc=description,
    )


def _filter_problem_groups(dataset, excluded_groups, keep, description):
    return dataset.filter(
        lambda group_id: (group_id in excluded_groups) == keep,
        input_columns=["problem_group_id"],
        desc=description,
    )


def _collect_source_question_keys(dataset):
    keys = defaultdict(set)
    columns = dataset.select_columns(["source", "source_question_key"])
    for batch in columns.iter(batch_size=10_000):
        for source, question in zip(batch["source"], batch["source_question_key"]):
            keys[source].add(question)
    return dict(keys)


def _value_counts(dataset, column):
    return dict(sorted(Counter(dataset[column]).items()))


def source_composition(dataset):
    row_counts = Counter()
    problem_groups = defaultdict(set)
    source_questions = defaultdict(set)
    fields = ["source", "problem_group_id"]
    if "source_question_key" in dataset.column_names:
        fields.append("source_question_key")
    for batch in dataset.select_columns(fields).iter(batch_size=10_000):
        source_question_values = batch.get("source_question_key", [None] * len(batch["source"]))
        for source, group_id, source_question in zip(
            batch["source"], batch["problem_group_id"], source_question_values,
        ):
            row_counts[source] += 1
            problem_groups[source].add(group_id)
            if source_question is not None:
                source_questions[source].add(source_question)

    total_rows = sum(row_counts.values())
    total_problem_counts = sum(len(groups) for groups in problem_groups.values())
    total_source_question_counts = sum(
        len(questions) for questions in source_questions.values()
    )
    return {
        "row_counts": dict(sorted(row_counts.items())),
        "row_fractions": {
            source: count / total_rows for source, count in sorted(row_counts.items())
        },
        "distinct_problem_group_counts": {
            source: len(groups) for source, groups in sorted(problem_groups.items())
        },
        "distinct_problem_group_fractions": {
            source: len(groups) / total_problem_counts
            for source, groups in sorted(problem_groups.items())
        },
        "distinct_source_question_counts": {
            source: len(questions)
            for source, questions in sorted(source_questions.items())
        },
        "distinct_source_question_fractions": {
            source: len(questions) / total_source_question_counts
            for source, questions in sorted(source_questions.items())
        },
    }


def _reason_counts(dataset):
    return dict(sorted(Counter(dataset["filter_reason"]).items()))


def token_length_statistics(dataset):
    count = 0
    prompt_sum = 0
    response_sum = 0
    prompt_min = None
    prompt_max = 0
    response_min = None
    response_max = 0
    responses_over_768 = 0
    columns = dataset.select_columns(["condition_input_ids", "input_ids"])
    for batch in columns.iter(batch_size=10_000):
        for condition_ids, response_ids in zip(
            batch["condition_input_ids"], batch["input_ids"],
        ):
            prompt_length = len(condition_ids)
            response_length = len(response_ids)
            count += 1
            prompt_sum += prompt_length
            response_sum += response_length
            prompt_min = prompt_length if prompt_min is None else min(prompt_min, prompt_length)
            response_min = (
                response_length if response_min is None else min(response_min, response_length)
            )
            prompt_max = max(prompt_max, prompt_length)
            response_max = max(response_max, response_length)
            responses_over_768 += response_length > 768
    return {
        "prompt_tokens_min": prompt_min,
        "prompt_tokens_max": prompt_max,
        "prompt_tokens_mean": prompt_sum / count if count else None,
        "response_tokens_including_eos_min": response_min,
        "response_tokens_including_eos_max": response_max,
        "response_tokens_including_eos_mean": response_sum / count if count else None,
        "responses_over_768_tokens": responses_over_768,
    }


def _iter_rejection_records(datasets_with_stage):
    fields = (
        "source", "source_index", "source_subtype", "input", "filter_reason",
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer_name", default=TOKENIZER_NAME)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--num_proc", type=int, default=DEFAULT_NUM_PROC)
    args = parser.parse_args()
    if args.num_proc <= 0:
        parser.error("--num_proc must be positive")
    return args


def _dataset_fingerprint(dataset):
    return getattr(dataset, "_fingerprint", None)


def _source_summary(raw, prepared, accepted_answers, answer_rejected):
    return {
        "fingerprint": _dataset_fingerprint(raw),
        "raw_rows": len(raw),
        "eligible_source_rows": len(prepared),
        "accepted_answer_rows": len(accepted_answers),
        "answer_rejected_rows": len(answer_rejected),
        "answer_rejection_counts": _reason_counts(answer_rejected),
    }


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
        OPENMATH_SOURCE, split="train", revision=OPENMATH_REVISION, cache_dir=args.cache_dir,
    )
    metamath_raw = load_dataset(
        METAMATH_SOURCE, split="train", revision=METAMATH_REVISION, cache_dir=args.cache_dir,
    )
    gsm8k_raw = load_dataset(
        GSM8K_SOURCE, GSM8K_CONFIG, split="test", revision=GSM8K_REVISION, cache_dir=args.cache_dir,
    )
    _check_columns(
        openmath_raw,
        {"problem", "generated_solution", "expected_answer", "problem_source"},
        OPENMATH_SOURCE,
    )
    _check_columns(
        metamath_raw,
        {"query", "response", "type", "original_question"},
        METAMATH_SOURCE,
    )
    _check_columns(gsm8k_raw, {"question", "answer"}, GSM8K_SOURCE)
    if len(gsm8k_raw) != EXPECTED_GSM8K_TEST_ROWS:
        raise ValueError(
            f"expected {EXPECTED_GSM8K_TEST_ROWS} GSM8K test rows, found {len(gsm8k_raw)}"
        )

    openmath_prepared = _map_source(
        openmath_raw, prepare_openmath_batch, args.num_proc, "Preparing OpenMath",
    )
    metamath_prepared = _map_source(
        metamath_raw, prepare_metamath_batch, args.num_proc, "Preparing MetaMathQA",
    )
    gsm8k_prepared = _map_source(
        gsm8k_raw, prepare_gsm8k_batch, args.num_proc, "Preparing GSM8K test",
    )

    openmath_answer_rejected = _rejected_rows(
        openmath_prepared, args.num_proc, "Collecting rejected OpenMath answers",
    )
    metamath_answer_rejected = _rejected_rows(
        metamath_prepared, args.num_proc, "Collecting rejected MetaMathQA answers",
    )
    openmath = _accepted_rows(
        openmath_prepared, args.num_proc, "Keeping valid OpenMath answers",
    )
    metamath = _accepted_rows(
        metamath_prepared, args.num_proc, "Keeping valid MetaMathQA answers",
    )
    gsm8k_answer_rejected = _rejected_rows(
        gsm8k_prepared, args.num_proc, "Checking GSM8K answers",
    )
    if len(gsm8k_answer_rejected):
        raise ValueError(
            f"official GSM8K test has invalid rows: {_reason_counts(gsm8k_answer_rejected)}"
        )
    gsm8k = _accepted_rows(
        gsm8k_prepared, args.num_proc, "Keeping valid GSM8K answers",
    )

    answer_valid = concatenate_datasets([openmath, metamath])
    tokenized = _tokenize_dataset(
        answer_valid,
        tokenizer,
        args.num_proc,
        "Tokenizing combined training candidates",
    )
    length_rejected = _rejected_rows(
        tokenized, args.num_proc, "Collecting over-length rows",
    )
    length_valid = _accepted_rows(
        tokenized, args.num_proc, "Keeping rows within token limits",
    )

    openmath_length_valid = length_valid.filter(
        lambda source: source == "openmath",
        input_columns=["source"],
        desc="Selecting length-valid OpenMath rows",
    )
    metamath_length_valid = length_valid.filter(
        lambda source: source == "metamath",
        input_columns=["source"],
        desc="Selecting length-valid MetaMathQA rows",
    )
    openmath_questions = set(openmath_length_valid.unique("question_key"))
    metamath_questions = set(metamath_length_valid.unique("question_key"))
    metamath_source_questions = set(
        metamath_length_valid.unique("source_question_key")
    )
    linked_pairs = set(zip(
        metamath_length_valid["question_key"],
        metamath_length_valid["source_question_key"],
    ))
    all_questions = openmath_questions | metamath_questions | metamath_source_questions
    question_to_group, similar_pairs = cluster_question_texts(
        all_questions, linked_pairs,
    )
    for pair in similar_pairs:
        pair["problem_group_id"] = question_to_group[pair["left"]]

    contamination_matches = find_contamination_matches(
        all_questions, set(gsm8k.unique("question_key")),
    )
    contaminated_groups = {
        question_to_group[match["candidate"]] for match in contamination_matches
    }
    for match in contamination_matches:
        match["problem_group_id"] = question_to_group[match["candidate"]]

    grouped = attach_problem_groups(length_valid, question_to_group)
    contaminated = _filter_problem_groups(
        grouped, contaminated_groups, True, "Auditing contaminated rows",
    )
    contaminated_rows_by_source = _value_counts(contaminated, "source")
    accepted = _filter_problem_groups(
        grouped, contaminated_groups, False, "Removing GSM8K-contaminated rows",
    )

    source_question_keys = _collect_source_question_keys(accepted)
    validation_groups, selected_questions = select_validation_groups(
        source_question_keys, question_to_group,
    )
    train_internal, validation_internal = split_by_problem_group(
        accepted, validation_groups,
    )
    validation_unique_internal, validation_subset_internal = build_validation_views(
        validation_internal,
    )

    train = finalize_dataset(train_internal)
    validation_full = finalize_dataset(validation_internal)
    validation_unique = finalize_dataset(validation_unique_internal)
    validation_1024 = finalize_dataset(validation_subset_internal)

    gsm8k_tokenized = _tokenize_dataset(
        gsm8k, tokenizer, args.num_proc, "Tokenizing GSM8K test",
    )
    bad_test_lengths = _rejected_rows(
        gsm8k_tokenized, args.num_proc, "Checking GSM8K token lengths",
    )
    if len(bad_test_lengths):
        raise ValueError(
            f"official GSM8K test has over-length rows: {_reason_counts(bad_test_lengths)}"
        )
    gsm8k_tokenized = _accepted_rows(
        gsm8k_tokenized, args.num_proc, "Keeping GSM8K rows within token limits",
    )
    gsm8k_tokenized = gsm8k_tokenized.add_column(
        "problem_group_id",
        [f"gsm8k_test_{index:04d}" for index in range(len(gsm8k_tokenized))],
    )
    gsm8k_test = finalize_dataset(gsm8k_tokenized)

    os.makedirs(args.output_dir)
    outputs = {
        "train": train,
        "validation_full": validation_full,
        "validation_unique": validation_unique,
        "validation_1024": validation_1024,
        "gsm8k_test": gsm8k_test,
    }
    internal_outputs = {
        "train": train_internal,
        "validation_full": validation_internal,
        "validation_unique": validation_unique_internal,
        "validation_1024": validation_subset_internal,
        "gsm8k_test": gsm8k_tokenized,
    }
    for name, dataset in outputs.items():
        dataset.save_to_disk(os.path.join(args.output_dir, name))

    write_jsonl(
        os.path.join(args.output_dir, "rejected_rows.jsonl"),
        _iter_rejection_records((
            (openmath_answer_rejected, "answer"),
            (metamath_answer_rejected, "answer"),
            (length_rejected, "length"),
        )),
    )
    write_jsonl(
        os.path.join(args.output_dir, "gsm8k_contamination_matches.jsonl"),
        contamination_matches,
    )
    write_jsonl(
        os.path.join(args.output_dir, "verified_similar_question_pairs.jsonl"),
        similar_pairs,
    )

    accepted_source_rows = _value_counts(accepted, "source")
    accepted_source_questions = {
        source: len(questions) for source, questions in source_question_keys.items()
    }
    total_accepted_rows = len(accepted)
    total_source_questions = sum(accepted_source_questions.values())
    metadata = {
        "version": 1,
        "sources": {
            "openmath": {
                "dataset": OPENMATH_SOURCE,
                "included_problem_sources": sorted(OPENMATH_SUBTYPES),
                **_source_summary(
                    openmath_raw, openmath_prepared, openmath, openmath_answer_rejected,
                ),
                "contaminated_rows": contaminated_rows_by_source.get("openmath", 0),
            },
            "metamath": {
                "dataset": METAMATH_SOURCE,
                "included_type_prefix": "GSM",
                **_source_summary(
                    metamath_raw, metamath_prepared, metamath, metamath_answer_rejected,
                ),
                "contaminated_rows": contaminated_rows_by_source.get("metamath", 0),
            },
            "gsm8k_test": {
                "dataset": GSM8K_SOURCE,
                "config": GSM8K_CONFIG,
                "fingerprint": _dataset_fingerprint(gsm8k_raw),
                "rows": len(gsm8k_test),
            },
        },
        "prompt_format": "Question: {question}\\nAnswer:",
        "target_format": "rationale\\n#### {numeric_answer}",
        "tokenizer": args.tokenizer_name,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_size_with_added_tokens": len(tokenizer),
        "eos_token": tokenizer.eos_token,
        "stored_eos_token_id": int(tokenizer.eos_token_id),
        "pad_token": tokenizer.pad_token,
        "stored_pad_token_id": int(tokenizer.pad_token_id),
        "token_column_dtype": "int32",
        "index_column_dtype": "int32",
        "max_input_length": MAX_INPUT_LENGTH,
        "max_length": MAX_LENGTH,
        "separate_response_length_limit": None,
        "generation_response_token_cap": MAX_LENGTH - MAX_INPUT_LENGTH,
        "token_lengths": token_length_statistics(accepted),
        "length_rejected_rows": len(length_rejected),
        "length_rejection_counts": _reason_counts(length_rejected),
        "matching": {
            "test_substring_threshold": TEST_SUBSTRING_THRESHOLD,
            "split_substring_threshold": SPLIT_SUBSTRING_THRESHOLD,
            "word_5gram_jaccard_threshold": WORD_5GRAM_THRESHOLD,
            "word_8gram_jaccard_threshold": WORD_8GRAM_THRESHOLD,
            "minhash_num_perm": MINHASH_NUM_PERM,
            "minhash_seed": MINHASH_SEED,
            "lsh_proposal_thresholds": LSH_PROPOSAL_THRESHOLDS,
            "verified_similar_pairs": len(similar_pairs),
            "gsm8k_contamination_matches": len(contamination_matches),
            "contaminated_problem_groups": len(contaminated_groups),
            "problem_groups": len(set(question_to_group.values())),
        },
        "split": {
            "validation_fraction_per_source": VALIDATION_FRACTION,
            "split_seed": SPLIT_SEED,
            "selected_source_questions": {
                source: len(questions) for source, questions in selected_questions.items()
            },
            "validation_subset_seed": VALIDATION_SUBSET_SEED,
            "validation_subset_size": VALIDATION_SUBSET_SIZE,
        },
        "accepted_rows_by_source": accepted_source_rows,
        "accepted_row_fractions": {
            source: count / total_accepted_rows
            for source, count in accepted_source_rows.items()
        },
        "accepted_source_questions_by_source": accepted_source_questions,
        "accepted_source_question_fractions": {
            source: count / total_source_questions
            for source, count in accepted_source_questions.items()
        },
        "fingerprints": {
            "openmath_input": _dataset_fingerprint(openmath_raw),
            "metamath_input": _dataset_fingerprint(metamath_raw),
            "gsm8k_test_input": _dataset_fingerprint(gsm8k_raw),
            "answer_valid_combined": _dataset_fingerprint(answer_valid),
            "tokenized_combined": _dataset_fingerprint(tokenized),
            "length_valid_combined": _dataset_fingerprint(length_valid),
            "decontaminated_combined": _dataset_fingerprint(accepted),
        },
        "outputs": {
            name: {
                "rows": len(dataset),
                "fingerprint": _dataset_fingerprint(dataset),
                "source_composition": source_composition(internal_outputs[name]),
            }
            for name, dataset in outputs.items()
        },
    }
    metadata["sources"]["openmath"]["revision"] = OPENMATH_REVISION
    metadata["sources"]["metamath"]["revision"] = METAMATH_REVISION
    metadata["sources"]["gsm8k_test"]["revision"] = GSM8K_REVISION
    metadata["tokenizer_revision"] = TOKENIZER_REVISION if args.tokenizer_name == TOKENIZER_NAME else None
    write_json(
        os.path.join(args.output_dir, "preprocessing_metadata.json"), metadata,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
