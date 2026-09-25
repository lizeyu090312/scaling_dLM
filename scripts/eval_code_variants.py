#!/usr/bin/env python3
"""Evaluate saved code generations with fixed source-handling variants."""

import argparse
import ast
import json
import os
import sys
from collections import Counter

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import eval_code


MODES = (
    "raw",
    "valid_first",
    "sanitize",
    "alias_single",
    "alias_last",
    "valid_first_alias_single",
    "sanitize_alias_single",
)
SINGLE_ALIAS_MODES = (
    "alias_single", "valid_first_alias_single", "sanitize_alias_single",
)
EVALPLUS_WORKERS = 2
MAX_MEMORY_BYTES = 6 * 1024 * 1024 * 1024
MIN_TIME_LIMIT_SECONDS = 20.0
REFERENCE_TIME_MULTIPLIER = 10.0

RESOURCE_SETTINGS = {
    "evalplus_workers": EVALPLUS_WORKERS,
    "max_memory_bytes_per_program": MAX_MEMORY_BYTES,
    "minimum_test_time_seconds": MIN_TIME_LIMIT_SECONDS,
    "reference_time_multiplier": REFERENCE_TIME_MULTIPLIER,
}

ALIAS_COUNTERS = (
    "applied",
    "skipped_invalid_source",
    "skipped_existing_binding",
    "skipped_no_function",
    "skipped_ambiguous",
)


class _ModuleBindingFinder(ast.NodeVisitor):
    """Find names bound outside function and class bodies."""

    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_FunctionDef(self, node):
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node):
        self.names.add(node.name)

    def visit_ClassDef(self, node):
        self.names.add(node.name)

    def visit_Lambda(self, node):
        pass

    def visit_Import(self, node):
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name != "*":
                self.names.add(alias.asname or alias.name)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_ListComp(self, node):
        pass

    def visit_SetComp(self, node):
        pass

    def visit_DictComp(self, node):
        pass

    def visit_GeneratorExp(self, node):
        pass


def _parse(code):
    if not code.strip():
        return None
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError):
        return None


def _has_module_binding(tree, name):
    finder = _ModuleBindingFinder()
    finder.visit(tree)
    return name in finder.names


def _add_alias(code, function, required_name):
    lines = code.splitlines(keepends=True)
    prefix = "".join(lines[: function.end_lineno])
    suffix = "".join(lines[function.end_lineno :])
    if prefix and not prefix.endswith(("\n", "\r")):
        prefix += "\n"
    return f"{prefix}{required_name} = {function.name}\n{suffix}"


class SourceTransformer:
    def __init__(self, mode, sanitizer):
        if mode not in MODES:
            raise ValueError(f"unknown evaluation mode: {mode}")
        self.mode = mode
        self.sanitizer = sanitizer
        self.counters = Counter()

    def __call__(self, code, *, entrypoint):
        tree = _parse(code)

        if self.mode == "raw":
            return code
        if self.mode == "sanitize":
            self.counters["sanitizer_calls"] += 1
            return self.sanitizer(code, entrypoint=entrypoint)
        if self.mode == "sanitize_alias_single":
            self.counters["sanitizer_calls"] += 1
            code = self.sanitizer(code, entrypoint=None)
            tree = _parse(code)
        if self.mode in ("valid_first", "valid_first_alias_single"):
            if tree is not None and self.mode == "valid_first":
                return code
            if tree is None:
                self.counters["sanitizer_calls"] += 1
                self.counters["sanitizer_fallbacks"] += 1
                return self.sanitizer(code, entrypoint=entrypoint)

        if tree is None:
            self.counters["skipped_invalid_source"] += 1
            return code
        if _has_module_binding(tree, entrypoint):
            self.counters["skipped_existing_binding"] += 1
            if self.mode == "sanitize_alias_single":
                self.counters["sanitizer_calls"] += 1
                return self.sanitizer(code, entrypoint=entrypoint)
            return code

        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not functions:
            self.counters["skipped_no_function"] += 1
            return code
        if self.mode in SINGLE_ALIAS_MODES and len(functions) != 1:
            self.counters["skipped_ambiguous"] += 1
            return code

        single_mode = self.mode in SINGLE_ALIAS_MODES
        function = functions[0] if single_mode else functions[-1]
        self.counters["applied"] += 1
        code = _add_alias(code, function, entrypoint)
        if self.mode == "sanitize_alias_single":
            self.counters["sanitizer_calls"] += 1
            return self.sanitizer(code, entrypoint=entrypoint)
        return code


def result_path(json_file, mode):
    if mode not in MODES:
        raise ValueError(f"unknown evaluation mode: {mode}")
    if not json_file.endswith(".jsonl"):
        raise ValueError(f"generation file must end in .jsonl: {json_file}")
    return json_file[:-6] + f".evalplus_{mode}.json"


def _fixed_evaluator(evaluate):
    def configured_evaluate(**kwargs):
        kwargs.update(
            parallel=EVALPLUS_WORKERS,
            min_time_limit=MIN_TIME_LIMIT_SECONDS,
            gt_time_limit_factor=REFERENCE_TIME_MULTIPLIER,
        )
        previous_memory_limit = os.environ.get("EVALPLUS_MAX_MEMORY_BYTES")
        os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = str(MAX_MEMORY_BYTES)
        try:
            return evaluate(**kwargs)
        finally:
            if previous_memory_limit is None:
                os.environ.pop("EVALPLUS_MAX_MEMORY_BYTES", None)
            else:
                os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = previous_memory_limit

    return configured_evaluate


def _atomic_write(path, value):
    temporary_path = path + ".variant.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def evaluate_variant(
    json_file,
    metadata_file,
    output_file,
    mode,
    *,
    source_path=None,
    dependencies=None,
):
    if dependencies is None:
        dependencies = eval_code._load_evalplus()
    get_humaneval, get_mbpp, sanitizer, evaluate = dependencies
    transformer = SourceTransformer(mode, sanitizer)

    result = eval_code.evaluate_generations(
        json_file,
        metadata_file,
        output_file,
        source_path=source_path,
        dependencies=(
            get_humaneval,
            get_mbpp,
            transformer,
            _fixed_evaluator(evaluate),
        ),
    )

    total = result["diagnostics"]["total"]
    diagnostics = result["diagnostics"]
    diagnostics["submitted_source_syntax"] = diagnostics.pop("sanitized_syntax")
    diagnostics["source_changes"] = diagnostics.pop("sanitizer_changes")
    fallback_count = transformer.counters["sanitizer_fallbacks"]
    diagnostics["sanitizer_fallbacks"] = {
        "count": fallback_count,
        "rate": fallback_count / total if total else 0.0,
    }
    diagnostics["alias"] = {
        name: transformer.counters[name] for name in ALIAS_COUNTERS
    }
    result["evaluation_mode"] = mode
    result["evaluation_settings"] = RESOURCE_SETTINGS.copy()
    _atomic_write(output_file, result)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--metadata_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--source_path")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_variant(
        args.json_file,
        args.metadata_file,
        args.output_file,
        args.mode,
        source_path=args.source_path,
    )


if __name__ == "__main__":
    main()
