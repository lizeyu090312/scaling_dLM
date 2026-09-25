"""Generate and report the fixed headline benchmarks using local ELF artifacts."""

import argparse
import hashlib
import json
import math
import subprocess
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

import headline_common as core
from configs.config import load_config_from_yaml
import eval_code
import eval_code_variants
import eval_mbpp500

DATA_ROOTS = {
    "gsm8k": "data/gsm-math-qwen3-embedding-0.6b-v1/gsm8k_test",
    "math500": "data/openmath-math-qwen3-embedding-0.6b-input256-max1024-5M-v1/math500_test",
    "mbpp": "data/opencodeinstruct-python-qwen3-embedding-0.6b-input512-max1024-v1/mbppplus_test",
    "mbpp500": "data/opencodeinstruct-python-qwen3-embedding-0.6b-input512-max1024-v1/mbpp500_test",
    "humaneval": "data/opencodeinstruct-python-qwen3-embedding-0.6b-input512-max1024-v1/humanevalplus_test",
}
QUESTIONS = {"gsm8k": 1319, "math500": 500, "mbpp": 378, "mbpp500": 500, "humaneval": 164}
LABELS = {"gsm8k": "GSM8K", "math500": "MATH-500", "mbpp": "MBPP-378", "mbpp500": "MBPP-500", "humaneval": "HumanEval"}
MODE = "sanitize_alias_single"
EVAL_SETTINGS = eval_code_variants.RESOURCE_SETTINGS
EVAL_VERSIONS = dict(evalplus_commit=eval_code.EVALPLUS_COMMIT,
                     humanevalplus=eval_code.HUMANEVAL_PLUS_VERSION,
                     mbppplus=eval_code.MBPP_PLUS_VERSION)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def generation_path(folder, seed, nfe):
    return Path(folder) / f"seed{seed}_nfe{nfe}.jsonl"


def metadata(manifest, seed, nfe):
    return dict({k: v for k, v in manifest.items() if k != "task_ids"}, seed=seed, nfe=nfe, master_nfe=8*nfe,
                denoiser_evaluations=nfe-1, decoder_evaluations=1)


def read_generations(folder, manifest, seed, nfe):
    path = generation_path(folder, seed, nfe)
    saved = json.loads(Path(str(path) + ".meta.json").read_text())
    raw = path.read_bytes()
    if saved != dict(metadata(manifest, seed, nfe), generation_sha256=hashlib.sha256(raw).hexdigest()):
        raise ValueError(f"Generation metadata or contents changed: {path}")
    rows = [json.loads(line) for line in raw.splitlines()]
    index_key = "example_id" if manifest["benchmark"] in ("gsm8k", "math500") else "id"
    if [r[index_key] for r in rows] != list(range(manifest["questions"])):
        raise ValueError("Missing, duplicated, or reordered generation rows")
    expected = metadata(manifest, seed, nfe)
    if any(any(r.get(k) != v for k, v in expected.items()) for r in rows):
        raise ValueError("Generation rows disagree with metadata")
    if "task_ids" in manifest and [r["task_id"] for r in rows] != manifest["task_ids"]:
        raise ValueError("Code benchmark task IDs differ from metadata")
    return saved, rows


@torch.no_grad()
def generate(args):
    benchmark = args.benchmark
    config = load_config_from_yaml(args.config)
    config.truncate_generation = False
    if config.model not in ("ELF-B", "ELF-L") or (benchmark != "gsm8k" and config.model != "ELF-L"):
        raise ValueError("The headline benchmark does not use this architecture")
    if config.use_model_attention_mask or config.max_length != 1024:
        raise ValueError("Headline evaluation requires length 1024 and no ELF attention mask")
    if config.max_input_length != (256 if benchmark in ("gsm8k", "math500") else 512):
        raise ValueError("Unexpected headline prompt length")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU for headline generation")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    model, checkpoint_info = core.load_elf_model(
        SimpleNamespace(checkpoint_path=args.checkpoint_path), config, tokenizer, device)
    encoder_config, encoder = core.load_clean_encoder(config, tokenizer, device)
    data_path = args.data_path or DATA_ROOTS[benchmark]
    dataset = load_from_disk(data_path)
    if len(dataset) != QUESTIONS[benchmark] or list(dataset["index"]) != list(range(len(dataset))):
        raise ValueError("Unexpected benchmark row count or indices")
    batch_size = args.batch_size or ({"gsm8k": 100 if config.model == "ELF-B" else 80,
                                     "math500": 50, "mbpp": 80, "mbpp500": 84, "humaneval": 82}[benchmark])
    model_name = config.model.replace("ELF-", "ELF-REG-") if config.reg_enabled else config.model + " baseline"
    manifest = dict(benchmark=benchmark, model_name=model_name, architecture=config.model,
                    objective="REPA+REG" if config.reg_enabled else "baseline",
                    checkpoint=args.checkpoint_path, checkpoint_bytes=Path(args.checkpoint_path).stat().st_size,
                    config=args.config, config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                    **checkpoint_info, dataset=data_path, questions=len(dataset),
                    dataset_fingerprint=dataset._fingerprint, seeds=args.seeds, nfes=args.nfe,
                    batch_size=batch_size, ratio=8, cfg=1.0,
                    self_cond_cfg=3.0 if benchmark == "gsm8k" else 2.0,
                    sampling_method="ode", time_schedule=config.time_schedule,
                    denoiser_p_mean=config.denoiser_p_mean, denoiser_p_std=config.denoiser_p_std,
                    denoiser_noise_scale=config.denoiser_noise_scale, t_eps=config.t_eps,
                    use_bf16=config.use_bf16, max_length=config.max_length,
                    max_input_length=config.max_input_length, pad_token=config.pad_token,
                    use_model_attention_mask=False, truncate_generation=False,
                    rng_protocol="per-example text/REG noise; seed+3000000 logit-normal grid")
    if benchmark not in ("gsm8k", "math500"):
        manifest["task_ids"] = list(dataset["task_id"])
        if len(set(manifest["task_ids"])) != len(dataset):
            raise ValueError("Duplicate code benchmark task IDs")
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output directory belongs to a different evaluation")
    write_json(manifest_path, manifest)
    decode = core.decode if benchmark == "gsm8k" else core.decode_math if benchmark == "math500" else core.decode_code
    requested = [(8, n) for n in args.nfe]
    for seed in args.seeds:
        if all(generation_path(folder, seed, n).exists() for n in args.nfe):
            for n in args.nfe:
                read_generations(folder, manifest, seed, n)
            continue
        schedules = {8*n: core.sampling_steps(config, seed=seed, device=device,
                     dtype=next(model.parameters()).dtype, n_steps=8*n-1) for n in args.nfe}
        records = {n: [] for n in args.nfe}
        for batch in core.dataloader(dataset, config, tokenizer, batch_size=batch_size):
            values = core.prepare_generation(batch, config, tokenizer, encoder, encoder_config, model, seed=seed)
            if benchmark == "math500":
                values["gold_answers"] = [dataset[i]["gold_answer"] for i in values["indices"]]
                values["problem_group_ids"] = [dataset[i]["problem_group_id"] for i in values["indices"]]
            elif benchmark != "gsm8k":
                values["task_ids"] = list(batch["task_id"])
            rows, _ = core.generate_batch(model, config, tokenizer, values, schedules, requested,
                sample_fn=partial(core.sample, self_cond_cfg_scale=manifest["self_cond_cfg"]), decode_fn=decode)
            for row in rows:
                nfe = row["nfe"]
                records[nfe].append(dict(metadata(manifest, seed, nfe), **row))
            print(f"{LABELS[benchmark]} seed={seed}: {values['indices'][-1]+1}/{len(dataset)}", flush=True)
        for nfe, rows in records.items():
            path = generation_path(folder, seed, nfe)
            raw = ("".join(json.dumps(r) + "\n" for r in rows)).encode()
            temporary = path.with_suffix(".jsonl.tmp")
            temporary.write_bytes(raw)
            temporary.replace(path)
            write_json(str(path) + ".meta.json", dict(metadata(manifest, seed, nfe),
                       generation_sha256=hashlib.sha256(raw).hexdigest()))
            read_generations(folder, manifest, seed, nfe)


def score_code(folder, manifest, image):
    for seed in manifest["seeds"]:
        for nfe in manifest["nfes"]:
            saved, _ = read_generations(folder, manifest, seed, nfe)
            path = generation_path(folder, seed, nfe)
            output = path.with_suffix(".eval.json")
            if not output.exists():
                subprocess.run(["bash", str(Path(__file__).with_name("evaluate_code_container.sh")),
                                image, str(path), str(output), manifest["benchmark"],
                                str(Path(manifest["dataset"]) / "evaluation_tasks.jsonl")], check=True)
            outcomes(json.loads(output.read_text()), saved, manifest)


def accuracy_statistics(matrix):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not matrix.size or not np.isin(matrix, [0, 1]).all():
        raise ValueError("Expected a nonempty binary questions-by-seeds matrix")
    per_seed = matrix.mean(axis=0) * 100
    n = matrix.shape[1]
    counts = matrix.sum(axis=1)
    subset = {str(k): float(np.mean([1 - (math.comb(n-int(c), k) / math.comb(n, k)
              if n-int(c) >= k else 0) for c in counts]) * 100) for k in range(1, n+1)}
    return dict(pass_at_1_mean=float(per_seed.mean()),
                pass_at_1_std=float(per_seed.std(ddof=1)) if n > 1 else None,
                per_seed_pass_at_1=per_seed.tolist(), subset_pass_at_k=subset)


def report(folder, manifest):
    benchmark = manifest["benchmark"]
    reports = []
    for nfe in manifest["nfes"]:
        matrices = {}
        for seed in manifest["seeds"]:
            saved, rows = read_generations(folder, manifest, seed, nfe)
            if benchmark in ("gsm8k", "math500"):
                column = "gsm8k_permissive_correct" if benchmark == "gsm8k" else "math_correct"
                columns = {LABELS[benchmark]: [r[column] for r in rows]}
            else:
                scored = outcomes(json.loads(generation_path(folder, seed, nfe).with_suffix(".eval.json").read_text()), saved, manifest)
                columns = {LABELS[benchmark]: [scored[t]["base_correct"] for t in manifest["task_ids"]]}
                if benchmark == "humaneval":
                    columns["HumanEval+"] = [scored[t]["plus_correct"] for t in manifest["task_ids"]]
            for label, values in columns.items():
                matrices.setdefault(label, []).append(values)
        for label, columns in matrices.items():
            reports.append(dict(benchmark=label, nfe=nfe, questions=manifest["questions"],
                                **accuracy_statistics(np.array(columns).T)))
    output = dict(settings=manifest, units="percent", standard_deviation="sample SD across seeds (ddof=1)", results=reports)
    write_json(Path(folder) / "metrics.json", output)
    for row in reports:
        print(f"{row['benchmark']} NFE={row['nfe']}: {row['pass_at_1_mean']:.2f}% (SD {row['pass_at_1_std']})")
    return output


def main(benchmark=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--config", required=True)
    gen.add_argument("--checkpoint_path", required=True)
    gen.add_argument("--output_dir", required=True)
    gen.add_argument("--data_path")
    gen.add_argument("--batch_size", type=int)
    gen.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 58)))
    gen.add_argument("--nfe", nargs="+", type=int, default=[64 if benchmark == "gsm8k" else 128])
    if benchmark is None:
        gen.add_argument("--benchmark", choices=("mbpp500", "mbpp", "humaneval"), required=True)
        score = commands.add_parser("score")
        score.add_argument("--output_dir", required=True)
        score.add_argument("--image", default="containers/evalplus.sif")
    else:
        gen.set_defaults(benchmark=benchmark)
    commands.add_parser("report").add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if (len(set(args.seeds)) != len(args.seeds) or len(set(args.nfe)) != len(args.nfe)
                or min(args.nfe) < 2 or (args.batch_size is not None and args.batch_size <= 0)):
            parser.error("Use unique seeds/NFE values, NFE >= 2, and a positive batch size")
        generate(args)
    else:
        manifest = json.loads((Path(args.output_dir) / "manifest.json").read_text())
        if (benchmark is not None and manifest["benchmark"] != benchmark) or (
                benchmark is None and manifest["benchmark"] not in ("mbpp500", "mbpp", "humaneval")):
            raise ValueError("Wrong benchmark entry point for this output directory")
        if args.command == "score":
            score_code(args.output_dir, manifest, args.image)
        else:
            report(args.output_dir, manifest)


def outcomes(result, saved, manifest):
    if manifest["benchmark"] == "mbpp500":
        return eval_mbpp500.outcomes(result, saved, manifest["task_ids"])
    if (result["generation_metadata"] != saved or result["evaluation_mode"] != MODE
            or result["evaluation_settings"] != EVAL_SETTINGS
            or result["benchmark"] != manifest["benchmark"] + "plus"
            or any(result["versions"].get(k) != v for k, v in EVAL_VERSIONS.items())):
        raise ValueError("incompatible evaluation result")
    native = result["evalplus_result"]
    if set(native["eval"]) != set(manifest["task_ids"]):
        raise ValueError("incomplete or unexpected evaluation task IDs")
    scores = {}
    for task_id, records in native["eval"].items():
        if len(records) != 1 or records[0]["task_id"] != task_id:
            raise ValueError("expected one result per task and seed")
        row = records[0]
        if any(row[k] not in ("pass", "fail", "timeout") for k in ("base_status", "plus_status")):
            raise ValueError("unknown execution status")
        base = row["base_status"] == "pass"
        scores[task_id] = dict(base_correct=int(base), plus_correct=int(base and row["plus_status"] == "pass"),
                               base_status=row["base_status"], plus_status=row["plus_status"])
    for label in ("base", "plus"):
        accuracy = sum(v[f"{label}_correct"] for v in scores.values()) / len(scores)
        if not math.isclose(accuracy, native["pass_at_k"][label]["pass@1"], abs_tol=1e-12):
            raise ValueError("native pass@1 disagrees with per-task outcomes")
    return scores

