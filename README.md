# ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks

This repository is the official repository for "ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks". The implementation builds on the PyTorch version of [ELF: Embedded Language Flows](https://github.com/lillian039/ELF). We are grateful for the authors of ELF for open-sourcing their training and evaluation code.

Download the prepared datasets and checkpoints to the paths below.

## Prepared data and checkpoints

| Prepared dataset directory under `data/` | Dataset link |
| --- | --- |
| `gsm-math-qwen3-embedding-0.6b-v1/` | [Hugging Face](https://huggingface.co/datasets/zl310/elf-reg-data/tree/main/gsm-math-qwen3-embedding-0.6b-v1) |
| `openmath-math-qwen3-embedding-0.6b-input256-max1024-5M-v1/` | [Hugging Face](https://huggingface.co/datasets/zl310/elf-reg-data/tree/main/openmath-math-qwen3-embedding-0.6b-input256-max1024-5M-v1) |
| `opencodeinstruct-python-qwen3-embedding-0.6b-input512-max1024-v1/` | [Hugging Face](https://huggingface.co/datasets/zl310/elf-reg-data/tree/main/opencodeinstruct-python-qwen3-embedding-0.6b-input512-max1024-v1) |

The code dataset includes HumanEval/MBPP benchmarks and test suites. You can also prepare the data yourself instead of downloading them (see below).

| Task | Model | Training epoch | Local checkpoint path | Checkpoint link |
| --- | --- | ---: | --- | --- |
| GSM8K | ELF-B baseline | 18 | `checkpoints/gsm8k/elf_b_baseline/checkpoint_84528` | [Hugging Face](https://huggingface.co/zl310/elf-reg/blob/main/gsm8k/elf_b_baseline/checkpoint_84528) |
| GSM8K | ELF-REG-B | 12 | `checkpoints/gsm8k/elf_b_repa_reg/checkpoint_56352` | [Hugging Face](https://huggingface.co/zl310/elf-reg/blob/main/gsm8k/elf_b_repa_reg/checkpoint_56352) |
| GSM8K | ELF-L baseline | 15 | `checkpoints/gsm8k/elf_l_baseline/checkpoint_70440` | [Hugging Face](https://huggingface.co/zl310/elf-reg/blob/main/gsm8k/elf_l_baseline/checkpoint_70440) |
| GSM8K | ELF-REG-L | 12 | `checkpoints/gsm8k/elf_l_repa_reg/checkpoint_56352` | [Hugging Face](https://huggingface.co/zl310/elf-reg/blob/main/gsm8k/elf_l_repa_reg/checkpoint_56352) |
| MATH-500 | ELF-L baseline | 8 | `checkpoints/math500/elf_l_baseline/checkpoint_55824` | [Hugging Face](https://huggingface.co/zl310/elf-reg/blob/main/math500/elf_l_baseline/checkpoint_55824) |
| MATH initialization (GSM8K) | ELF-L baseline | 7.5 | `checkpoints/math_initialization/elf_l_baseline/checkpoint_35220` | [Hugging Face](https://huggingface.co/zl310/elf-reg/blob/main/math_initialization/elf_l_baseline/checkpoint_35220) |
| MATH-500 | ELF-REG-L | 8 | `checkpoints/math500/elf_l_repa_reg/checkpoint_55824` | [Hugging Face](https://huggingface.co/zl310/elf-reg/blob/main/math500/elf_l_repa_reg/checkpoint_55824) |
| MATH initialization (GSM8K) | ELF-REG-L | 7.5 | `checkpoints/math_initialization/elf_l_repa_reg/checkpoint_35220` | [Hugging Face](https://huggingface.co/zl310/elf-reg/blob/main/math_initialization/elf_l_repa_reg/checkpoint_35220) |
| Code | ELF-L baseline | 12 | `checkpoints/code/elf_l_baseline/checkpoint_64248` | [Hugging Face](https://huggingface.co/zl310/elf-reg/blob/main/code/elf_l_baseline/checkpoint_64248) |
| Code | ELF-REG-L | 12 | `checkpoints/code/elf_l_repa_reg/checkpoint_64248` | [Hugging Face](https://huggingface.co/zl310/elf-reg/blob/main/code/elf_l_repa_reg/checkpoint_64248) |

The checkpoints total about 100 GB. Preprocessed data occupies about 29 GB. Note: the MATH training requires initializing the model weights from the epoch-7.5 GSM8K checkpoints at EMA 0.999.

## Installation

Run commands from the repository root:

```bash
conda create -n elf python=3.10 -y
conda activate elf
pip install -r requirements.txt
```

Build the image for code-scoring using Apptainer: 

```bash
apptainer build --fakeroot containers/evalplus.sif containers/evalplus.def
```

## Data preparation

The data preparation scripts download public datasets and the Qwen3-Embedding-0.6B tokenizer. Lower `--num_proc` if needed to reduce CPU load.

### GSM8K training and evaluation

```bash
python scripts/prepare_gsm_math_qwen3.py \
  --output_dir data/gsm-math-qwen3-embedding-0.6b-v1 --num_proc 10
```

The prepared training split has 2,404,028 examples. The directory also contains the 1,319-question GSM8K test split and three validation splits: `validation_full`, `validation_unique`, and `validation_1024`. We use `validation_1024` during training.

### MATH training and evaluation

```bash
python scripts/prepare_openmath_math_qwen3.py --openmath_split train_5M \
  --output_dir data/openmath-math-qwen3-embedding-0.6b-input256-max1024-5M-v1 \
  --max_input_length 256 --max_response_length none --max_length 1024 \
  --truncate_math500_prompts --num_proc 10
```

The prepared training split has 3,572,583 examples. The directory also contains the 500 MATH-500 test questions and the validation splits.

### Code training and evaluation

```bash
python scripts/prepare_opencodeinstruct_python_qwen3.py \
  --output_dir data/opencodeinstruct-python-qwen3-embedding-0.6b-input512-max1024-v1 \
  --max_input_length 512 --max_length 1024 --num_proc 16

python scripts/prepare_mbpp500_qwen3.py
```

The first command produces the training dataset (2,740,479 examples), together with the 378-question MBPP and 164-question HumanEval benchmark inputs. It saves the benchmark test suites under `benchmark_tests/`. The second command adds `mbpp500_test/`, which includes the 500 original MBPP tasks.

## Training

Configs are saved in `configs/training/`.

| Config | Model | Epochs |
| --- | --- | ---: |
| `gsm8k_elf_b_baseline.yml` | ELF-B baseline | 18 |
| `gsm8k_elf_b_repa_reg.yml` | ELF-REG-B | 12 |
| `gsm8k_elf_l_baseline.yml` | ELF-L baseline | 15 |
| `gsm8k_elf_l_repa_reg.yml` | ELF-REG-L | 12 |
| `math500_elf_l_baseline.yml` | ELF-L baseline | 8 |
| `math500_elf_l_repa_reg.yml` | ELF-REG-L | 8 |
| `code_elf_l_baseline.yml` | ELF-L baseline | 12 |
| `code_elf_l_repa_reg.yml` | ELF-REG-L | 12 |

For example, train ELF-REG-L on GSM8K with two GPUs:

```bash
NGPU=2 bash scripts/launch.sh train configs/training/gsm8k_elf_l_repa_reg.yml
```

### MATH initialization

MATH training for both ELF-REG and the baseline starts from the corresponding ELF-L GSM8K checkpoint at epoch 7.5, using EMA 0.999. After training the baseline and REPA+REG GSM8K models, place their initialization checkpoints as follows:

```bash
mkdir -p checkpoints/math_initialization/elf_l_baseline
mkdir -p checkpoints/math_initialization/elf_l_repa_reg
cp outputs/gsm8k_elf_l_baseline/checkpoint_35220 \
  checkpoints/math_initialization/elf_l_baseline/
cp outputs/gsm8k_elf_l_repa_reg/checkpoint_35220 \
  checkpoints/math_initialization/elf_l_repa_reg/

NGPU=2 bash scripts/launch.sh train configs/training/math500_elf_l_baseline.yml
NGPU=2 bash scripts/launch.sh train configs/training/math500_elf_l_repa_reg.yml
```

If a different GPU setup produces different paths to checkpoints, point `resume_only_weights` to the checkpoint saved at epoch 7.5.

## Headline evaluation

When training using the code dataset, periodic training-time evaluation saves generated code without scoring; use the headline generation and scoring commands below to report results.

The `evaluate_*` scripts reproduce the headline early-stop denoising process. The launcher's `eval` mode provides the original full-span evaluator; use the commands below to reproduce Tables 1 and 2.

Each command defaults to seeds 42 through 57, EMA 0.9999, ODE sampling, and early-stop ratio 8. It takes `NFE - 1` denoiser steps, then decodes the predicted-clean state. The NFE includes the decoder call. We use CFG=1; we use SCCFG=3 for GSM8K and SCCFG=2 for MATH-500 and code.

The commands below use pretrained checkpoints from the [checkpoint table](#prepared-data-and-checkpoints). For models you train yourself, use the appropriate path.

### GSM8K

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_gsm8k.py generate \
  --config configs/training/gsm8k_elf_l_repa_reg.yml \
  --checkpoint_path checkpoints/gsm8k/elf_l_repa_reg/checkpoint_56352 \
  --output_dir outputs/eval_gsm8k_elf_reg_l

python scripts/evaluate_gsm8k.py report --output_dir outputs/eval_gsm8k_elf_reg_l
```

By default, we use 64 NFE; batch sizes are 100 for ELF-B and 80 for ELF-L.

### MATH-500

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_math500.py generate \
  --config configs/training/math500_elf_l_repa_reg.yml \
  --checkpoint_path checkpoints/math500/elf_l_repa_reg/checkpoint_55824 \
  --output_dir outputs/eval_math500_elf_reg_l

python scripts/evaluate_math500.py report --output_dir outputs/eval_math500_elf_reg_l
```

By default, we use 128 NFE with batch size 50.

### Code benchmarks

Run code generation and code scoring separately. Scoring executes generated code inside the isolated container:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_code.py generate \
  --benchmark humaneval --config configs/training/code_elf_l_repa_reg.yml \
  --checkpoint_path checkpoints/code/elf_l_repa_reg/checkpoint_64248 \
  --output_dir outputs/eval_humaneval_elf_reg_l

python scripts/evaluate_code.py score \
  --output_dir outputs/eval_humaneval_elf_reg_l --image containers/evalplus.sif
python scripts/evaluate_code.py report --output_dir outputs/eval_humaneval_elf_reg_l
```

Use `--benchmark mbpp` for MBPP-378 or `--benchmark mbpp500` for MBPP-500 (remember to change the output directory). By default, we use 128 NFE.

### Reports and reference results

The generation script accepts `--seeds`, `--nfe`, `--batch_size`, and `--data_path` overrides. For example, `--seeds 42 --nfe 4` runs one seed at a smaller budget. The `report` command writes mean pass@1, its standard deviation across seeds, and subset pass@k to `metrics.json`.

These are the expected results (standard deviations in parentheses):

| Model | GSM8K, 64 NFE | MATH-500, 128 NFE |
| --- | ---: | ---: |
| ELF-B baseline | 37.71 (0.95) | |
| ELF-REG-B | 38.11 (1.07) | |
| ELF-L baseline | 51.85 (0.88) | 10.55 (0.90) |
| ELF-REG-L | 55.96 (0.94) | 13.39 (1.03) |

| Model | MBPP-500 | MBPP-378 | HumanEval | HumanEval+ |
| --- | ---: | ---: | ---: | ---: |
| ELF-L baseline | 17.19 (1.34) | 26.75 (1.58) | 19.66 (1.52) | 18.75 (1.52) |
| ELF-REG-L | 19.10 (1.24) | 28.92 (1.38) | 22.56 (2.25) | 21.46 (2.10) |

Hardware and numerical-library differences can affect sampled outputs.

## License

The code retains the MIT license in `LICENSE`. 

## Acknowledgements

The implementation builds on the PyTorch version of [ELF: Embedded Language Flows](https://github.com/lillian039/ELF). We thank the authors of ELF for open-sourcing their training and evaluation code.

## Citation

If you find this work helpful for your research, please consider citing our paper!

```bib
@misc{elf_reg2026,
      title={ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks}, 
      author={Zeyu Michael Li and William Xingxu Chen and Bingshuo Qian and Jiayin Liu and Xiang Cheng},
      year={2026},
      eprint={2609.29102},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.29102}, 
}
```