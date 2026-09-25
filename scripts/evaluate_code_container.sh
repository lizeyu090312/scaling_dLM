#!/usr/bin/env bash
# Execute one complete benchmark file in a container without network or home access.
set -e
IMAGE=$(realpath "$1")
JSON_FILE=$(realpath "$2")
OUTPUT_FILE=$(realpath -m "$3")
BENCHMARK=$4
METADATA_FILE=$(realpath "${JSON_FILE}.meta.json")
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
test -f "$IMAGE"
command -v apptainer >/dev/null
mkdir -p "$PROJECT_ROOT/.cache"
SCRATCH_DIR=$(mktemp -d "$PROJECT_ROOT/.cache/code-eval.XXXXXX")
trap 'rm -rf "$SCRATCH_DIR"' EXIT
EVALUATOR=(python /opt/eval_code_variants.py
    --json_file /input/generations.jsonl --metadata_file /input/generations.meta.json
    --output_file /scratch/result.json --mode sanitize_alias_single)
EXTRA_BINDS=()
case "$BENCHMARK" in
    mbpp|humaneval) ;;
    mbpp500)
        TASKS_FILE=$(realpath "$5")
        EXTRA_BINDS=(--bind "$TASKS_FILE:/input/tasks.jsonl:ro")
        EVALUATOR=(python /opt/eval_mbpp500.py --json_file /input/generations.jsonl
            --metadata_file /input/generations.meta.json --tasks_file /input/tasks.jsonl
            --output_file /scratch/result.json)
        ;;
    *) echo "Unknown code benchmark: $BENCHMARK" >&2; exit 1 ;;
esac
apptainer exec --userns --containall --cleanenv --no-eval --no-privs \
    --no-mount home,cwd,hostfs,bind-paths --net --network none --pwd /scratch \
    --bind "$JSON_FILE:/input/generations.jsonl:ro" \
    --bind "$METADATA_FILE:/input/generations.meta.json:ro" \
    --bind "$SCRATCH_DIR:/scratch:rw" \
    "${EXTRA_BINDS[@]}" "$IMAGE" "${EVALUATOR[@]}"
mv "$SCRATCH_DIR/result.json" "$OUTPUT_FILE"
