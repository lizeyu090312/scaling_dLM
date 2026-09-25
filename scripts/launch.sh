#!/usr/bin/env bash
# Run from the repository root, inside your GPU allocation.
set -e
if [[ $# -lt 2 ]]; then
    echo "usage: bash scripts/launch.sh <train|eval> <config.yml> [extra args...]" >&2
    exit 1
fi
MODE=$1
CONFIG=$2
shift 2
case "$MODE" in
    train) ENTRY=src/train.py ;;
    eval) ENTRY=src/eval.py ;;
    *) echo "Unknown mode: $MODE" >&2; exit 1 ;;
esac
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
exec python -m torch.distributed.run --standalone --nproc_per_node="${NGPU:-1}" \
    "$ENTRY" --config "$CONFIG" "$@"
