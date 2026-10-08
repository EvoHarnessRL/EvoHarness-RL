#!/usr/bin/env bash
# GRPO with the harness in the loop, on training/verl-agent (training path A).
#   ENV=alfworld MODEL=/path/to/sft_checkpoint BANK=/path/to/bank.json \
#   EVOLVER_MODEL=<served name> EVOLVER_URL=http://host:8001/v1 bash scripts/train.sh [hydra overrides]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/training/verl-agent:$ROOT:${PYTHONPATH:-}"
ENV="${ENV:?alfworld | webshop | webarena}"
MODEL="${MODEL:?SFT checkpoint to start from}"
DATA_DIR="${DATA_DIR:-$HOME/data/evoharness}"

case "$ENV" in
  alfworld) TRAIN=16; VAL=128 ;;
  webshop)  TRAIN=16; VAL=128 ;;
  webarena) TRAIN=32; VAL=50 ;;
  *) echo "unknown ENV=$ENV" >&2; exit 1 ;;
esac

python -m evoharness.rl.prepare_data --env "$ENV" --out-dir "$DATA_DIR" --train-size "$TRAIN" --val-size "$VAL"

ARGS=(
  "actor_rollout_ref.model.path=$MODEL"
  "data.train_files=$DATA_DIR/$ENV/train.parquet"
  "data.val_files=$DATA_DIR/$ENV/test.parquet"
)
if [ -n "${BANK:-}" ]; then
  ARGS+=("env.harness.bank.path=$BANK")
fi
if [ -n "${EVOLVER_MODEL:-}" ]; then
  # EVOLVER_THINKING=false only for vLLM-served reasoning models (Qwen3); hosted APIs reject the flag.
  ARGS+=("env.harness.bank.evolver={model:$EVOLVER_MODEL,base_url:${EVOLVER_URL:-null},temperature:0.7,max_tokens:6144,enable_thinking:${EVOLVER_THINKING:-null}}")
fi

python -m evoharness.rl.train --config-name "$ENV" "${ARGS[@]}" "$@"
