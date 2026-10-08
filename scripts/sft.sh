#!/usr/bin/env bash
# Supervised initialization: teacher trajectories -> parquet -> LoRA -> merged checkpoint.
# These are the settings the released SFT checkpoints were trained with. Trailing
# arguments are Hydra overrides for the training stage.
#   ENV=alfworld OPENAI_BASE_URL=... OPENAI_API_KEY=... bash scripts/sft.sh
#   STAGES="train merge" bash scripts/sft.sh trainer.total_epochs=2
#   STAGES=collect CONFIG=... bash scripts/sft.sh      # collection alone, resumable
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/training/verl-agent:$ROOT:${PYTHONPATH:-}"
ENV="${ENV:?alfworld | webshop | webarena}"
CONFIG="${CONFIG:-$ROOT/configs/sft/$ENV.yaml}"
BASE="${BASE:-Qwen/Qwen3-8B}"
OUT="${OUT:-$ROOT/results/${ENV}_sft}"
TEACHER_DIR="${TEACHER_DIR:-$OUT/teacher}"
DATA_DIR="${DATA_DIR:-$OUT/data}"
SAVE_DIR="${SAVE_DIR:-$OUT/lora}"
MERGED_DIR="${MERGED_DIR:-$OUT/merged}"
STAGES="${STAGES:-collect convert train merge}"

# The trainer asserts (train_batch_size / GPUS) % micro_batch_size_per_gpu == 0, so
# GPUS and the micro batch move together. Note `data.micro_batch_size` (without
# _per_gpu) is not read by this trainer -- only the _per_gpu form below is.
case "$ENV" in
  alfworld|webshop)
    GPUS="${GPUS:-8}"
    PER_ENV=(
      data.train_batch_size=16
      data.micro_batch_size_per_gpu=2
      data.max_length=4096
      data.truncation=right
      trainer.total_epochs=4
    ) ;;
  webarena)
    # Long accumulated conversations: fewer, larger rows, and truncate on the left
    # so an overlong prompt loses stale pages rather than the instruction.
    GPUS="${GPUS:-2}"
    PER_ENV=(
      data.train_batch_size=8
      data.micro_batch_size_per_gpu=1
      data.max_length=16384
      data.truncation=left
      trainer.total_epochs=3
    ) ;;
  *) echo "unknown ENV=$ENV" >&2; exit 1 ;;
esac

ARGS=(
  "data.train_files=$DATA_DIR/sft_train.parquet"
  "data.val_files=$DATA_DIR/sft_val.parquet"
  data.prompt_key=prompt
  data.response_key=response
  data.prompt_dict_keys=null
  data.response_dict_keys=null
  data.multiturn.enable=false
  "${PER_ENV[@]}"
  "model.partial_pretrain=$BASE"
  model.lora_rank=64
  model.lora_alpha=128
  model.target_modules=all-linear
  model.enable_gradient_checkpointing=true
  use_remove_padding=true
  optim.lr=1e-5
  "trainer.default_local_dir=$SAVE_DIR"
  trainer.default_hdfs_dir=null
  "trainer.project_name=${ENV}-sft"
  "trainer.experiment_name=${ENV}-sft-lora"
  "trainer.logger=[console]"
)

for stage in $STAGES; do
  echo "[sft] === $stage ==="
  case "$stage" in
    collect) python -m evoharness.sft.collect --config "$CONFIG" "out_dir=$TEACHER_DIR" ;;
    convert) python -m evoharness.sft.dataset --config "$CONFIG" \
               "out_dir=$TEACHER_DIR" "sft.data_dir=$DATA_DIR" ;;
    train)   torchrun --standalone --nnodes=1 --nproc_per_node="$GPUS" \
               -m verl.trainer.fsdp_sft_trainer "${ARGS[@]}" "$@" ;;
    merge)   python -m evoharness.sft.merge --base "$BASE" \
               --adapter "$SAVE_DIR" --out "$MERGED_DIR" ;;
    # Opt in with STAGES="... sanity"; needs scripts/serve_vllm.sh on $MERGED_DIR.
    sanity)  python -m evoharness eval --config "$ROOT/configs/eval/$ENV.yaml" \
               "policy.model=$MERGED_DIR" limit=20 "out_dir=$OUT/sanity" ;;
    *) echo "unknown stage $stage" >&2; exit 1 ;;
  esac
done
