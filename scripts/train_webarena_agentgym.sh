#!/usr/bin/env bash
# WebArena GRPO on the AgentGym-RL trainer (the setup behind the paper's WebArena RL row):
# accumulated multi-turn history, meta-actions intercepted inside the vLLM rollout,
# shaped reward, and skill-bank consolidation at epoch boundaries.
#
#   MODEL=/path/to/sft_checkpoint BANK_PATH=/path/to/seed_bank.json \
#   REFLECTOR_MODEL=<model> OPENAI_BASE_URL=... OPENAI_API_KEY=... \
#   bash scripts/train_webarena_agentgym.sh [hydra overrides]
#
# Needs the AgentGym-RL environment (training/agentgym-rl, vLLM 0.8.5) and a running
# agentenv-webarena server (ENV_SERVER_URL). Set HARNESS=0 for the sparse GRPO baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RL_DIR="$ROOT/training/agentgym-rl/AgentGym-RL"
# The rollout finds harness/ next to AgentGym-RL/; the fork's verl must win on sys.path.
export PYTHONPATH="$RL_DIR:$ROOT/training/agentgym-rl:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL="${MODEL:?SFT checkpoint to start from}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints/webarena_agentgym}"
EXP_NAME="${EXP_NAME:-webarena_grpo_harness}"
HARNESS="${HARNESS:-1}"
mkdir -p "$SAVE_DIR"

ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.rounds_ctrl.type=scaling_inter_stepwise
  "algorithm.rounds_ctrl.rounds=${ROUNDS:-[10,14,17]}"
  algorithm.rounds_ctrl.steps_scaling_inter=80
  algorithm.kl_ctrl.kl_coef=0.01
  data.train_file="$RL_DIR/AgentItemId/webarena_train.json"
  data.train_batch_size=32
  data.max_prompt_length=2048
  data.max_response_length=8192
  data.eval_path="$RL_DIR/AgentEval/webarena"
  data.eval_task_name=webarena
  data.eval_n_samples=1
  data.eval_max_rounds=15
  actor_rollout_ref.agentgym.task_name=webarena
  "actor_rollout_ref.agentgym.env_addr=${ENV_SERVER_URL:-http://127.0.0.1:36005}"
  actor_rollout_ref.agentgym.timeout=600
  "actor_rollout_ref.model.path=$MODEL"
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.01
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.ppo_epochs=2
  actor_rollout_ref.actor.ppo_mini_batch_size=4
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.n=4
  actor_rollout_ref.rollout.max_model_len=32768
  actor_rollout_ref.rollout.max_tokens=512
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7
  "actor_rollout_ref.rollout.rollout_log_dir=$SAVE_DIR/rollout_logs"
  "trainer.default_local_dir=$SAVE_DIR"
  "trainer.logger=['console','wandb']"
  trainer.project_name=webarena_grpo
  "trainer.experiment_name=$EXP_NAME"
  "trainer.n_gpus_per_node=${GPUS_PER_NODE:-8}"
  "trainer.nnodes=${NNODES:-1}"
  trainer.total_epochs=25
  trainer.save_freq=11
  trainer.eval_freq=11
  trainer.val_before_train=False
)

if [ "$HARNESS" = "1" ]; then
  BANK_DIR="$SAVE_DIR/banks"
  mkdir -p "$BANK_DIR"
  if [ ! -f "$BANK_DIR/shared.json" ]; then
    if [ -n "${BANK_PATH:-}" ]; then
      cp "$BANK_PATH" "$BANK_DIR/shared.json"
    else
      echo '{"general_skills": [], "task_specific_skills": {}, "common_mistakes": []}' > "$BANK_DIR/shared.json"
    fi
  fi
  BANK_PATH="${BANK_PATH:-$BANK_DIR/shared.json}"
  ARGS+=(
    reward_shaping.enable=True
    reward_shaping.lambda_eff=0.02
    reward_shaping.lambda_div=0.02
    reward_shaping.lambda_spam=0.01
    reward_shaping.spam_cap=10
    reward_shaping.lambda_invalid=0.01
    harness.enable=True
    harness.prompt_variant=paced
    harness.recall_top_k=3
    harness.max_obs_chars=12000
    harness.evolve=True
    "harness.reflector_model=${REFLECTOR_MODEL:?model used to reflect on episodes}"
    "harness.reconciler_model=${RECONCILER_MODEL:-$REFLECTOR_MODEL}"
    harness.skill_reconcile=True
    harness.skill_min_evidence=3
    harness.evolve_max_episodes=48
    harness.evolve_per_type_success=2
    harness.evolve_per_type_fail=2
    harness.delete_veto_value=3
    harness.bank_cap_per_category=80
    harness.bank_max_field_chars=300
    "harness.bank_path=$BANK_PATH"
    "harness.bank_dir=$BANK_DIR"
    "harness.bank_snapshot=$BANK_DIR/shared.json"
  )
fi

cd "$RL_DIR"
HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_ppo "${ARGS[@]}" "$@"
