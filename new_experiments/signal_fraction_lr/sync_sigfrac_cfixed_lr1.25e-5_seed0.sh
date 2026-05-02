#!/bin/bash
set -xeuo pipefail

# Phase 1: find c_fixed — eta_c=0 freezes c_t after handoff
# alpha_base=1.25e-5, r_boot=0.05 → c_fixed = 1.25e-5/0.05 = 2.5e-4
# typical alpha_t = c_fixed * r_typical(0.018) ≈ 4.5e-6

VERL_ROOT=/data/250010176/codes/verl/verl
DATA_ROOT=/data/250010176/data
MODEL_PATH=/data/250010176/codes/models/DeepSeek-R1-Distill-Qwen-1.5B
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Resume training configuration
RESUME_MODE=${RESUME_MODE:-"disable"}  # Options: "auto", "resume_path", "disable"
RESUME_FROM_PATH=${RESUME_FROM_PATH:-""}
USER_CKPTS_DIR=${CKPTS_DIR:-""}

# Checkpoint retention configuration
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-2}

# Use existing checkpoint directory if resuming, otherwise create new one
if [ "$RESUME_MODE" = "resume_path" ] && [ -n "$RESUME_FROM_PATH" ]; then
  CKPTS_DIR=$(dirname "$RESUME_FROM_PATH")
  TIMESTAMP=$(basename "$CKPTS_DIR" | sed 's/.*-//')
  echo "Resuming from specified checkpoint path: $RESUME_FROM_PATH"
  echo "Using checkpoint directory: $CKPTS_DIR"
elif [ -n "$USER_CKPTS_DIR" ]; then
  CKPTS_DIR="$USER_CKPTS_DIR"
  TIMESTAMP=$(basename "$CKPTS_DIR" | sed 's/.*-//')
  echo "Using user-specified checkpoint directory: $CKPTS_DIR"
elif [ "$RESUME_MODE" = "auto" ]; then
    LATEST_CKPT=$(find ${VERL_ROOT}/ckpts/DeepSeek1.5B -name "DeepSeek1.5B-Sync-8gpu-sigfrac-cfixed-lr1.25e-5-seed0-*" -type d 2>/dev/null | sort | tail -1)
  if [ -n "$LATEST_CKPT" ]; then
    CKPTS_DIR="$LATEST_CKPT"
    TIMESTAMP=$(basename "$CKPTS_DIR" | sed 's/.*-//')
    echo "Auto-found existing checkpoint directory: $CKPTS_DIR"
  else
    CKPTS_DIR=${VERL_ROOT}/ckpts/DeepSeek1.5B/DeepSeek1.5B-Sync-8gpu-sigfrac-cfixed-lr1.25e-5-seed0-${TIMESTAMP}
    echo "No existing checkpoint found, creating new directory: $CKPTS_DIR"
  fi
else
    CKPTS_DIR=${VERL_ROOT}/ckpts/DeepSeek1.5B/DeepSeek1.5B-Sync-8gpu-sigfrac-cfixed-lr1.25e-5-seed0-${TIMESTAMP}
    echo "Resume disabled, creating new checkpoint directory: $CKPTS_DIR"
fi

LOG_DIR=${VERL_ROOT}/logs
LOG_FILE="${LOG_DIR}/sync_sigfrac_cfixed_lr1.25e-5_seed0_${TIMESTAMP}.log"
mkdir -p "$CKPTS_DIR" "$LOG_DIR"

if [ -n "${MASTER_ADDR:-}" ]; then HN="$MASTER_ADDR"; elif [ -n "${NODE_0_IP:-}" ]; then HN="$NODE_0_IP"; else HN="$(hostname -I | awk '{print $1}')"; fi
HEAD_IP="$(getent hosts "$HN" | awk '{print $1}' | head -1)"
[ -z "$HEAD_IP" ] && HEAD_IP="$HN"

source /data/250010176/yrh/miniconda3/etc/profile.d/conda.sh
conda activate verl2
cd "${VERL_ROOT}/.."
export PYTHONPATH="${VERL_ROOT}/..:${PYTHONPATH:-}"

# Data: 17k-sampled train, DeepScaler test sets (same as 32b)
DEEPSCALER_DIR=/data/250010176/dataset/deepscaler
export TRAIN_FILE=${DATA_ROOT}/dapo-math-17k-sampled.parquet
export TEST_FILES=${TEST_FILES:-"[${DEEPSCALER_DIR}/aime24.parquet,${DEEPSCALER_DIR}/aime25.parquet,${DEEPSCALER_DIR}/gpqa_diamond_100.parquet,${DEEPSCALER_DIR}/minerva_fixed_100.parquet,${DEEPSCALER_DIR}/olympiad_bench_fixed_100.parquet]"}
export MODEL_PATH="$MODEL_PATH"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=1

# Resource allocation: 8 GPUs total
export NNODES=1
export NGPUS_PER_NODE=8
export CKPTS_DIR="$CKPTS_DIR"

IS_HEAD=0
NODE_RANK_CAND="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-}}"
if [ "${NODE_RANK_CAND:-}" = "0" ] || [ "${RANK:-}" = "0" ]; then IS_HEAD=1; fi
echo "HEAD_IP=$HEAD_IP IS_HEAD=$IS_HEAD RANK=${RANK:-unset} NODE_RANK=${NODE_RANK_CAND:-unset}"

RAY=/data/250010176/yrh/miniconda3/envs/verl2/bin/ray
$RAY stop -f || true
pkill -f redis-server || true
rm -rf /tmp/ray || true

if [ "$IS_HEAD" = "1" ]; then
  $RAY start --head --node-ip-address="$HEAD_IP" --port=6379 --dashboard-port=8265 --num-cpus=$(nproc)
  ready=0
  for i in $(seq 1 30); do
    if $RAY status >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    echo "Ray head failed to start"
    exit 1
  fi
  export RAY_ADDRESS="$HEAD_IP:6379"

  python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILES}" \
    data.prompt_key=prompt \
    data.seed=0 \
    data.truncation='left' \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.train_batch_size=128 \
    data.filter_overlong_prompts=True \
    ++actor_rollout_ref.rollout.n=8 \
    ++actor_rollout_ref.actor.rollout_n=8 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    ++actor_rollout_ref.actor.clip_ratio_low=0.2 \
    ++actor_rollout_ref.actor.clip_ratio_high=0.2 \
    ++actor_rollout_ref.actor.clip_ratio_c=10.0 \
    ++actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=12288 \
    ++actor_rollout_ref.actor.use_dynamic_bsz=True \
    ++actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    ++actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    ++actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((1024+8192+300)) \
    ++actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((1024+8192+300)) \
    ++actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((1024+8192+300)) \
    ++actor_rollout_ref.rollout.name=vllm \
    ++actor_rollout_ref.model.path="${MODEL_PATH}" \
    ++actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.actor.optim.lr=1.25e-5 \
    ++actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    ++actor_rollout_ref.actor.optim.weight_decay=0.1 \
    ++actor_rollout_ref.actor.optim.lr_scheduler_type=signal_fraction \
    ++actor_rollout_ref.actor.optim.signal_fraction_eta_c=0.0 \
    ++actor_rollout_ref.actor.optim.signal_fraction_c_min=1e-8 \
    ++actor_rollout_ref.actor.optim.signal_fraction_c_max=1e-2 \
    ++actor_rollout_ref.actor.optim.signal_fraction_calib_freq=5 \
    ++actor_rollout_ref.actor.optim.signal_fraction_phi_ema_beta=0.9 \
    ++actor_rollout_ref.actor.optim.signal_fraction_r_min=0.01 \
    ++actor_rollout_ref.actor.optim.signal_fraction_calib_frac=0.0 \
    ++actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    ++actor_rollout_ref.actor.ppo_micro_batch_size=null \
    ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    ++actor_rollout_ref.actor.fsdp_config.param_offload=True \
    ++actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    ++actor_rollout_ref.actor.entropy_coeff=0 \
    ++actor_rollout_ref.actor.grad_clip=1.0 \
    ++actor_rollout_ref.actor.loss_agg_mode=token-mean \
    ++actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    ++actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    ++actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    ++actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    ++actor_rollout_ref.rollout.enable_chunked_prefill=True \
    ++actor_rollout_ref.rollout.max_num_batched_tokens=24576 \
    ++actor_rollout_ref.rollout.enforce_eager=True \
    ++actor_rollout_ref.rollout.temperature=1.0 \
    ++actor_rollout_ref.rollout.top_p=1.0 \
    ++actor_rollout_ref.rollout.top_k=-1 \
    ++actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    ++actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    ++actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    ++actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    ++actor_rollout_ref.rollout.val_kwargs.n=16 \
    ++actor_rollout_ref.rollout.calculate_log_probs=True \
    ++actor_rollout_ref.ref.fsdp_config.param_offload=True \
    ++actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    reward_model.reward_manager=dapo \
    custom_reward_function.path=${VERL_ROOT}/custom_reward_functions/reward_fn.py \
    custom_reward_function.name=compute_score \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
    +reward_model.reward_kwargs.max_resp_len=$((1024+8192)) \
    trainer.logger='["console","swanlab","file"]' \
    trainer.project_name="deepseek1.5b_lr" \
    trainer.experiment_name="deepseek1.5b_sync_8gpu_sigfrac_cfixed_lr1.25e-5_seed0" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=300 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${RESUME_MODE}" \
    ${RESUME_FROM_PATH:+trainer.resume_from_path="${RESUME_FROM_PATH}"} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2} \
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-2} \
    trainer.log_val_generations=10 \
    2>&1 | tee -a "$LOG_FILE"

  echo "Training completed successfully"
  $RAY stop || true
  exit 0
else
  $RAY start --address="$HEAD_IP:6379" --num-cpus=$(nproc)
  echo "Worker joined cluster"
  while $RAY status >/dev/null 2>&1; do sleep 10; done
  echo "Ray cluster stopped, worker exiting"
  exit 0
fi
