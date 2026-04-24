#!/bin/bash
set -xeuo pipefail

VERL_ROOT=/data/250010176/codes/verl/verl
DATA_ROOT=/data/250010176/data
MODEL_PATH=/data/250010176/codes/models/DeepSeek-R1-Distill-Qwen-7B
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Resume training configuration
RESUME_MODE=${RESUME_MODE:-"disable"}  # Options: "auto", "resume_path", "disable"
RESUME_FROM_PATH=${RESUME_FROM_PATH:-""}
USER_CKPTS_DIR=${CKPTS_DIR:-""}
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
    LATEST_CKPT=$(find ${VERL_ROOT}/ckpts/DeepSeek7B -name "DeepSeek7B-FullyAsync-PartialRollout-32gpu-*" -type d 2>/dev/null | sort | tail -1)
  if [ -n "$LATEST_CKPT" ]; then
    CKPTS_DIR="$LATEST_CKPT"
    TIMESTAMP=$(basename "$CKPTS_DIR" | sed 's/.*-//')
    echo "Auto-found existing checkpoint directory: $CKPTS_DIR"
  else
    CKPTS_DIR=${VERL_ROOT}/ckpts/DeepSeek7B/DeepSeek7B-FullyAsync-PartialRollout-32gpu-${TIMESTAMP}
    echo "No existing checkpoint found, creating new directory: $CKPTS_DIR"
  fi
else
    CKPTS_DIR=${VERL_ROOT}/ckpts/DeepSeek7B/DeepSeek7B-FullyAsync-PartialRollout-32gpu-${TIMESTAMP}
    echo "Resume disabled, creating new checkpoint directory: $CKPTS_DIR"
fi

LOG_DIR=${VERL_ROOT}/logs
LOG_FILE="${LOG_DIR}/deepseek7b_fully_async_partial_rollout_32gpu_${TIMESTAMP}.log"
mkdir -p "$CKPTS_DIR" "$LOG_DIR"

if [ -n "${MASTER_ADDR:-}" ]; then HN="$MASTER_ADDR"; elif [ -n "${NODE_0_IP:-}" ]; then HN="$NODE_0_IP"; else HN="$(hostname -I | awk '{print $1}')"; fi
HEAD_IP="$(getent hosts "$HN" | awk '{print $1}' | head -1)"
[ -z "$HEAD_IP" ] && HEAD_IP="$HN"

source /data/250010176/yrh/miniconda3/etc/profile.d/conda.sh
conda activate verl2
cd "${VERL_ROOT}/.."
export PYTHONPATH="${VERL_ROOT}/..:${PYTHONPATH:-}"

# Data: 17k-sampled train, DeepScaler test sets
DEEPSCALER_DIR=/data/250010176/dataset/deepscaler
export TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/dapo-math-17k-sampled.parquet}
export TEST_FILES=${TEST_FILES:-"[${DEEPSCALER_DIR}/aime24.parquet,${DEEPSCALER_DIR}/aime25.parquet,${DEEPSCALER_DIR}/gpqa_diamond_100.parquet,${DEEPSCALER_DIR}/minerva_fixed_100.parquet,${DEEPSCALER_DIR}/olympiad_bench_fixed_100.parquet]"}
export MODEL_PATH="$MODEL_PATH"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=1

# Resource allocation: 32 GPUs total
# 4 nodes x 8 GPUs = 32 GPUs
# Rollout: 4 GPUs per node = 16 GPUs
# Training: 4 GPUs per node = 16 GPUs
export NNODES=4
export NGPUS_PER_NODE=8
export NGPUS_ROLLOUT_PER_NODE=4
export NGPUS_TRAINING_PER_NODE=4
export CKPTS_DIR="$CKPTS_DIR"

# 7B model params
max_prompt_length=1024
max_response_length=16384
n_resp_per_prompt=16
ppo_max_token_per_gpu=$((max_prompt_length + max_response_length))

# Fully async specific parameters (With Partial Rollout)
rollout_mode="async"
rollout_name="vllm"
return_raw_chat="True"
train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=32
test_freq=5
staleness_threshold=4
trigger_parameter_sync_step=1
require_batches=16
required_samples=$((train_prompt_mini_bsz * require_batches))
num_param_sync_steps=300
total_rollout_steps=$((num_param_sync_steps * trigger_parameter_sync_step * required_samples))
partial_rollout=True

IS_HEAD=0
NODE_RANK_CAND="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-}}"
if [ "${NODE_RANK_CAND:-}" = "0" ] || [ "${RANK:-}" = "0" ]; then IS_HEAD=1; fi
echo "HEAD_IP=$HEAD_IP IS_HEAD=$IS_HEAD RANK=${RANK:-unset} NODE_RANK=${NODE_RANK_CAND:-unset}"
echo "Rollout: ${NNODES} nodes * ${NGPUS_ROLLOUT_PER_NODE} GPUs/node = $((NNODES * NGPUS_ROLLOUT_PER_NODE)) GPUs total"
echo "Training: ${NNODES} nodes * ${NGPUS_TRAINING_PER_NODE} GPUs/node = $((NNODES * NGPUS_TRAINING_PER_NODE)) GPUs total"
echo "Partial Rollout: total_rollout_steps=${total_rollout_steps} staleness=${staleness_threshold} trigger=${trigger_parameter_sync_step}"

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
  
  python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILES}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_per_gpu} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ppo_max_token_per_gpu} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    reward_model.reward_manager=dapo \
    custom_reward_function.path=${VERL_ROOT}/custom_reward_functions/reward_fn.py \
    custom_reward_function.name=compute_score \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="DeepSeek7B" \
    trainer.experiment_name="DeepSeek7B-FA-staleness-4" \
    trainer.n_gpus_per_node="${NGPUS_TRAINING_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq} \
    trainer.log_val_generations=10 \
    trainer.save_freq=5 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${RESUME_MODE}" \
    ${RESUME_FROM_PATH:+trainer.resume_from_path="${RESUME_FROM_PATH}"} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2} \
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-2} \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${NGPUS_ROLLOUT_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
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
