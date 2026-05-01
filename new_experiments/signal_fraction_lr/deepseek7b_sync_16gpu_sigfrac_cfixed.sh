#!/bin/bash
set -xeuo pipefail

# DeepSeek-7B sync Phase 1: c_fixed sweep for signal-fraction LR.
#
# Goal:
# - Recalibrate the signal-fraction scale on 7B before porting the 1.5B conclusions.
# - Keep c-side disabled (eta_c=0) and no held-out calibration split.
# - Sweep BASE_LR, which sets c_fixed at handoff via c_T ~= BASE_LR / r_boot.
#
# Suggested first sweep:
# - BASE_LR=5e-6     -> c_fixed ~= 1.0e-4
# - BASE_LR=7.5e-6   -> c_fixed ~= 1.5e-4
# - BASE_LR=1e-5     -> c_fixed ~= 2.0e-4

VERL_ROOT=/data/250010176/codes/verl/verl
DATA_ROOT=/data/250010176/data
MODEL_PATH=/data/250010176/codes/models/DeepSeek-R1-Distill-Qwen-7B
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

SEED=${SEED:-42}
BASE_LR=${BASE_LR:-7.5e-6}
LR_TAG=${LR_TAG:-${BASE_LR}}

# Resume is auto by default because 7B long-response runs may need multiple
# submissions to reach 300 steps under per-job GPU-hour caps.
RESUME_MODE=${RESUME_MODE:-"auto"}  # Options: "auto", "resume_path", "disable"
RESUME_FROM_PATH=${RESUME_FROM_PATH:-""}
USER_CKPTS_DIR=${CKPTS_DIR:-""}

MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-2}

# Resource allocation: default 16 GPUs = 2 nodes * 8 GPUs. Override NNODES=4
# for a 32-GPU run when the scheduler budget allows shorter wall-clock jobs.
export NNODES=${NNODES:-2}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TOTAL_GPUS=$((NNODES * NGPUS_PER_NODE))
GPU_TAG=${GPU_TAG:-${TOTAL_GPUS}gpu}

EXP_NAME_BASE=deepseek7b_sync_${GPU_TAG}_sigfrac_cfixed_lr${LR_TAG}
EXP_NAME=${EXP_NAME:-"${EXP_NAME_BASE}_seed${SEED}"}
CKPT_PREFIX=DeepSeek7B-Sync-${GPU_TAG}-sigfrac-cfixed-lr${LR_TAG}-seed${SEED}

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
  LATEST_CKPT=$(find ${VERL_ROOT}/ckpts/DeepSeek7B -name "${CKPT_PREFIX}-*" -type d 2>/dev/null | sort | tail -1)
  if [ -n "$LATEST_CKPT" ]; then
    CKPTS_DIR="$LATEST_CKPT"
    TIMESTAMP=$(basename "$CKPTS_DIR" | sed 's/.*-//')
    echo "Auto-found existing checkpoint directory: $CKPTS_DIR"
  else
    CKPTS_DIR=${VERL_ROOT}/ckpts/DeepSeek7B/${CKPT_PREFIX}-${TIMESTAMP}
    echo "No existing checkpoint found, creating new directory: $CKPTS_DIR"
  fi
else
  CKPTS_DIR=${VERL_ROOT}/ckpts/DeepSeek7B/${CKPT_PREFIX}-${TIMESTAMP}
  echo "Resume disabled, creating new checkpoint directory: $CKPTS_DIR"
fi

LOG_DIR=${VERL_ROOT}/logs
LOG_FILE="${LOG_DIR}/deepseek7b_sync_${GPU_TAG}_sigfrac_cfixed_lr${LR_TAG}_seed${SEED}_${TIMESTAMP}.log"
LOCAL_TRACKING_DIR="${LOG_DIR}/local_tracking"
mkdir -p "$CKPTS_DIR" "$LOG_DIR" "$LOCAL_TRACKING_DIR"
export VERL_FILE_LOGGER_ROOT="$LOCAL_TRACKING_DIR"

if [ -n "${MASTER_ADDR:-}" ]; then HN="$MASTER_ADDR"; elif [ -n "${NODE_0_IP:-}" ]; then HN="$NODE_0_IP"; else HN="$(hostname -I | awk '{print $1}')"; fi
HEAD_IP="$(getent hosts "$HN" | awk '{print $1}' | head -1)"
[ -z "$HEAD_IP" ] && HEAD_IP="$HN"

source /data/250010176/yrh/miniconda3/etc/profile.d/conda.sh
conda activate verl2
cd "${VERL_ROOT}/.."
export PYTHONPATH="${VERL_ROOT}/..:${PYTHONPATH:-}"

# Data: 17k-sampled train, DeepScaler test sets.
DEEPSCALER_DIR=/data/250010176/dataset/deepscaler
export TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/dapo-math-17k-sampled.parquet}
export TEST_FILES=${TEST_FILES:-"[${DEEPSCALER_DIR}/aime24.parquet,${DEEPSCALER_DIR}/aime25.parquet,${DEEPSCALER_DIR}/gpqa_diamond_100.parquet,${DEEPSCALER_DIR}/minerva_fixed_100.parquet,${DEEPSCALER_DIR}/olympiad_bench_fixed_100.parquet]"}
export MODEL_PATH="$MODEL_PATH"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=1

export CKPTS_DIR="$CKPTS_DIR"

# 7B sync parameters. Use 10k response length for the first c_fixed sweep; 16k
# made each step too expensive for scale search.
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-10240}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-2048}
train_batch_size=${TRAIN_BATCH_SIZE:-256}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
val_n_resp_per_prompt=${VAL_N_RESP_PER_PROMPT:-8}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
ppo_max_token_per_gpu=${PPO_MAX_TOKEN_PER_GPU:-$((max_prompt_length + max_response_length))}
total_training_steps=${TOTAL_TRAINING_STEPS:-300}
test_freq=${TEST_FREQ:-30}
save_freq=${SAVE_FREQ:-10}
val_before_train=${VAL_BEFORE_TRAIN:-True}
gen_tp=${GEN_TP:-4}
sp_size=${SP_SIZE:-1}

IS_HEAD=0
NODE_RANK_CAND="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-}}"
if [ "${NODE_RANK_CAND:-}" = "0" ] || [ "${RANK:-}" = "0" ]; then IS_HEAD=1; fi
echo "HEAD_IP=$HEAD_IP IS_HEAD=$IS_HEAD RANK=${RANK:-unset} NODE_RANK=${NODE_RANK_CAND:-unset}"
echo "DeepSeek7B sync cfixed: BASE_LR=${BASE_LR}, LR_TAG=${LR_TAG}, SEED=${SEED}"
echo "Total GPUs: ${NNODES} nodes * ${NGPUS_PER_NODE} GPUs/node = $((NNODES * NGPUS_PER_NODE))"
echo "Validation: VAL_N_RESP_PER_PROMPT=${val_n_resp_per_prompt}, TEST_FREQ=${test_freq}, VAL_BEFORE_TRAIN=${val_before_train}"
echo "Checkpointing: SAVE_FREQ=${save_freq}"

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
    data.seed=${SEED} \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.filter_overlong_prompts=True \
    ++actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    ++actor_rollout_ref.actor.rollout_n=${n_resp_per_prompt} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.rollout_correction.bypass_mode=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    ++actor_rollout_ref.actor.clip_ratio_low=0.2 \
    ++actor_rollout_ref.actor.clip_ratio_high=0.28 \
    ++actor_rollout_ref.actor.clip_ratio_c=10.0 \
    ++actor_rollout_ref.model.use_remove_padding=True \
    ++actor_rollout_ref.hybrid_engine=True \
    ++actor_rollout_ref.actor.use_dynamic_bsz=True \
    ++actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    ++actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    ++actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_per_gpu} \
    ++actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_per_gpu} \
    ++actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_per_gpu} \
    ++actor_rollout_ref.rollout.name=vllm \
    ++actor_rollout_ref.model.path="${MODEL_PATH}" \
    ++actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.model.use_liger=True \
    ++actor_rollout_ref.actor.optim.lr=${BASE_LR} \
    ++actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    ++actor_rollout_ref.actor.optim.weight_decay=0.1 \
    ++actor_rollout_ref.actor.optim.lr_scheduler_type=signal_fraction \
    +actor_rollout_ref.actor.optim.signal_fraction_eta_c=0.0 \
    +actor_rollout_ref.actor.optim.signal_fraction_c_min=1e-8 \
    +actor_rollout_ref.actor.optim.signal_fraction_c_max=1e-2 \
    +actor_rollout_ref.actor.optim.signal_fraction_calib_freq=5 \
    +actor_rollout_ref.actor.optim.signal_fraction_phi_ema_beta=0.9 \
    +actor_rollout_ref.actor.optim.signal_fraction_r_min=0.01 \
    +actor_rollout_ref.actor.optim.signal_fraction_calib_frac=0.0 \
    ++actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    ++actor_rollout_ref.actor.ppo_micro_batch_size=null \
    +actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    ++actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.actor.fsdp_config.param_offload=True \
    ++actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    ++actor_rollout_ref.actor.entropy_coeff=0 \
    ++actor_rollout_ref.actor.grad_clip=1.0 \
    ++actor_rollout_ref.actor.loss_agg_mode=token-mean \
    ++actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    ++actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    ++actor_rollout_ref.rollout.dtype=bfloat16 \
    ++actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    ++actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    ++actor_rollout_ref.rollout.enable_chunked_prefill=True \
    ++actor_rollout_ref.rollout.max_num_batched_tokens=${ppo_max_token_per_gpu} \
    ++actor_rollout_ref.rollout.free_cache_engine=True \
    ++actor_rollout_ref.rollout.enforce_eager=False \
    ++actor_rollout_ref.rollout.temperature=1.0 \
    ++actor_rollout_ref.rollout.top_p=1.0 \
    ++actor_rollout_ref.rollout.top_k=-1 \
    ++actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    ++actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    ++actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    ++actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    ++actor_rollout_ref.rollout.val_kwargs.n=${val_n_resp_per_prompt} \
    ++actor_rollout_ref.rollout.calculate_log_probs=True \
    ++actor_rollout_ref.ref.fsdp_config.dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.param_offload=True \
    ++actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    reward_model.reward_manager=dapo \
    custom_reward_function.path=${VERL_ROOT}/custom_reward_functions/reward_fn.py \
    custom_reward_function.name=compute_score \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","swanlab","file"]' \
    trainer.project_name="DeepSeek7B" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=10 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${RESUME_MODE}" \
    ${RESUME_FROM_PATH:+trainer.resume_from_path="${RESUME_FROM_PATH}"} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP} \
    trainer.log_val_generations=10 \
    trainer.balance_batch=False \
    2>&1 | tee -a "$LOG_FILE"

  echo "Training completed successfully"
  $RAY stop || true
  exit 0
else
  ready=0
  for i in $(seq 1 60); do
    if $RAY status --address="$HEAD_IP:6379" >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    echo "Ray head at $HEAD_IP:6379 was not ready for worker join"
    exit 1
  fi
  $RAY start --address="$HEAD_IP:6379" --num-cpus=$(nproc)
  echo "Worker joined cluster"
  while $RAY status >/dev/null 2>&1; do sleep 10; done
  echo "Ray cluster stopped, worker exiting"
  exit 0
fi
