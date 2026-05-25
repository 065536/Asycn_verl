#!/bin/bash
set -xeuo pipefail

# Coordinate-descent LR schedule test:
#   LR: 1e-5 -> 3.1e-6
#   Scheduler: linear / cosine
#   Validation: AIME2024 only, evaluate once at step 300
#
# Required env:
#   DECAY_TYPE=linear|cosine
#   SEED=0|1|42
#
# Optional env:
#   BASE_LR, MIN_LR, NCPUS, RESUME_MODE, CKPTS_DIR
#   NNODES, NGPUS_PER_NODE
#   HEAD_IP, MASTER_ADDR, NODE_0_IP
#   RAY_PORT, ENABLE_RAY_DASHBOARD, DASHBOARD_PORT

: "${DECAY_TYPE:?'Set DECAY_TYPE=linear|cosine'}"
: "${SEED:?'Set SEED, e.g. 0/1/42'}"

if [[ "${DECAY_TYPE}" != "linear" && "${DECAY_TYPE}" != "cosine" ]]; then
  echo "DECAY_TYPE must be linear or cosine, got: ${DECAY_TYPE}"
  exit 1
fi

BASE_LR=${BASE_LR:-"1e-5"}
MIN_LR=${MIN_LR:-"3.1e-6"}
RESUME_MODE=${RESUME_MODE:-"disable"}
NCPUS=${NCPUS:-$(nproc)}

RAY_PORT=${RAY_PORT:-6380}
ENABLE_RAY_DASHBOARD=${ENABLE_RAY_DASHBOARD:-0}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
CLEAN_RAY_PROCESSES=${CLEAN_RAY_PROCESSES:-1}
WAIT_HEAD_TIMEOUT=${WAIT_HEAD_TIMEOUT:-300}
WAIT_NODES_TIMEOUT=${WAIT_NODES_TIMEOUT:-300}

# Important:
# VERL_REPO is the repository root.
# VERL_PKG is the Python package directory.
VERL_REPO=${VERL_REPO:-/data/250010176/codes/verl}
VERL_PKG=${VERL_PKG:-${VERL_REPO}/verl}

DATA_ROOT=/data/250010176/data
MODEL_PATH=/data/250010176/codes/models/DeepSeek-R1-Distill-Qwen-1.5B
DEEPSCALER_DIR=/data/250010176/dataset/deepscaler
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MIN_LR_RATIO=$(python3 - <<PY
base_lr = float("${BASE_LR}")
min_lr = float("${MIN_LR}")
print(f"{min_lr / base_lr:.10f}")
PY
)

EXP_NAME="deepseek1.5b_sync_8gpu_cd_${DECAY_TYPE}_1e-5_to_3.1e-6_seed${SEED}"
CKPT_PREFIX="DeepSeek1.5B-Sync-8gpu-cd-${DECAY_TYPE}-1e-5-to-3.1e-6-seed${SEED}"

# Keep logs/checkpoints under the original verl package directory style.
CKPTS_DIR=${CKPTS_DIR:-${VERL_PKG}/ckpts/DeepSeek1.5B/${CKPT_PREFIX}-${TIMESTAMP}}
LOG_DIR=${VERL_PKG}/logs
LOG_FILE="${LOG_DIR}/${EXP_NAME}_${TIMESTAMP}.log"
mkdir -p "${CKPTS_DIR}" "${LOG_DIR}"

source /data/250010176/yrh/miniconda3/etc/profile.d/conda.sh
conda activate verl2

cd "${VERL_REPO}"
export PYTHONPATH="${VERL_REPO}:${PYTHONPATH:-}"

echo "[DEBUG] VERL_REPO=${VERL_REPO}" | tee -a "${LOG_FILE}"
echo "[DEBUG] VERL_PKG=${VERL_PKG}" | tee -a "${LOG_FILE}"
echo "[DEBUG] PYTHONPATH=${PYTHONPATH}" | tee -a "${LOG_FILE}"

python3 - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import sys
import verl
print("[DEBUG] Python executable:", sys.executable)
print("[DEBUG] verl package:", verl.__file__)
PY

CUSTOM_REWARD_PATH="${VERL_PKG}/custom_reward_functions/reward_fn.py"
if [ ! -f "${CUSTOM_REWARD_PATH}" ]; then
  echo "[ERROR] custom reward function not found: ${CUSTOM_REWARD_PATH}" | tee -a "${LOG_FILE}"
  echo "[DEBUG] Listing possible reward files:" | tee -a "${LOG_FILE}"
  find "${VERL_REPO}" -path '*reward_fn.py' -o -path '*custom_reward*' 2>/dev/null | head -50 | tee -a "${LOG_FILE}" || true
  exit 1
fi

export TRAIN_FILE=${DATA_ROOT}/dapo-math-17k-sampled.parquet
export TEST_FILES="[${DEEPSCALER_DIR}/aime24.parquet]"
export MODEL_PATH="${MODEL_PATH}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=1

RAY=/data/250010176/yrh/miniconda3/envs/verl2/bin/ray

# ============================================================
# Auto-detect cluster information
# ============================================================

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

detect_ngpus_per_node() {
  if [ -n "${NGPUS_PER_NODE:-}" ]; then
    echo "${NGPUS_PER_NODE}"
  elif [ -n "${GPUS_PER_NODE:-}" ]; then
    echo "${GPUS_PER_NODE}"
  elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    echo "${SLURM_GPUS_ON_NODE}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l
  else
    echo 8
  fi
}

detect_nnodes() {
  if [ -n "${NNODES:-}" ]; then
    echo "${NNODES}"
  elif [ -n "${SENSECORE_NNODES:-}" ]; then
    echo "${SENSECORE_NNODES}"
  elif [ -n "${SLURM_NNODES:-}" ]; then
    echo "${SLURM_NNODES}"
  elif [ -n "${SLURM_JOB_NUM_NODES:-}" ]; then
    echo "${SLURM_JOB_NUM_NODES}"
  elif [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
    sort -u "${PBS_NODEFILE}" | wc -l
  elif [ -n "${LSB_HOSTS:-}" ]; then
    echo "${LSB_HOSTS}" | tr ' ' '\n' | sort -u | wc -l
  elif [ -n "${HOSTFILE:-}" ] && [ -f "${HOSTFILE}" ]; then
    sort -u "${HOSTFILE}" | wc -l
  elif is_positive_int "${WORLD_SIZE:-}" && is_positive_int "${NGPUS_PER_NODE:-}"; then
    if [ "${WORLD_SIZE}" -ge "${NGPUS_PER_NODE}" ] && [ $((WORLD_SIZE % NGPUS_PER_NODE)) -eq 0 ]; then
      echo $((WORLD_SIZE / NGPUS_PER_NODE))
    else
      echo "${WORLD_SIZE}"
    fi
  else
    echo 1
  fi
}

detect_node_rank() {
  if [ -n "${SENSECORE_PYTORCH_NODE_RANK:-}" ]; then
    echo "${SENSECORE_PYTORCH_NODE_RANK}"
  elif [ -n "${NODE_RANK:-}" ]; then
    echo "${NODE_RANK}"
  elif [ -n "${SLURM_NODEID:-}" ]; then
    echo "${SLURM_NODEID}"
  elif [ -n "${GROUP_RANK:-}" ]; then
    echo "${GROUP_RANK}"
  elif [ -n "${RANK:-}" ]; then
    echo "${RANK}"
  elif [ -n "${OMPI_COMM_WORLD_RANK:-}" ]; then
    echo "${OMPI_COMM_WORLD_RANK}"
  elif [ -n "${PMI_RANK:-}" ]; then
    echo "${PMI_RANK}"
  else
    echo 0
  fi
}

detect_local_rank() {
  if [ -n "${LOCAL_RANK:-}" ]; then
    echo "${LOCAL_RANK}"
  elif [ -n "${SLURM_LOCALID:-}" ]; then
    echo "${SLURM_LOCALID}"
  elif [ -n "${OMPI_COMM_WORLD_LOCAL_RANK:-}" ]; then
    echo "${OMPI_COMM_WORLD_LOCAL_RANK}"
  else
    echo 0
  fi
}

get_local_ip() {
  hostname -I | awk '{print $1}'
}

resolve_host_to_ip() {
  local host="$1"
  local resolved=""
  resolved="$(getent hosts "${host}" 2>/dev/null | awk '{print $1}' | head -1 || true)"
  if [ -n "${resolved}" ]; then
    echo "${resolved}"
  else
    echo "${host}"
  fi
}

first_host_from_scheduler() {
  if [ -n "${HEAD_IP:-}" ]; then
    echo "${HEAD_IP}"
    return 0
  fi

  if [ -n "${MASTER_ADDR:-}" ]; then
    echo "${MASTER_ADDR}"
    return 0
  fi

  if [ -n "${NODE_0_IP:-}" ]; then
    echo "${NODE_0_IP}"
    return 0
  fi

  if [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1
    return 0
  fi

  if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
    head -1 "${PBS_NODEFILE}"
    return 0
  fi

  if [ -n "${LSB_HOSTS:-}" ]; then
    echo "${LSB_HOSTS}" | awk '{print $1}'
    return 0
  fi

  if [ -n "${HOSTFILE:-}" ] && [ -f "${HOSTFILE}" ]; then
    head -1 "${HOSTFILE}"
    return 0
  fi

  return 1
}

NGPUS_PER_NODE="$(detect_ngpus_per_node)"
NNODES="$(detect_nnodes)"
NODE_RANK_CAND="$(detect_node_rank)"
LOCAL_RANK_CAND="$(detect_local_rank)"
LOCAL_IP="$(get_local_ip)"

export NNODES
export NGPUS_PER_NODE
export CKPTS_DIR="${CKPTS_DIR}"

# If the launcher starts one process per GPU, only local rank 0 should manage Ray.
# For one-process-per-node launchers, LOCAL_RANK is usually unset or 0.
if [ "${LOCAL_RANK_CAND}" != "0" ]; then
  echo "[DEBUG] LOCAL_RANK=${LOCAL_RANK_CAND}; skip Ray management on this process." | tee -a "${LOG_FILE}"
  exit 0
fi

if [ "${NNODES}" = "1" ]; then
  IS_HEAD=1
  NODE_RANK_CAND=0
  HEAD_IP="${LOCAL_IP}"
else
  if [ "${NODE_RANK_CAND}" = "0" ]; then
    IS_HEAD=1
  else
    IS_HEAD=0
  fi

  HEAD_HOST="$(first_host_from_scheduler || true)"

  if [ -z "${HEAD_HOST:-}" ]; then
    if [ "${IS_HEAD}" = "1" ]; then
      HEAD_IP="${LOCAL_IP}"
    else
      echo "[ERROR] Cannot infer head node address for multi-node worker." | tee -a "${LOG_FILE}"
      echo "[ERROR] Need one of: HEAD_IP, MASTER_ADDR, NODE_0_IP, SLURM_JOB_NODELIST, PBS_NODEFILE, LSB_HOSTS, HOSTFILE." | tee -a "${LOG_FILE}"
      env | sort | egrep 'MASTER|HEAD|NODE|RANK|SLURM|PBS|LSB|SENSE|WORLD|HOST|LOCAL' | tee -a "${LOG_FILE}" || true
      exit 1
    fi
  else
    HEAD_IP="$(resolve_host_to_ip "${HEAD_HOST}")"
  fi
fi

# Head node must bind to an IP available on local interfaces.
if [ "${IS_HEAD}" = "1" ] && command -v ip >/dev/null 2>&1; then
  if ! ip -o -4 addr show | awk '{print $4}' | cut -d/ -f1 | grep -qx "${HEAD_IP}"; then
    echo "[WARN] HEAD_IP=${HEAD_IP} is not bound on local interfaces; fallback to LOCAL_IP=${LOCAL_IP}" | tee -a "${LOG_FILE}"
    HEAD_IP="${LOCAL_IP}"
  fi
fi

RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
export RAY_ADDRESS

echo "[DEBUG] hostname=$(hostname)" | tee -a "${LOG_FILE}"
echo "[DEBUG] hostname -I=$(hostname -I)" | tee -a "${LOG_FILE}"
echo "[DEBUG] LOCAL_IP=${LOCAL_IP}" | tee -a "${LOG_FILE}"
echo "[DEBUG] HEAD_IP=${HEAD_IP}" | tee -a "${LOG_FILE}"
echo "[DEBUG] RAY_ADDRESS=${RAY_ADDRESS}" | tee -a "${LOG_FILE}"
echo "[DEBUG] IS_HEAD=${IS_HEAD}" | tee -a "${LOG_FILE}"
echo "[DEBUG] NODE_RANK=${NODE_RANK_CAND}" | tee -a "${LOG_FILE}"
echo "[DEBUG] LOCAL_RANK=${LOCAL_RANK_CAND}" | tee -a "${LOG_FILE}"
echo "[DEBUG] RANK=${RANK:-unset}" | tee -a "${LOG_FILE}"
echo "[DEBUG] NNODES=${NNODES}" | tee -a "${LOG_FILE}"
echo "[DEBUG] NGPUS_PER_NODE=${NGPUS_PER_NODE}" | tee -a "${LOG_FILE}"
echo "[DEBUG] scheduler env snapshot:" | tee -a "${LOG_FILE}"
env | sort | egrep 'MASTER|HEAD|NODE|RANK|SLURM|PBS|LSB|SENSE|WORLD|HOST|LOCAL' | tee -a "${LOG_FILE}" || true

# Keep this short. Ray creates Unix sockets under this path.
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/r${SEED}_${NODE_RANK_CAND}_$$}

dump_ray_logs() {
  echo "========== Ray logs =========="
  echo "[DEBUG] RAY_TMPDIR=${RAY_TMPDIR}"

  for d in "${RAY_TMPDIR}" /tmp/ray; do
    if [ -d "${d}" ]; then
      echo "[DEBUG] Listing ${d}"
      find "${d}" -maxdepth 4 -type f 2>/dev/null | head -100 || true

      echo "[DEBUG] gcs_server logs from ${d}"
      tail -n 200 "${d}"/session_latest/logs/gcs_server* 2>/dev/null || true

      echo "[DEBUG] raylet logs from ${d}"
      tail -n 200 "${d}"/session_latest/logs/raylet* 2>/dev/null || true

      echo "[DEBUG] dashboard logs from ${d}"
      tail -n 200 "${d}"/session_latest/logs/dashboard* 2>/dev/null || true
    fi
  done

  echo "========== End Ray logs =========="
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"

  python3 - "${host}" "${port}" "${timeout_s}" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout_s = int(sys.argv[3])

start = time.time()
last_err = None

while time.time() - start < timeout_s:
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"[DEBUG] Port ready: {host}:{port}", flush=True)
            sys.exit(0)
    except Exception as e:
        last_err = e
        time.sleep(2)

print(f"[ERROR] Timeout waiting for {host}:{port}. Last error: {last_err}", flush=True)
sys.exit(1)
PY
}

wait_for_ray_nodes() {
  local target_nodes="$1"
  local timeout_s="$2"

  python3 - "${RAY_ADDRESS}" "${target_nodes}" "${timeout_s}" <<'PY'
import sys
import time
import ray

address = sys.argv[1]
target_nodes = int(sys.argv[2])
timeout_s = int(sys.argv[3])

ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")

start = time.time()
while time.time() - start < timeout_s:
    alive_nodes = [n for n in ray.nodes() if n.get("Alive")]
    print(f"[DEBUG] Alive Ray nodes: {len(alive_nodes)}/{target_nodes}", flush=True)

    if len(alive_nodes) >= target_nodes:
        ray.shutdown()
        sys.exit(0)

    time.sleep(5)

ray.shutdown()
print(f"[ERROR] Timeout waiting for Ray nodes: target={target_nodes}", flush=True)
sys.exit(1)
PY
}

# ============================================================
# Clean old Ray state
# ============================================================

"${RAY}" stop -f || true

if [ "${CLEAN_RAY_PROCESSES}" = "1" ]; then
  if command -v pkill >/dev/null 2>&1; then
    pkill -u "$(id -u)" -f "gcs_server|raylet|dashboard|redis-server" || true
  fi
fi

rm -rf "${RAY_TMPDIR}" || true
mkdir -p "${RAY_TMPDIR}"

# ============================================================
# Start Ray cluster
# ============================================================

if [ "${IS_HEAD}" = "1" ]; then
  echo "[DEBUG] Starting Ray head" | tee -a "${LOG_FILE}"

  RAY_HEAD_CMD=(
    "${RAY}" start
    --head
    --node-ip-address="${HEAD_IP}"
    --port="${RAY_PORT}"
    --num-cpus="${NCPUS}"
    --temp-dir="${RAY_TMPDIR}"
    --disable-usage-stats
  )

  if [ "${ENABLE_RAY_DASHBOARD}" = "1" ]; then
    RAY_HEAD_CMD+=(--dashboard-port="${DASHBOARD_PORT}")
  else
    RAY_HEAD_CMD+=(--include-dashboard=false)
  fi

  if ! timeout 120s "${RAY_HEAD_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "[ERROR] Ray head failed to start or timed out" | tee -a "${LOG_FILE}"
    dump_ray_logs 2>&1 | tee -a "${LOG_FILE}"
    exit 1
  fi

  if ! wait_for_port "${HEAD_IP}" "${RAY_PORT}" 60 2>&1 | tee -a "${LOG_FILE}"; then
    echo "[ERROR] Ray head port is not reachable after start" | tee -a "${LOG_FILE}"
    dump_ray_logs 2>&1 | tee -a "${LOG_FILE}"
    exit 1
  fi

  if ! "${RAY}" status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
    echo "[ERROR] ray status failed after Ray head start" | tee -a "${LOG_FILE}"
    dump_ray_logs 2>&1 | tee -a "${LOG_FILE}"
    exit 1
  fi

  if [ "${NNODES}" != "1" ]; then
    echo "[DEBUG] Waiting for ${NNODES} Ray nodes to join" | tee -a "${LOG_FILE}"

    if ! wait_for_ray_nodes "${NNODES}" "${WAIT_NODES_TIMEOUT}" 2>&1 | tee -a "${LOG_FILE}"; then
      echo "[ERROR] Not all Ray nodes joined before timeout" | tee -a "${LOG_FILE}"
      dump_ray_logs 2>&1 | tee -a "${LOG_FILE}"
      exit 1
    fi
  fi

  echo "[DEBUG] Ray head is ready: ${RAY_ADDRESS}" | tee -a "${LOG_FILE}"

  python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILES}" \
    data.prompt_key=prompt \
    data.seed="${SEED}" \
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
    ++actor_rollout_ref.actor.optim.lr="${BASE_LR}" \
    ++actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    ++actor_rollout_ref.actor.optim.weight_decay=0.1 \
    ++actor_rollout_ref.actor.optim.lr_scheduler_type="${DECAY_TYPE}" \
    ++actor_rollout_ref.actor.optim.min_lr_ratio="${MIN_LR_RATIO}" \
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
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
    +reward_model.reward_kwargs.max_resp_len=$((1024+8192)) \
    trainer.logger='["console","swanlab","file"]' \
    trainer.project_name="deepseek1.5b_lr_cd" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=300 \
    trainer.save_freq=50 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=300 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.log_val_generations=0 \
    2>&1 | tee -a "${LOG_FILE}"

  echo "Training completed successfully" | tee -a "${LOG_FILE}"
  "${RAY}" stop || true
  exit 0

else
  echo "[DEBUG] Worker waiting for Ray head: ${RAY_ADDRESS}" | tee -a "${LOG_FILE}"

  if ! wait_for_port "${HEAD_IP}" "${RAY_PORT}" "${WAIT_HEAD_TIMEOUT}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "[ERROR] Worker cannot reach Ray head: ${RAY_ADDRESS}" | tee -a "${LOG_FILE}"
    dump_ray_logs 2>&1 | tee -a "${LOG_FILE}"
    exit 1
  fi

  echo "[DEBUG] Starting Ray worker, connecting to ${RAY_ADDRESS}" | tee -a "${LOG_FILE}"

  if ! timeout 120s "${RAY}" start \
    --address="${RAY_ADDRESS}" \
    --num-cpus="${NCPUS}" \
    --disable-usage-stats \
    2>&1 | tee -a "${LOG_FILE}"; then

    echo "[ERROR] Ray worker failed to join cluster" | tee -a "${LOG_FILE}"
    dump_ray_logs 2>&1 | tee -a "${LOG_FILE}"
    exit 1
  fi

  echo "Worker joined cluster" | tee -a "${LOG_FILE}"

  while "${RAY}" status --address="${RAY_ADDRESS}" >/dev/null 2>&1; do
    sleep 10
  done

  echo "Ray cluster stopped, worker exiting" | tee -a "${LOG_FILE}"
  exit 0
fi