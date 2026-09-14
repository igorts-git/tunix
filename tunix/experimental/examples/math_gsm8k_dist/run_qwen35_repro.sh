#!/usr/bin/env bash
# Qwen3.5-35B-A3B distributed GRPO on TPU v5p -- reproduction launcher.
#
#   ./run_qwen35_repro.sh start     # launch orchestrator + trainer + 8 rollouts
#   ./run_qwen35_repro.sh stop      # tear everything down
#   DRY_RUN=true ./run_qwen35_repro.sh start   # print manifests, apply nothing
#
# Overridable knobs (all have defaults below):
#   MAX_STEPS, BATCH_SIZE, NUM_GENERATIONS, RUN_USER, RUN_TAG, TUNIX_IMAGE, DEBUG
#
# See qwen35_report_v6.md for the full rationale behind every value here.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Identity
# ---------------------------------------------------------------------------
# k8s object names must be RFC-1123 labels, so no underscores. On corp GCE VMs
# $USER is e.g. "igorts_google_com", which produces jobset names the API server
# rejects. Everything downstream derives its job names from $USER, so sanitize
# it here rather than in a dozen places.
export USER="${RUN_USER:-$(echo "${USER:-$(whoami)}" | cut -d_ -f1 | tr -cd 'a-z0-9-')}"
if [[ -z "${USER}" || ! "${USER}" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]]; then
  echo "RUN_USER must be a valid k8s name component; got '${USER}'." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Cluster
# ---------------------------------------------------------------------------
export PROJECT="cloud-tpu-shared-capacity"
export CLUSTER="bodaborg-v5p-nap"
export LOCATION_NAME="europe-west4"
export REGION="europe-west4"
export ZONE="europe-west4-a"
export K8S_NAMESPACE="trellis"
export NAMESPACE="trellis"
export QUEUE_NAME="default"
export KUEUE_QUEUE_NAME="default"
export CPU_MACHINE="n2d-standard-64"
# Kueue WorkloadPriorityClass. Without it the workload sits at priority 0 and
# anything else in the tpu-shared-cohort evicts it; the previous 100-step
# attempt died at step 43 that way. medium == 500.
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"

# ---------------------------------------------------------------------------
# 2. Images
# ---------------------------------------------------------------------------
# Yixuan's e2e image plus a 4-file overlay; build with ../../../../build_qwen35_overlay.sh
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/igorts_google_com-runner:qwen35-repro-v7}"
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904"
export PATHWAYS_PROXY_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904"
# The whole model is staged on the proxy host during Raiden D2H sync; the
# yaml generator's 100G default OOM-kills it for a 35B model.
export PATHWAYS_PROXY_MEMORY_LIMIT="250G"

# ---------------------------------------------------------------------------
# 3. Model and base checkpoint
# ---------------------------------------------------------------------------
export TRAINER_BACKEND="maxtext"
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
# Must match the MaxText pydantic literal, which is lowercase and dotted.
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_CKPT="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items"
export MAXTEXT_OUTPUT_DIR="gs://yixuannwang-maxtext-dataset/${USER}/qwen35_repro"

# ---------------------------------------------------------------------------
# 4. Checkpointing: OFF
# ---------------------------------------------------------------------------
# 0 means "never save" on both sides: the orchestrator stops issuing save
# requests, and maxtext_utils keeps enable_checkpointing=True (needed to
# *restore* the base weights) while pushing checkpoint_period out to 1e9.
# Saving is being fixed separately; leaving it on here costs a 64 GiB write.
export CHECKPOINT_SAVE_INTERVAL_STEPS=0
export ENABLE_PATHWAYS_PERSISTENCE=0

# ---------------------------------------------------------------------------
# 5. Trainer topology: one Pathways 2x2x2 slice, 8 chips, pure FSDP
# ---------------------------------------------------------------------------
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="tpuv5p:2x2x2"
export TRAINER_MESH_FSDP=8
export TRAINER_MESH_TP=1
export TRAINER_MESH_EXPERT=1

# ---------------------------------------------------------------------------
# 6. Rollout topology: 8 independent 2x2x1 slices, 32 chips, pure TP
# ---------------------------------------------------------------------------
# Each slice is 4 chips, filled as TP=2 x DP=2. ROLLOUT_MESH_FSDP is a misnomer
# on the rollout side: nothing shards weights FSDP-style there, the value is
# handed to vLLM as data_parallel_size, i.e. whole engine replicas.
#
# TP must stay at 2. At TP=4, maxtext_utils replicates base_num_kv_heads 2 -> 4
# (which MaxText then rejects outright, since the model yml also sets it) and
# GMM_v2 pads the MoE MLP dim 512 -> 1024 because 512/4 is not a multiple of
# 2*128. The MoE experts are ~32B of this 35B model, so that padding roughly
# doubles what the trainer has to hold. At TP=2, 512/2 = 256 is already
# aligned and num_kv_heads is untouched.
export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="tpuv5p:2x2x1"
export ROLLOUT_MESH_FSDP=2
export ROLLOUT_MESH_TP=2
export ROLLOUT_REPLICAS=8
export SAMPLER="vllm"

# ---------------------------------------------------------------------------
# 7. Weight synchronization
# ---------------------------------------------------------------------------
export WEIGHT_SYNC_MODE="raiden"
export USE_WEIGHT_CONVERTER=true
export PREFUSE_MOE_WEIGHTS=true
export VERIFY_WEIGHTS=false
# Off: prompts in a GRPO batch share no prefix worth caching, and with it off we
# are free to round robin generations across rollout workers.
export ENABLE_PREFIX_CACHING=false

# ---------------------------------------------------------------------------
# 8. Batch shape
# ---------------------------------------------------------------------------
export MAX_STEPS="${MAX_STEPS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
# One optimizer update per step over all BATCH_SIZE*NUM_GENERATIONS rollouts.
export MINI_BATCH_SIZE=$((BATCH_SIZE * NUM_GENERATIONS))
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=512
# Sequence packing. The packed row count per microbatch is
# trainer_fsdp * trainer_dp = 8, and TRAIN_MICRO_BATCH_SIZE has to match that so
# MaxText gets per_device_batch_size = 8/8 = 1.
export TRAIN_MICRO_BATCH_SIZE=8
# Budget per packed row. It cannot exceed MaxText's max_target_length, which
# maxtext_utils fixes at max_prompt_length + max_response_length -- so 1024 is
# the ceiling, not a tuning choice. Packing still pays: a GSM8K prompt plus
# answer is ~250-350 tokens, so ~3 trajectories share each row and the trainer
# does roughly a third as many forward/backward passes as with padding.
export MAX_SEQ_TOKEN_PER_TPU=1024
export USE_ROLLOUT_LOGPS=true

# ---------------------------------------------------------------------------
# 9. Optimizer
# ---------------------------------------------------------------------------
# Note: USE_LORA is not plumbed through to either node in this image, so this
# is a full-parameter fine-tune. Do not set USE_LORA and expect LoRA.
export LEARNING_RATE=2.0e-7
export ADAM_B1=0.9
export ADAM_B2=0.99
export WEIGHT_DECAY=0.01
export MAX_GRAD_NORM=1.0
export BETA=0
export EPSILON=0.2

# ---------------------------------------------------------------------------
# 10. Metrics
# ---------------------------------------------------------------------------
# .bashrc returns early for non-interactive shells, so sourcing it is not
# enough to pick the key up; read the export line directly.
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.bashrc" ]]; then
  eval "$(grep -m1 '^export WANDB_API_KEY=' "${HOME}/.bashrc" || true)"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "WARNING: WANDB_API_KEY is unset; the run will not log to Weights & Biases." >&2
fi
export WANDB_PROJECT="${WANDB_PROJECT:-qwen35-35b-a3b-grpo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${USER}-${RUN_TAG:-repro}-b${BATCH_SIZE}g${NUM_GENERATIONS}-s${MAX_STEPS}}"
export FLUSH_METRICS_EVERY_N_STEPS=1

export DEBUG="${DEBUG:-0}"
export DRY_RUN="${DRY_RUN:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/k8s_launcher.sh" "${1:-start}"
