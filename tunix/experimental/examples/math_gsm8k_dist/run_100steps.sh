#!/usr/bin/env bash
# Reproduction launcher for Qwen3.5-35B-A3B 100-step distributed RL run.
# Usable directly by any teammate with access to the cloud-tpu-shared-capacity GCP project.

set -euo pipefail

# 1. Cluster & Workload Identification
export USER="${USER:-$(whoami)}"
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

# 2. Container Images
# Build with ../../../../build_repro_image.sh and point TUNIX_IMAGE at the result. No
# default: the image the v4 run used (...-runner:igorts-qwen35-fast-0912) overlays only
# tunix, so it predates the maxtext-side checkpoint drain and would OOMKill the proxy
# now that saving is on.
export TUNIX_IMAGE="${TUNIX_IMAGE:?set TUNIX_IMAGE to an image built by build_repro_image.sh}"
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904"
export PATHWAYS_PROXY_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904"
export PATHWAYS_PROXY_MEMORY_LIMIT="250G"

# 3. Model & Scanned Checkpoint
export TRAINER_BACKEND="maxtext"
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
# CRITICAL: MaxText model name must match Pydantic literal 'qwen3.5-35b-a3b' (not qwen3-35b)
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
# Verified scanned checkpoint in GCS:
export MAXTEXT_CKPT="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items"
# Also where checkpoints are written, and where a restart looks for one to resume from.
export MAXTEXT_OUTPUT_DIR="gs://yixuannwang-maxtext-dataset/${USER}/qwen35_run_100steps"

# 4. Checkpoint Configuration (save and restore both enabled)
# A 100-step run takes >8h and the v4 attempt lost 42 steps to a preemption at step 43
# because saving was off. Saving every 25 steps (~2h) costs ~90s of GCS write, under 1.2%
# of runtime, and a restart resumes from the last saved step automatically.
export DISABLE_CHECKPOINTING="false"
export CHECKPOINT_SAVE_INTERVAL_STEPS=25
export CHECKPOINT_MAX_TO_KEEP=4
# Pathways writes array shards to GCS from the TPU workers instead of streaming ~355 GB of
# model + optimizer state through the proxy, which is what used to OOMKill it.
export ENABLE_PATHWAYS_PERSISTENCE=1

# 5. Trainer Topology (Pathways multi-host 2x2x2 = 8 chips, 1 slice)
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="tpuv5p:2x2x2"
export TRAINER_MESH_FSDP=8
export TRAINER_MESH_TP=1
export TRAINER_MESH_EXPERT=1

# 6. Rollout Topology (8 standalone 2x2x1 slices = 32 chips total)
export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="tpuv5p:2x2x1"
export ROLLOUT_MESH_FSDP=2
export ROLLOUT_MESH_TP=2
export ROLLOUT_REPLICAS=8

# 7. Raiden Weight Synchronization
export WEIGHT_SYNC_MODE="raiden"
# RAIDEN_DEVICES_PER_HOST is read by nothing in tunix -- raiden_synchronizer derives the
# value from the mesh instead. Kept only to match the Google Doc's config verbatim.
export RAIDEN_DEVICES_PER_HOST=4
export USE_WEIGHT_CONVERTER=true
# Likewise unread; PREFUSE_MOE_WEIGHTS below is the one that takes effect.
export ROLLOUT_PREFUSE_MOE_WEIGHTS=true
export PREFUSE_MOE_WEIGHTS=true
export VERIFY_WEIGHTS=false
export ENABLE_PREFIX_CACHING=false

# 8. Training & Hyperparameters (100 steps)
export MAX_STEPS=100
export BATCH_SIZE=2
export NUM_GENERATIONS=4
export MINI_BATCH_SIZE=8
export TRAIN_MICRO_BATCH_SIZE=8
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=512
export USE_ROLLOUT_LOGPS=true
# Also unread by tunix; left in place to match the documented config.
export ROLLOUT_TIMEOUT_S=1800
export SAMPLER="inprocess_vllm"
export DEBUG=1
export DRY_RUN=${DRY_RUN:-false}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-start}"

bash "${SCRIPT_DIR}/k8s_launcher.sh" "${COMMAND}"

