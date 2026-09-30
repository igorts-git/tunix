#!/bin/bash
# FP8 MoE comparison on v5p (DeepSWE, Qwen3.5-35B-A3B): bf16 = bf16 rollout and trainer,
# r8 = FP8 rollout experts, r8t8 = FP8 rollout experts + FP8 fake-quant trainer experts.
# Usage: fp8_deepswe_35b_v5p.sh <bf16|r8|r8t8> [launcher args, e.g. --dry-run | stop]
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
variant=$1; shift
case "$variant" in
  bf16) export ROLLOUT_FP8=false TRAINER_FP8=false ;;
  r8)   export ROLLOUT_FP8=true  TRAINER_FP8=false ;;
  r8t8) export ROLLOUT_FP8=true  TRAINER_FP8=true ;;
  *) echo "unknown variant $variant"; exit 1 ;;
esac
export JOB_PREFIX="igfp8-${variant}"
export TUNIX_IMAGE="gcr.io/cloud-tpu-multipod-dev/igorts/trellis:igorts-pc"
# bodaborg-v5p-nap (europe-west4), trellis queue: 64-chip trainer + 8 x 4-chip rollouts = 96 chips per variant.
export TPU_RESERVATION=cloudtpu-20260902214500-1810493672
export WANDB_RUN_NAME="${JOB_PREFIX}-35b-v5p"
export ROLLOUT_REPLICAS=8 MAX_STEPS=${MAX_STEPS:-20}
exec "${DIR}/mlperf_35b_128_v5p.sh" "$@"
