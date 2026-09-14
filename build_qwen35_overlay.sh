#!/usr/bin/env bash
# Builds and pushes the 4-file overlay image used by the Qwen3.5-35B-A3B RL run.
#
#   ./build_qwen35_overlay.sh [image-tag]
#
# Defaults to gcr.io/cloud-tpu-multipod-dev/${USER}-runner:qwen35-repro-v7.
# Run `gcloud auth configure-docker gcr.io -q` once first.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-e2e-0912head-v8}"
IMAGE="${1:-gcr.io/cloud-tpu-multipod-dev/${USER}-runner:qwen35-repro-v6}"

DOCKER=(docker)
if ! docker info &>/dev/null; then
  DOCKER=(sudo -n -E env "HOME=$HOME" "DOCKER_CONFIG=$HOME/.docker" docker)
fi

OVERLAY_FILES=(
  tunix/experimental/orchestrator/algorithm_adapter.py
  tunix/experimental/orchestrator/distributed_rl_engine.py
  tunix/experimental/examples/math_gsm8k_dist/run_gsm8k_dist_grpo.py
  tunix/experimental/examples/common/run_rollout_node.py
)

# Build from a context holding only the overlaid files; the repo root is
# several GB and would otherwise all be shipped to the daemon.
CTX="$(mktemp -d)"
trap 'rm -rf "${CTX}"' EXIT
for f in "${OVERLAY_FILES[@]}"; do
  mkdir -p "${CTX}/$(dirname "$f")"
  cp "${REPO_DIR}/$f" "${CTX}/$f"
done
cp "${REPO_DIR}/Dockerfile.qwen35_overlay" "${CTX}/Dockerfile"

"${DOCKER[@]}" pull "${BASE_IMAGE}"
"${DOCKER[@]}" build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${IMAGE}" \
  "${CTX}"
"${DOCKER[@]}" push "${IMAGE}"

echo "Built and pushed ${IMAGE}"
