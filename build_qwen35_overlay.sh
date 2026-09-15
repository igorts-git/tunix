#!/usr/bin/env bash
# Builds and pushes the 8-file overlay image used by the Qwen3.5-35B-A3B RL run.
#
#   ./build_qwen35_overlay.sh [image-tag]
#
# Seven files come from this tunix checkout and one from the maxtext checkout
# alongside it (override with MAXTEXT_DIR).
#
# Defaults to gcr.io/cloud-tpu-multipod-dev/${USER}-runner:qwen35-repro-v11.
# Run `gcloud auth configure-docker gcr.io -q` once first.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The overlay spans two checkouts: tunix (this repo) and maxtext.
MAXTEXT_DIR="${MAXTEXT_DIR:-${REPO_DIR}/../maxtext}"
BASE_IMAGE="${BASE_IMAGE:-gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-e2e-0912head-v8}"
IMAGE="${1:-gcr.io/cloud-tpu-multipod-dev/${USER}-runner:qwen35-repro-v11}"

DOCKER=(docker)
if ! docker info &>/dev/null; then
  DOCKER=(sudo -n -E env "HOME=$HOME" "DOCKER_CONFIG=$HOME/.docker" docker)
fi

TUNIX_FILES=(
  tunix/experimental/orchestrator/distributed_rl_engine.py
  tunix/experimental/examples/math_gsm8k_dist/run_gsm8k_dist_grpo.py
  tunix/experimental/examples/math_gsm8k_dist/gsm8k.py
  tunix/experimental/examples/common/run_rollout_node.py
  tunix/utils/maxtext_utils.py
  tunix/experimental/examples/common/run_trainer_node.py
  tunix/rl/agentic/trajectory/trajectory_collect_engine.py
)
# Paths relative to MAXTEXT_DIR; staged under maxtext/ in the build context.
MAXTEXT_FILES=(
  src/maxtext/training_engine/maxtext_engine.py
)

# Build from a context holding only the overlaid files; the repo root is
# several GB and would otherwise all be shipped to the daemon.
CTX="$(mktemp -d)"
trap 'rm -rf "${CTX}"' EXIT
for f in "${TUNIX_FILES[@]}"; do
  mkdir -p "${CTX}/$(dirname "$f")"
  cp "${REPO_DIR}/$f" "${CTX}/$f"
done
for f in "${MAXTEXT_FILES[@]}"; do
  mkdir -p "${CTX}/maxtext/$(dirname "$f")"
  cp "${MAXTEXT_DIR}/$f" "${CTX}/maxtext/$f"
done
cp "${REPO_DIR}/Dockerfile.qwen35_overlay" "${CTX}/Dockerfile"

"${DOCKER[@]}" pull "${BASE_IMAGE}"
"${DOCKER[@]}" build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${IMAGE}" \
  "${CTX}"
"${DOCKER[@]}" push "${IMAGE}"

echo "Built and pushed ${IMAGE}"
