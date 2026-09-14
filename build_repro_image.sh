#!/usr/bin/env bash
# Builds the Qwen3.5-35B-A3B reproduction image by overlaying the tunix and maxtext
# working trees onto the shared runner base. Both repos are needed: the round-robin
# rollout and metadata-stripping fixes live in tunix, the Pathways checkpoint drain
# lives in maxtext.
#
#   TUNIX_REPO=~/git/tunix MAXTEXT_REPO=~/git/maxtext ./build_repro_image.sh
#
# The resulting tag is what run_100steps.sh should be pointed at via TUNIX_IMAGE.

set -euo pipefail

TUNIX_REPO="${TUNIX_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MAXTEXT_REPO="${MAXTEXT_REPO:-${TUNIX_REPO}/../maxtext}"
IMAGE="${IMAGE:-gcr.io/cloud-tpu-multipod-dev/${USER}-runner:qwen35-repro-$(date +%m%d-%H%M)}"
BASE_IMAGE="${BASE_IMAGE:-gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-pr-tunix-2202-09120021}"
# Where the base image keeps its editable installs. The Dockerfile asserts the overlay
# took effect, so a wrong value fails the build rather than shipping unpatched sources.
TUNIX_DEST="${TUNIX_DEST:-/app/tunix}"
MAXTEXT_DEST="${MAXTEXT_DEST:-/app/maxtext}"

for repo in "${TUNIX_REPO}" "${MAXTEXT_REPO}"; do
  [[ -d "${repo}/.git" ]] || { echo "not a git checkout: ${repo}" >&2; exit 1; }
done

echo "tunix   $(git -C "${TUNIX_REPO}" rev-parse --abbrev-ref HEAD) $(git -C "${TUNIX_REPO}" rev-parse --short HEAD)"
echo "maxtext $(git -C "${MAXTEXT_REPO}" rev-parse --abbrev-ref HEAD) $(git -C "${MAXTEXT_REPO}" rev-parse --short HEAD)"

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

# Docker needs both trees under one context root, and the venvs are multi-GB.
for name in tunix maxtext; do
  src_var="$(echo "${name}" | tr '[:lower:]' '[:upper:]')_REPO"
  rsync -a \
    --exclude=.git --exclude=.venv --exclude=venv --exclude=maxtext_venv \
    --exclude=__pycache__ --exclude='*.pyc' \
    "${!src_var}/" "${STAGE}/${name}/"
done
cp "${TUNIX_REPO}/Dockerfile.qwen35_fast" "${STAGE}/Dockerfile"

docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg TUNIX_DEST="${TUNIX_DEST}" \
  --build-arg MAXTEXT_DEST="${MAXTEXT_DEST}" \
  -t "${IMAGE}" "${STAGE}"

docker push "${IMAGE}"
echo
echo "Built and pushed: ${IMAGE}"
echo "Set TUNIX_IMAGE to it in run_100steps.sh before launching."
