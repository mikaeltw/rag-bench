#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${IMAGE_REF:-}" ]]; then
  if [[ -n "${GCP_PROJECT:-}" ]]; then
    GCP_ARTIFACT_REGION="${GCP_ARTIFACT_REGION:-us}"
    IMAGE_REF="${GCP_ARTIFACT_REGION}-docker.pkg.dev/${GCP_PROJECT}/rag-bench/rag-bench-gpu-tests:latest"
  else
    IMAGE_REF="rag-bench-gpu-tests:latest"
  fi
fi

DOCKER_ENTRYPOINT=()
if [[ $# -eq 0 ]]; then
  DOCKER_ENTRYPOINT=(--entrypoint /bin/bash)
  COMMAND=(-lc "make setup && make sync && make test-all-gpu")
else
  COMMAND=("$@")
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to run GPU tests inside a container." >&2
  exit 1
fi

HUGGINGFACE_CACHE="${HUGGINGFACE_CACHE:-$HOME/.cache/huggingface}"
TORCH_CACHE="${TORCH_CACHE:-$HOME/.cache/torch}"
UV_CACHE="${UV_CACHE:-$HOME/.cache/uv}"

mkdir -p "${HUGGINGFACE_CACHE}" "${TORCH_CACHE}" "${UV_CACHE}"

docker run --rm --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  -v "${HUGGINGFACE_CACHE}":/root/.cache/huggingface \
  -v "${TORCH_CACHE}":/root/.cache/torch \
  -v "${UV_CACHE}":/root/.cache/uv \
  -e RAG_BENCH_DEVICE=gpu \
  "${DOCKER_ENTRYPOINT[@]}" \
  "${IMAGE_REF}" \
  "${COMMAND[@]}"
