ARG CUDA_IMAGE_DEVEL=nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
ARG CUDA_IMAGE_RUNTIME=nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

FROM ${CUDA_IMAGE_DEVEL} AS base-build

ENV DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/rag-bench/.venv \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      ca-certificates \
      pkg-config \
      python3 \
      python3-venv \
      python3-pip \
      python3-dev \
      make \
      jq && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

FROM base-build AS deps

COPY pyproject.toml uv.lock README.md LICENSE Makefile ./
COPY src ./src

RUN make setup && make sync && \
    uv python install 3.13 && \
    uv python install 3.14 && \
    uv cache prune --ci

FROM ${CUDA_IMAGE_RUNTIME} AS runtime

ENV UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/rag-bench/.venv \
    PATH="/root/.local/bin:${PATH}"

# Minimal runtime packages to run the tests/venv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      python3 \
      python3-venv \
      python3-distutils \
      make \
      jq && \
    rm -rf /var/lib/apt/lists/*

COPY --from=deps /root/.local /root/.local
COPY --from=deps /opt/rag-bench/.venv /opt/rag-bench/.venv

RUN printf '#!/usr/bin/env bash\nset -euo pipefail\nmake setup && make sync && make test-all-gpu\n' \
    > /usr/local/bin/run-gpu-tests.sh && \
    chmod +x /usr/local/bin/run-gpu-tests.sh

ENTRYPOINT ["/bin/bash"]
