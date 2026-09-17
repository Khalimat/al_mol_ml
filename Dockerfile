FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# torch ships as two mutually exclusive builds (see pyproject.toml's
# tool.uv.sources): build with `--build-arg TORCH_VARIANT=gpu` for a CUDA
# image, or leave the default for a portable CPU-only one.
ARG TORCH_VARIANT=cpu

# Install dependencies before copying the package source, so this (slow:
# torch, rdkit) layer stays cached across source-only changes. `graph` adds
# torch_geometric, for `--architecture gnn` (see model_backends.py).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra ${TORCH_VARIANT} --extra graph

COPY README.md ./
COPY almolml ./almolml
COPY Datasets ./Datasets
RUN uv sync --frozen --extra ${TORCH_VARIANT} --extra graph

ENTRYPOINT ["python", "-m", "almolml"]
CMD ["--help"]
