FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install torch from the CPU-only wheel index first: the default PyPI
# "torch" wheel bundles CUDA runtime libraries that are dead weight (and
# multiple GB) in a CPU-only container.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY almolml ./almolml
COPY Datasets ./Datasets
RUN pip install --no-deps -e .

ENTRYPOINT ["python", "-m", "almolml"]
CMD ["--help"]
