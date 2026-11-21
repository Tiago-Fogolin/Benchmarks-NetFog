FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir maturin

WORKDIR /netfog
COPY . .
RUN maturin build --release --out /wheels

FROM condaforge/miniforge3


ENV CONDA_OVERRIDE_ARCHSPEC=x86_64

RUN mamba create -y -n bench -c conda-forge \
        python=3.11 python-igraph networkx networkit graph-tool \
    && mamba clean -afy

COPY --from=builder /wheels /wheels
RUN /opt/conda/envs/bench/bin/pip install --no-cache-dir /wheels/*.whl

WORKDIR /app
COPY benchmarks/runner-python/entrypoint.py .
COPY benchmarks/runner-python/bench_adapters.py .
COPY benchmarks/runner-python/bench_runner.py .

ENTRYPOINT ["/opt/conda/envs/bench/bin/python", "entrypoint.py"]
