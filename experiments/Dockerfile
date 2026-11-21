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

FROM python:3.11-slim

RUN pip install --no-cache-dir networkx matplotlib

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl

WORKDIR /app
COPY benchmarks/runner-python/bench_adapters.py .
COPY benchmarks/experiments/dynamic_inserts.py .

ENTRYPOINT ["python", "dynamic_inserts.py"]
