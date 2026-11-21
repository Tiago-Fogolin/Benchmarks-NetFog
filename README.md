# Benchmarks NetFog

Scripts e dados brutos do experimento comparativo usado no TCC em que a NetFog foi desenvolvida.

Repositório da biblioteca: https://github.com/Tiago-Fogolin/NetFog

## Estrutura

- `runner-python/` — runner do benchmark comparativo (NetFog vs igraph, NetworkX,
  graph-tool, NetworKit), seis kernels do GAP Benchmark Suite.
- `experiments/` — experimento complementar de inserção dinâmica de arestas
- `resultados_brutos/` — dados brutos das execuções usadas no TCC:
  - `kernels_raw.json`: tempos por repetição (8 cada) dos seis kernels, para todas
    as estruturas/bibliotecas e especificações de grafo.
  - `dynamic_inserts_raw.csv`: resultados do experimento de inserção dinâmica
    (tempo mediano por aresta, por estrutura, K e especificação de grafo).
- `docker-compose.yml` / `docker-compose.dynamic.yml` / `.env.example` — como os
  containers foram orquestrados no servidor do LIA² (Unimar) durante a coleta.

