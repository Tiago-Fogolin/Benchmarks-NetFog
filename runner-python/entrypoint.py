import json
import os
from datetime import datetime, timezone

from bench_adapters import ALL_KERNEL_NAMES
from bench_runner import run_isolated

RESULTS_DIR = "/results"


def _meta_path(run_dir):
    return os.path.join(run_dir, "_meta.json")


def _write_meta(run_dir, **fields):
    path = _meta_path(run_dir)
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.update(fields)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

NETFOG_STRUCTURES = [
    "AdjacencyList",
    "AdjacencyMatrix",
    "CompressedSparseRow",
    "PackedCompressedSparseRow",
]
NETFOG_MODES = ["id", "label"]
OTHER_LIBS = ["igraph", "networkx"]
THREAD_VARIANT_LIBS = ["graph_tool", "networkit"]

ER_SIZES = [1000, 20000, 75000, 200000]
ER_DENSITIES = {"esparsa": 3, "densa": 20}


def build_matrix():
    specs = []
    for size in ER_SIZES:
        for density_name, avg_degree in ER_DENSITIES.items():
            specs.append({
                "name": f"er_{size}_{density_name}",
                "n": size,
                "avg_degree": avg_degree,
                "kind": "er",
                "seed": 42,
            })
    specs.append({
        "name": "ba_20000_densa",
        "n": 20000,
        "avg_degree": 20,
        "kind": "ba",
        "seed": 42,
    })
    specs.append({
        "name": "ws_20000_densa",
        "n": 20000,
        "avg_degree": 20,
        "kind": "ws",
        "seed": 42,
    })
    return specs


def all_combos():
    combos = []
    for structure in NETFOG_STRUCTURES:
        for mode in NETFOG_MODES:
            combos.append(("netfog", {"structure": structure, "mode": mode}))
    for lib in OTHER_LIBS:
        combos.append((lib, None))
    for lib in THREAD_VARIANT_LIBS:
        combos.append((lib, {"threads": 1}))
        combos.append((lib, {"threads": "max"}))
    return combos


def combo_key(graph_spec, library, variant):
    if variant and "structure" in variant:
        suffix = f"_{variant['structure']}_{variant['mode']}"
    elif variant and "threads" in variant:
        suffix = "_1thread" if variant["threads"] == 1 else "_parallel"
    else:
        suffix = ""
    return f"{graph_spec['name']}__{library}{suffix}"



BETWEENNESS_MAX_N = 20000


def kernels_for_spec(graph_spec):
    if graph_spec["n"] > BETWEENNESS_MAX_N:
        wanted = [k for k in ALL_KERNEL_NAMES if k != "betweenness_centrality"]
    else:
        wanted = list(ALL_KERNEL_NAMES)

    kernels_filter = os.environ.get("BENCH_KERNELS", "").strip()
    if kernels_filter:
        kernels_wanted_set = {name.strip() for name in kernels_filter.split(",") if name.strip()}
        wanted = [k for k in wanted if k in kernels_wanted_set]

    return wanted


def run_matrix(graph_specs, repeats, timeout_s, run_dir):
    libs_filter = os.environ.get("BENCH_LIBS", "").strip()
    libs_wanted = {name.strip() for name in libs_filter.split(",") if name.strip()} if libs_filter else None

    index = []
    for graph_spec in graph_specs:
        kernels_wanted = kernels_for_spec(graph_spec)
        for library, variant in all_combos():
            if libs_wanted is not None and library not in libs_wanted:
                continue
            key = combo_key(graph_spec, library, variant)
            out_path = f"{run_dir}/{key}.json"
            print(f"[{datetime.now(timezone.utc).isoformat()}] rodando {key}...", flush=True)
            status = run_isolated(library, variant, graph_spec, kernels_wanted, repeats, out_path, timeout_s)
            print(f"  -> {status}", flush=True)
            index.append({"key": key, "status": status, "path": out_path})
    return index


def _collect_kernel_values(index, kernel_name):
    values = {}
    for entry in index:
        if entry["status"] != "ok" or not os.path.exists(entry["path"]):
            continue
        with open(entry["path"], encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status") != "concluido":
            continue
        kernel_result = data["kernels"].get(kernel_name, {})
        if kernel_result.get("status") == "ok":
            values[entry["key"]] = kernel_result["value"]
    return values


def smoke_test(run_dir):
    smallest = {
        "name": "smoke_1000_esparsa",
        "n": 1000,
        "avg_degree": 3,
        "kind": "er",
        "seed": 42,
    }
    print("=== SMOKE TEST: matriz reduzida (1 grafo pequeno, 1 repeticao) ===", flush=True)
    index = run_matrix([smallest], repeats=1, timeout_s=300, run_dir=run_dir)

    tri_values = _collect_kernel_values(index, "triangle_count")
    cc_values = _collect_kernel_values(index, "connected_components")

    tri_consistent = len(set(tri_values.values())) <= 1
    cc_consistent = len(set(cc_values.values())) <= 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "combos_total": len(index),
        "combos_ok": sum(1 for e in index if e["status"] == "ok"),
        "combos_status": {e["key"]: e["status"] for e in index},
        "triangle_count_values": tri_values,
        "triangle_count_consistent": tri_consistent,
        "connected_components_values": cc_values,
        "connected_components_consistent": cc_consistent,
    }

    with open(f"{run_dir}/smoke_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    passed = tri_consistent and cc_consistent
    print("=== SMOKE TEST: " + ("PASSOU" if passed else "FALHOU") + " ===", flush=True)
    return passed


def full_run(run_dir):
    specs = build_matrix()

    sizes_filter = os.environ.get("BENCH_SIZES", "").strip()
    if sizes_filter:
        wanted = {name.strip() for name in sizes_filter.split(",") if name.strip()}
        all_names = {spec["name"] for spec in specs}
        unknown = wanted - all_names
        if unknown:
            print(f"AVISO: BENCH_SIZES tem nome(s) desconhecido(s), ignorados: {sorted(unknown)}", flush=True)
        specs = [spec for spec in specs if spec["name"] in wanted]
        print(f"BENCH_SIZES ativo -- rodando so: {[s['name'] for s in specs]}", flush=True)

    repeats = int(os.environ.get("BENCH_REPEATS", "8"))
    timeout_s = int(os.environ.get("BENCH_TIMEOUT_S", "1800"))
    index = run_matrix(specs, repeats, timeout_s, run_dir=run_dir)
    with open(f"{run_dir}/run_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"=== RUN COMPLETO: {len(index)} combinacoes, ver run_index.json ===", flush=True)


def main():
    mode = os.environ.get("BENCH_MODE", "smoke")
    run_id = f"{mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    _write_meta(run_dir, mode=mode, status="em_andamento", started_at=datetime.now(timezone.utc).isoformat())

    if mode == "smoke":
        passed = smoke_test(run_dir)
        _write_meta(run_dir, status="concluido" if passed else "falhou", finished_at=datetime.now(timezone.utc).isoformat())
    else:
        full_run(run_dir)
        _write_meta(run_dir, status="concluido", finished_at=datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
