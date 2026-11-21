

import csv
import json
import multiprocessing
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

try:
   from bench_adapters import generate_graph, build_netfog
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runner-python"))
    from bench_adapters import generate_graph, build_netfog 

RESULTS_ROOT = os.environ.get("DYN_RESULTS_ROOT", "/results/dynamic")

GRAPH_SPECS = {
    "er_1000_esparsa": {"n": 1000, "avg_degree": 3},
    "er_1000_densa": {"n": 1000, "avg_degree": 20},
    "er_20000_esparsa": {"n": 20000, "avg_degree": 3},
    "er_20000_densa": {"n": 20000, "avg_degree": 20},
    "er_75000_esparsa": {"n": 75000, "avg_degree": 3},
    "er_75000_densa": {"n": 75000, "avg_degree": 20},
    "er_200000_esparsa": {"n": 200000, "avg_degree": 3},
    "er_200000_densa": {"n": 200000, "avg_degree": 20},
}

ALL_STRUCTURES = [
    "AdjacencyList",
    "AdjacencyMatrix",
    "CompressedSparseRow",
    "PackedCompressedSparseRow",
]

DEFAULTS = {
    "DYN_GRAPHS": "er_1000_esparsa,er_1000_densa,er_20000_esparsa,er_20000_densa",
    "DYN_STRUCTURES": ",".join(ALL_STRUCTURES),
    "DYN_K": "1,10,100,1000",
    "DYN_MODE": "id",
    "DYN_REPEATS": "8",
    "DYN_TIMEOUT_S": "600",
    "DYN_SOFT_MEM_FRACTION": "0.85",
}


def _env(name):
    value = os.environ.get(name, "").strip()
    return value if value else DEFAULTS[name]


def _parse_list(csv_value):
    return [item.strip() for item in csv_value.split(",") if item.strip()]




def _write_result(out_path, result):
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)


def _soft_memory_limit_bytes():
    env_value = os.environ.get("DYN_SOFT_MEM_BYTES")
    if env_value:
        return int(env_value)
    hard_limit_gb = os.environ.get("BENCH_MEMORY_LIMIT_GB")
    if hard_limit_gb:
        fraction = float(_env("DYN_SOFT_MEM_FRACTION"))
        return int(float(hard_limit_gb) * fraction * (1024 ** 3))
    return None


def _pick_new_edges(n, existing_edges, k, seed):
    rng = random.Random(seed)
    existing = set()
    for u, v in existing_edges:
        existing.add((u, v))
        existing.add((v, u))

    chosen = []
    chosen_set = set()
    attempts = 0
    max_attempts = k * 200 + 10000
    while len(chosen) < k and attempts < max_attempts:
        attempts += 1
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        pair = (u, v)
        rev = (v, u)
        if pair in existing or pair in chosen_set or rev in chosen_set:
            continue
        chosen.append(pair)
        chosen_set.add(pair)

    if len(chosen) < k:
        raise RuntimeError(
            f"nao foi possivel amostrar {k} arestas novas unicas "
            f"(conseguiu {len(chosen)}) -- grafo denso demais pro n dado"
        )
    return chosen


def _child_main(graph_name, structure, k, mode, repeats, out_path):
    soft_limit = _soft_memory_limit_bytes()
    if soft_limit:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (soft_limit, resource.RLIM_INFINITY))
        except (ImportError, ValueError, OSError):
            pass

    spec = GRAPH_SPECS[graph_name]
    result = {
        "graph": graph_name,
        "structure": structure,
        "mode": mode,
        "k": k,
        "status": "em_andamento",
        "per_repeat_total_s": [],
        "per_repeat_per_edge_s": [],
    }
    _write_result(out_path, result)

    try:
        n, edges, _start_node = generate_graph(spec["n"], spec["avg_degree"], kind="er", seed=42)
    except Exception as exc:
        result["status"] = "erro"
        result["error"] = f"geracao do grafo base falhou: {exc}"
        _write_result(out_path, result)
        return

    build_times = []
    for repeat_idx in range(repeats):
        try:
            g, build_elapsed, _ = build_netfog(n, edges, {"structure": structure, "mode": mode})
            build_times.append(build_elapsed)
        except MemoryError:
            result["status"] = "inviavel"
            result["error"] = "MemoryError na construcao do grafo base (soft limit atingido)"
            result["failure_reason"] = "oom"
            _write_result(out_path, result)
            return
        except Exception as exc:
            result["status"] = "erro"
            result["error"] = f"construcao do grafo base falhou: {exc}"
            _write_result(out_path, result)
            return

        try:
            new_edges = _pick_new_edges(n, edges, k, seed=1000 + repeat_idx)
        except Exception as exc:
            result["status"] = "erro"
            result["error"] = f"amostragem de arestas novas falhou: {exc}"
            _write_result(out_path, result)
            return

        def _ref(idx):
            return idx if mode == "id" else str(idx)

        try:
            t0 = time.perf_counter()
            for u, v in new_edges:
                g.create_connection(_ref(u), _ref(v), 1.0, False)
            elapsed = time.perf_counter() - t0
        except MemoryError:
            result["status"] = "inviavel"
            result["error"] = "MemoryError durante insercao (soft limit atingido)"
            result["failure_reason"] = "oom"
            _write_result(out_path, result)
            return
        except Exception as exc:
            result["status"] = "erro"
            result["error"] = f"insercao falhou: {exc}"
            _write_result(out_path, result)
            return

        result["per_repeat_total_s"].append(elapsed)
        result["per_repeat_per_edge_s"].append(elapsed / k)
        _write_result(out_path, result)

    result["build_time_s_median"] = sorted(build_times)[len(build_times) // 2]
    result["status"] = "concluido"
    _write_result(out_path, result)


def _mark_incomplete(out_path, graph_name, structure, k, mode, reason, failure_reason):
    existing = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing.setdefault("graph", graph_name)
    existing.setdefault("structure", structure)
    existing.setdefault("mode", mode)
    existing.setdefault("k", k)
    existing.setdefault("per_repeat_total_s", [])
    existing.setdefault("per_repeat_per_edge_s", [])
    existing["status"] = "inviavel"
    existing["error"] = reason
    existing["failure_reason"] = failure_reason
    _write_result(out_path, existing)


def run_isolated(graph_name, structure, k, mode, repeats, out_path, timeout_s):
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_child_main,
        args=(graph_name, structure, k, mode, repeats, out_path),
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        _mark_incomplete(out_path, graph_name, structure, k, mode, f"timeout ({timeout_s}s)", "timeout")
        return "timeout"

    if proc.exitcode != 0:
        if proc.exitcode is not None and proc.exitcode < 0:
            sig = -proc.exitcode
            try:
                sig_name = signal.Signals(sig).name
            except ValueError:
                sig_name = str(sig)
            failure_reason = "oom" if sig_name == "SIGKILL" else "signal"
            reason = f"subprocesso morreu com sinal {sig_name} ({sig})"
        else:
            failure_reason = "exit_code"
            reason = f"subprocesso saiu com codigo {proc.exitcode}"
        _mark_incomplete(out_path, graph_name, structure, k, mode, reason, failure_reason)
        return "crash"

    return "ok"



def _netfog_commit():
    env_commit = os.environ.get("NETFOG_COMMIT")
    if env_commit:
        return env_commit
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=7"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "desconhecido"


def _write_meta(run_dir):
    import netfog
    meta = {
        "netfog_commit": _netfog_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "netfog_module_file": getattr(netfog, "__file__", "?"),
        "env": {k: v for k, v in os.environ.items() if k.startswith("DYN_") or k.startswith("BENCH_")},
    }
    with open(os.path.join(run_dir, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def _write_csv(run_dir, rows):
    path = os.path.join(run_dir, "resultados.csv")
    fieldnames = [
        "graph", "n", "avg_degree", "structure", "mode", "k", "status",
        "median_total_s", "median_per_edge_s", "build_time_s_median", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path

def main():
    graphs = _parse_list(_env("DYN_GRAPHS"))
    structures = _parse_list(_env("DYN_STRUCTURES"))
    k_values = [int(k) for k in _parse_list(_env("DYN_K"))]
    mode = _env("DYN_MODE")
    repeats = int(_env("DYN_REPEATS"))
    timeout_s = int(_env("DYN_TIMEOUT_S"))

    unknown_graphs = [g for g in graphs if g not in GRAPH_SPECS]
    if unknown_graphs:
        raise ValueError(f"DYN_GRAPHS tem nome(s) desconhecido(s): {unknown_graphs}")
    unknown_structures = [s for s in structures if s not in ALL_STRUCTURES]
    if unknown_structures:
        raise ValueError(f"DYN_STRUCTURES tem nome(s) desconhecido(s): {unknown_structures}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(RESULTS_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    meta = _write_meta(run_dir)
    print(f"=== insercao dinamica -- run {run_id} ===", flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    print(
        f"grafos={graphs} estruturas={structures} K={k_values} "
        f"mode={mode} repeats={repeats} timeout_s={timeout_s}",
        flush=True,
    )

    rows = []
    total_combos = len(graphs) * len(structures) * len(k_values)
    done = 0
    for graph_name in graphs:
        spec = GRAPH_SPECS[graph_name]
        for structure in structures:
            for k in k_values:
                done += 1
                key = f"{graph_name}__{structure}_{mode}__k{k}"
                out_path = os.path.join(run_dir, f"{key}.json")
                print(f"[{done}/{total_combos}] rodando {key}...", flush=True)
                status = run_isolated(graph_name, structure, k, mode, repeats, out_path, timeout_s)
                print(f"  -> {status}", flush=True)

                with open(out_path, encoding="utf-8") as f:
                    data = json.load(f)

                median_total = None
                median_per_edge = None
                if data.get("per_repeat_per_edge_s"):
                    times = sorted(data["per_repeat_per_edge_s"])
                    median_per_edge = times[len(times) // 2]
                    totals = sorted(data["per_repeat_total_s"])
                    median_total = totals[len(totals) // 2]

                rows.append({
                    "graph": graph_name,
                    "n": spec["n"],
                    "avg_degree": spec["avg_degree"],
                    "structure": structure,
                    "mode": mode,
                    "k": k,
                    "status": data.get("status", status),
                    "median_total_s": median_total,
                    "median_per_edge_s": median_per_edge,
                    "build_time_s_median": data.get("build_time_s_median"),
                    "error": data.get("error", ""),
                })

    csv_path = _write_csv(run_dir, rows)
    print(f"CSV gerado: {csv_path}", flush=True)



if __name__ == "__main__":
    main()
