
import json
import math
import multiprocessing
import os
import signal
import time

from bench_adapters import BUILDERS, KERNELS, generate_graph

DEFAULT_SOFT_MEM_FRACTION = 0.85  # fracao do limite hard do container


def _json_safe_float(x):
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return "Infinity" if x > 0 else ("-Infinity" if x < 0 else "NaN")
    return x


def summarize(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return _json_safe_float(value)
    if isinstance(value, dict):
        vals = list(value.values())
    else:
        try:
            vals = list(value)
        except TypeError:
            return str(value)

    if not vals:
        return {"n": 0}
    if isinstance(vals[0], (int, float)):
        finite = [v for v in vals if not (isinstance(v, float) and (math.isinf(v) or math.isnan(v)))]
        return {
            "n": len(vals),
            "n_nao_finito": len(vals) - len(finite),
            "sum": _json_safe_float(sum(finite)) if finite else None,
            "min": _json_safe_float(min(finite)) if finite else None,
            "max": _json_safe_float(max(finite)) if finite else None,
        }
    return {"n": len(vals)}


def _soft_memory_limit_bytes():
    env_value = os.environ.get("BENCH_SOFT_MEM_BYTES")
    if env_value:
        return int(env_value)
    hard_limit_gb = os.environ.get("BENCH_MEMORY_LIMIT_GB")
    if hard_limit_gb:
        return int(float(hard_limit_gb) * DEFAULT_SOFT_MEM_FRACTION * (1024 ** 3))
    return None


def _write_result(out_path, result):
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)


def _child_main(library, variant, graph_spec, kernels_wanted, repeats, out_path):
    soft_limit = _soft_memory_limit_bytes()
    if soft_limit:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (soft_limit, resource.RLIM_INFINITY))
        except (ImportError, ValueError, OSError):
            pass

    result = {
        "library": library,
        "variant": variant,
        "graph_spec": graph_spec,
        "status": "em_andamento",
        "kernels": {},
    }

    try:
        gen_kwargs = {k: v for k, v in graph_spec.items() if k != "name"}
        n, edges, start_node = generate_graph(**gen_kwargs)
        result["start_node"] = start_node
    except Exception as exc:
        result["status"] = "erro"
        result["error"] = f"geracao do grafo falhou: {exc}"
        _write_result(out_path, result)
        return

    try:
        graph_obj, build_elapsed, build_time_ms_internal = BUILDERS[library](n, edges, variant)
    except MemoryError:
        result["status"] = "inviavel"
        result["error"] = "MemoryError na construcao (soft limit atingido)"
        result["failure_reason"] = "oom"
        _write_result(out_path, result)
        return
    except Exception as exc:
        result["status"] = "erro"
        result["error"] = f"construcao falhou: {exc}"
        _write_result(out_path, result)
        return

    result["build_time_s"] = build_elapsed
    if build_time_ms_internal is not None:
        result["build_time_ms_internal"] = build_time_ms_internal
    result["status"] = "em_andamento"
    _write_result(out_path, result)

    kernels = KERNELS[library](variant, start_node)

    for kname in kernels_wanted:
        if kname not in kernels:
            result["kernels"][kname] = {"status": "nao_suportado"}
            _write_result(out_path, result)
            continue

        try:
            times = []
            value = None
            for _ in range(repeats):
                value, elapsed = kernels[kname](graph_obj)
                times.append(elapsed)
            result["kernels"][kname] = {
                "status": "ok",
                "times_s": times,
                "value": summarize(value),
            }
        except MemoryError:
            result["kernels"][kname] = {
                "status": "inviavel",
                "error": "MemoryError (soft limit atingido)",
                "failure_reason": "oom",
            }
        except Exception as exc:
            result["kernels"][kname] = {"status": "erro", "error": str(exc)}

        _write_result(out_path, result)

    result["status"] = "concluido"
    _write_result(out_path, result)


def run_isolated(library, variant, graph_spec, kernels_wanted, repeats, out_path, timeout_s=1800):
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_child_main,
        args=(library, variant, graph_spec, kernels_wanted, repeats, out_path),
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        _mark_incomplete(out_path, library, variant, graph_spec, f"timeout ({timeout_s}s)", "timeout")
        return "timeout"

    if proc.exitcode != 0:
        if proc.exitcode is not None and proc.exitcode < 0:
            sig = -proc.exitcode
            try:
                sig_name = signal.Signals(sig).name
            except ValueError:
                sig_name = str(sig)
            if sig_name == "SIGKILL":
                failure_reason = "oom"
            elif sig_name == "SIGILL":
                failure_reason = "cpu_incompativel"
            else:
                failure_reason = "signal"
            reason = (
                f"subprocesso morreu com sinal {sig_name} ({sig}) -- "
                "provavel OOM (cgroup) ou instrucao de CPU incompativel"
            )
        else:
            failure_reason = "exit_code"
            reason = f"subprocesso saiu com codigo {proc.exitcode}"
        _mark_incomplete(out_path, library, variant, graph_spec, reason, failure_reason)
        return "crash"

    return "ok"


def _mark_incomplete(out_path, library, variant, graph_spec, reason, failure_reason):
    existing = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing.setdefault("library", library)
    existing.setdefault("variant", variant)
    existing.setdefault("graph_spec", graph_spec)
    existing.setdefault("kernels", {})
    existing["status"] = "inviavel"
    existing["error"] = reason
    existing["failure_reason"] = failure_reason
    _write_result(out_path, existing)
