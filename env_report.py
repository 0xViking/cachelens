"""CacheLens serving-environment report (reads local files only, no network).

Collects what a serving setup actually exposes about itself and reports
cache-relevant facts with their sources. It never probes hosts, never calls
a network API, and never guesses: every field is either parsed from the
given file or reported "not available".

Inputs (all local files the app owner provides):
  --log PATH        vLLM server startup/runtime log (plain text)
  --metrics PATH    Prometheus /metrics text (e.g. curl server:8000/metrics),
                    or a JSON dict of {series: value}
  --metrics-later PATH optional: a second /metrics snapshot taken AFTER --metrics;
                    only a window between two snapshots supports any
                    hit-rate or eviction reading
  --mean-prompt-tokens N   optional: your app's mean prompt size in tokens,
                    used only for an arithmetic capacity ratio
  --with-nvidia-smi        additionally query THIS machine's nvidia-smi
                    (only meaningful when run on the serving host)

Observations are tied to observed values. Static metadata alone is never
used to claim a hardware bottleneck.

Usage: python3 env_report.py --log vllm.log --metrics metrics.txt
"""
import json, re, subprocess, sys

NA = "not available"

def parse_log(text):
    f = {}
    m = re.search(r"Initializing a V1 LLM engine \(v([0-9][^)]*)\) with config: model='([^']+)'", text)
    if m:
        f["vllm_version"] = m.group(1)
        f["model"] = m.group(2)
    m = re.search(r"enable_prefix_caching=(\w+)", text)
    if m:
        f["enable_prefix_caching"] = m.group(1) == "True"
    m = re.search(r"dtype=torch\.(\w+)", text)
    if m:
        f["dtype"] = m.group(1)
    m = re.search(r"max_seq_len=(\d+)", text)
    if m:
        f["max_seq_len"] = int(m.group(1))
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if m:
        f["gpu_kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Available KV cache memory: ([\d.]+) GiB", text)
    if m:
        f["kv_cache_memory_gib"] = float(m.group(1))
    m = re.search(r"Free memory on device \(([\d.]+)/([\d.]+) GiB\)", text)
    if m:
        f["gpu_memory_free_gib_at_startup"] = float(m.group(1))
        f["gpu_memory_total_gib"] = float(m.group(2))
    m = re.search(r"Desired GPU memory utilization is \(([\d.]+),", text)
    if m:
        f["gpu_memory_utilization_setting"] = float(m.group(1))
    m = re.search(r"tensor_parallel_size=(\d+)", text)
    if m:
        f["tensor_parallel_size"] = int(m.group(1))
    return f

def parse_metrics(text):
    """Return {name: value} for the series CacheLens uses. Missing -> absent."""
    out = {}
    if text.lstrip().startswith("{"):
        data = json.loads(text)
        for k, v in data.items():
            name = k.split("{")[0]
            try:
                out[name] = float(v)
            except (TypeError, ValueError):
                pass
        return out
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"(vllm:[a-z_]+)(?:\{[^}]*\})?\s+([-\d.eE+]+)$", line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out

def gpu_names():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            return [l.strip() for l in r.stdout.strip().splitlines()]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None

def report(log_text, metrics_text, mean_prompt_tokens, use_smi, metrics_later_text=None):
    facts, sources, obs = {}, {}, []

    if log_text is not None:
        lf = parse_log(log_text)
        for k, v in lf.items():
            facts[k] = v
            sources[k] = "server log"
    mt = None
    if metrics_text is not None:
        mt = parse_metrics(metrics_text)
        for k in ("vllm:gpu_cache_usage_perc", "vllm:prefix_cache_queries_total",
                  "vllm:prefix_cache_hits_total"):
            if k in mt:
                facts[k] = mt[k]
                sources[k] = "metrics endpoint (snapshot 1)"
    mt2 = None
    if metrics_later_text is not None:
        mt2 = parse_metrics(metrics_later_text)
    if use_smi:
        g = gpu_names()
        facts["nvidia_smi_gpus_this_machine"] = g if g else NA
        sources["nvidia_smi_gpus_this_machine"] = "nvidia-smi on the machine running this report"

    expected = ["vllm_version", "model", "enable_prefix_caching", "dtype",
                "max_seq_len", "gpu_kv_cache_tokens", "kv_cache_memory_gib",
                "gpu_memory_total_gib", "gpu_memory_utilization_setting",
                "tensor_parallel_size"]
    for k in expected:
        facts.setdefault(k, NA)
        sources.setdefault(k, NA)

    # Observations: each one cites the values it stands on.
    if facts["enable_prefix_caching"] is False:
        obs.append({"observation": "prefix caching is disabled on this engine",
                    "evidence": "enable_prefix_caching=False in the server config",
                    "note": "no prompt layout can hit a prefix cache that is off"})
    elif facts["enable_prefix_caching"] is True:
        obs.append({"observation": "prefix caching is enabled",
                    "evidence": "enable_prefix_caching=True in the server config",
                    "note": "misses then point to layout, capacity, eviction or routing"})

    if isinstance(facts.get("gpu_kv_cache_tokens"), int) and mean_prompt_tokens:
        ratio = facts["gpu_kv_cache_tokens"] / mean_prompt_tokens
        obs.append({"observation": "KV cache holds about %.1f mean-size prompts' worth of tokens" % ratio,
                    "evidence": "GPU KV cache size %d tokens vs mean prompt %d tokens (both observed/given)"
                                % (facts["gpu_kv_cache_tokens"], mean_prompt_tokens),
                    "note": "a small ratio means new prompts evict old prefixes quickly; "
                            "block granularity and concurrency also matter"})

    usage = facts.get("vllm:gpu_cache_usage_perc")
    if usage is not None:
        obs.append({"observation": "GPU KV cache usage was %.0f%% in one snapshot" % (100 * usage),
                    "evidence": "vllm:gpu_cache_usage_perc=%.3f (metrics snapshot 1)" % usage,
                    "note": "a single snapshot cannot separate eviction pressure from a "
                            "cold start or a burst; no conclusion drawn from it"})
    if mt is not None and mt2 is not None:
        dq = mt2.get("vllm:prefix_cache_queries_total", 0) - mt.get("vllm:prefix_cache_queries_total", 0)
        dh = mt2.get("vllm:prefix_cache_hits_total", 0) - mt.get("vllm:prefix_cache_hits_total", 0)
        if dq > 0:
            win = 100.0 * dh / dq
            facts["window_prefix_cache_hit_pct"] = round(win, 1)
            sources["window_prefix_cache_hit_pct"] = "delta between the two metrics snapshots"
            obs.append({"observation": "prefix cache hit rate between the two snapshots was %.1f%% (%d of %d queries)"
                                       % (win, int(dh), int(dq)),
                        "evidence": "counter delta between metrics snapshot 1 and snapshot 2",
                        "note": "window-based and engine-reported; low values with early prompt "
                                "variation point at layout, with a small/full cache at capacity"})
        else:
            obs.append({"observation": "no prefix-cache queries between the two snapshots",
                        "evidence": "query counter delta is %d" % int(dq),
                        "note": "the window carried no traffic CacheLens can read"})
    elif mt is not None:
        facts["prefix_cache_counters_note"] = ("lifetime counters since engine start are facts about "
                                               "totals, not a hit-rate reading; a second snapshot "
                                               "(--metrics-later) enables a window")
    # lifetime counters stay facts only; never an interpretation
    if not obs:
        obs.append({"observation": "no cache-relevant signal in the provided files",
                    "evidence": "parsed fields are all 'not available' or neutral",
                    "note": "provide the server startup log and/or a /metrics snapshot"})
    return {"facts": facts, "sources": sources, "observations": obs,
            "scope_note": ("facts are parsed from the files/settings given; 'not available' "
                           "means the file did not show it. This report does not probe any "
                           "host and does not claim a hardware bottleneck from metadata")}

def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="CacheLens serving-environment report")
    ap.add_argument("--log"); ap.add_argument("--metrics"); ap.add_argument("--metrics-later")
    ap.add_argument("--mean-prompt-tokens", type=int)
    ap.add_argument("--with-nvidia-smi", action="store_true")
    a = ap.parse_args(argv)
    log_text = metrics_text = metrics_later = None
    try:
        if a.log:
            log_text = open(a.log, errors="replace").read()
        if a.metrics:
            metrics_text = open(a.metrics, errors="replace").read()
        if a.metrics_later:
            metrics_later = open(a.metrics_later, errors="replace").read()
    except OSError as e:
        print(f"INCONCLUSIVE - cannot read input: {e.strerror or e}")
        return 2
    if log_text is None and metrics_text is None and not a.with_nvidia_smi:
        print("INCONCLUSIVE - give --log and/or --metrics (or --with-nvidia-smi on the server host)")
        return 2
    res = report(log_text, metrics_text, a.mean_prompt_tokens, a.with_nvidia_smi,
                 metrics_later_text=metrics_later)
    if all(v == NA for v in res["facts"].values()):
        print("INCONCLUSIVE - nothing parseable in the provided input")
        return 2
    print("== CacheLens serving-environment report ==")
    print(json.dumps(res, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
