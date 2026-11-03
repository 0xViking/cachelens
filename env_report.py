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
import json, math, re, subprocess, sys

class BadInput(Exception):
    pass

NA = "not available"

def _last(pattern, text):
    ms = list(re.finditer(pattern, text))
    return ms[-1] if ms else None

def parse_log(text):
    f = {}
    inits = list(re.finditer(r"Initializing a V1 LLM engine \(v([0-9][^)]*)\) with config: model='([^']+)'", text))
    f["engine_inits_in_log"] = len(inits)
    if inits:
        # facts below describe the LAST engine start in this log
        last = inits[-1]
        f["vllm_version"] = last.group(1)
        f["model"] = last.group(2)
        seg = text[last.start():]
    else:
        seg = text
    for key, pat, conv in (
        ("enable_prefix_caching", r"enable_prefix_caching=(\w+)", lambda g: g == "True"),
        ("served_model_name", r"served_model_name=([^,\s]+)", str),
        ("dtype", r"dtype=torch\.(\w+)", str),
        ("max_seq_len", r"max_seq_len=(\d+)", int),
        ("tensor_parallel_size", r"tensor_parallel_size=(\d+)", int),
        ("gpu_kv_cache_tokens", r"GPU KV cache size: ([\d,]+) tokens", lambda g: int(g.replace(",", ""))),
        ("kv_cache_memory_gib", r"Available KV cache memory: ([\d.]+) GiB", float),
        ("gpu_memory_utilization_setting", r"Desired GPU memory utilization is \(([\d.]+),", float),
    ):
        m = _last(pat, seg)
        if m:
            f[key] = conv(m.group(1))
    m = _last(r"Free memory on device \(([\d.]+)/([\d.]+) GiB\)", seg)
    if m:
        f["gpu_memory_free_gib_at_startup"] = float(m.group(1))
        f["gpu_memory_total_gib"] = float(m.group(2))
    return f

def _split_sample(line):
    """Split one Prometheus sample line into (series_id, value_text), or None.
    Handles quoted label values containing '}' or escaped quotes, digits in
    names, and an optional trailing timestamp."""
    m = re.match(r"[A-Za-z_:][A-Za-z0-9_:]*", line)
    if not m:
        return None
    i = m.end()
    if i < len(line) and line[i] == "{":
        j, in_q = i + 1, False
        while j < len(line):
            c = line[j]
            if in_q:
                if c == "\\":
                    j += 1
                elif c == '"':
                    in_q = False
            elif c == '"':
                in_q = True
            elif c == "}":
                break
            j += 1
        if j >= len(line):
            return None
        i = j + 1
    rest = line[i:].split()
    if len(rest) not in (1, 2):
        return None
    return line[:i], rest[0]

USAGE_NAMES = ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc")
PREFIX_PAIRS = (("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"),
                ("vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_hits_total"))
WANTED = USAGE_NAMES + tuple(n for pair in PREFIX_PAIRS for n in pair)

def parse_metrics(text):
    """Return {series_id: float} keeping every label set (no collapsing).
    series_id is the metric name plus its label text, e.g. name{engine="0"}.
    Non-finite values and unparseable JSON raise BadInput."""
    out = {}
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise BadInput("metrics file looks like JSON but is invalid (%s)" % e.msg)
        if not isinstance(data, dict):
            raise BadInput("metrics JSON must be an object of {series: value}")
        items = data.items()
    else:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = _split_sample(line)
            if parsed:
                items.append(parsed)
    for k, v in items:
        name = k.split("{")[0]
        if name not in WANTED:
            continue
        try:
            val = float(v)
        except (TypeError, ValueError):
            raise BadInput("non-numeric value for %s" % k)
        if not math.isfinite(val):
            raise BadInput("non-finite value for %s" % k)
        if k in out:
            raise BadInput("duplicate series %s" % k)
        out[k] = val
    return out

def series_of(mt, name):
    """{label_text: value} for one metric name."""
    return {k[len(name):]: v for k, v in mt.items() if k.split("{")[0] == name}

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

def window_for(mt, mt2):
    """Per-series hit-rate windows between two snapshots. Never collapses label
    sets; any reset, missing counter or inconsistency is INCONCLUSIVE for that series."""
    out = []
    for qn, hn in PREFIX_PAIRS:
        q1, q2 = series_of(mt, qn), series_of(mt2, qn)
        h1, h2 = series_of(mt, hn), series_of(mt2, hn)
        labels = sorted(set(q1) | set(q2) | set(h1) | set(h2))
        for lab in labels:
            row = {"series": qn.split(":")[1].replace("_queries_total", "") + (lab or "")}
            vals = (q1.get(lab), q2.get(lab), h1.get(lab), h2.get(lab))
            if any(v is None for v in vals):
                row.update(status="inconclusive",
                           reason="series missing from a snapshot or from one of the counters "
                                  "(different pod/engine/labels, or counters not exported)")
            else:
                dq, dh = vals[1] - vals[0], vals[3] - vals[2]
                if dq < 0 or dh < 0:
                    row.update(status="inconclusive",
                               reason="counter went down between snapshots (engine restart or a "
                                      "different server); no window claim possible")
                elif dh > dq:
                    row.update(status="inconclusive",
                               reason="hits grew more than queries in the window (%d > %d): "
                                      "inconsistent counters" % (dh, dq))
                elif dq == 0:
                    row.update(status="no_traffic", dq=0, dh=int(dh),
                               reason="no prefix-cache queries between the snapshots")
                else:
                    row.update(status="ok", dq=int(dq), dh=int(dh),
                               hit_pct=round(100.0 * dh / dq, 1))
            out.append(row)
    if not out:
        out.append({"series": "(none)", "status": "inconclusive",
                    "reason": "neither snapshot contains prefix-cache query/hit counters"})
    return out

def report(log_text, metrics_text, mean_prompt_tokens, use_smi, metrics_later_text=None):
    if mean_prompt_tokens is not None and not (isinstance(mean_prompt_tokens, (int, float))
                                               and math.isfinite(mean_prompt_tokens)
                                               and mean_prompt_tokens > 0):
        raise BadInput("--mean-prompt-tokens must be a positive finite number")
    facts, sources, obs, unknowns = {}, {}, [], []
    windows = []

    if log_text is not None:
        lf = parse_log(log_text)
        if len(lf) <= 1:
            raise BadInput("no recognizable vLLM engine settings in --log (is this a vLLM startup log?)")
        for k, v in lf.items():
            facts[k] = v
            sources[k] = "server log"
        if lf.get("engine_inits_in_log", 0) > 1:
            obs.append({"observation": "the log contains %d engine starts; facts describe the LAST one"
                                       % lf["engine_inits_in_log"],
                        "evidence": "multiple 'Initializing a V1 LLM engine' lines",
                        "note": "a restart resets cache counters and contents"})
        if lf.get("engine_inits_in_log", 0) == 0:
            obs.append({"observation": "no engine start line in the log; settings below may be incomplete",
                        "evidence": "no 'Initializing a V1 LLM engine' line found",
                        "note": "pass the startup section of the server log"})
    mt = mt2 = None
    if metrics_text is not None:
        mt = parse_metrics(metrics_text)
        if not mt:
            raise BadInput("no recognized vLLM series in --metrics (expected vllm:* cache metrics)")
        for k, v in sorted(mt.items()):
            facts[k] = v
            sources[k] = "metrics endpoint (snapshot 1)"
    if metrics_later_text is not None:
        if mt is None:
            raise BadInput("--metrics-later needs --metrics (the earlier snapshot)")
        mt2 = parse_metrics(metrics_later_text)
        if not mt2:
            raise BadInput("no recognized vLLM series in --metrics-later")
    if use_smi:
        g = gpu_names()
        facts["nvidia_smi_gpus_this_machine"] = g if g else NA
        sources["nvidia_smi_gpus_this_machine"] = (
            "nvidia-smi on the machine running THIS report - only the model server's "
            "hardware if you run it on the serving host")

    expected = ["vllm_version", "model", "enable_prefix_caching", "dtype",
                "max_seq_len", "gpu_kv_cache_tokens", "kv_cache_memory_gib",
                "gpu_memory_total_gib", "gpu_memory_utilization_setting",
                "tensor_parallel_size"]
    for k in expected:
        if k not in facts:
            facts[k] = NA
            sources[k] = NA
            unknowns.append(k)
    if not use_smi:
        unknowns.append("GPU model name (needs nvidia-smi on the serving host, or platform specs)")

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
                    "evidence": "GPU KV cache size %d tokens vs mean prompt %s tokens (observed / given)"
                                % (facts["gpu_kv_cache_tokens"], mean_prompt_tokens),
                    "note": "a small ratio means new prompts evict old prefixes quickly; "
                            "block granularity and concurrency also matter"})

    if mt is not None:
        n_series = {n: len(series_of(mt, n)) for n in WANTED}
        for name in USAGE_NAMES:
            for lab, v in sorted(series_of(mt, name).items()):
                obs.append({"observation": "KV cache usage was %.0f%% in one snapshot (%s%s)"
                                           % (100 * v, name.split(":")[1], lab),
                            "evidence": "%s%s=%.3f (snapshot 1)" % (name, lab, v),
                            "note": "a single snapshot cannot separate eviction pressure from a "
                                    "cold start or a burst; no conclusion drawn from it"})
        nq = sum(n_series[q] for q, _ in PREFIX_PAIRS)
        if nq > 1:
            obs.append({"observation": "metrics contain %d separate prefix-cache series "
                                       "(several engines/models/labels); they are kept separate" % nq,
                        "evidence": "multiple label sets on vllm:prefix_cache_queries_total",
                        "note": "data-parallel or multi-model servers are not merged into one number"})
        if mt2 is None:
            facts["prefix_cache_counters_note"] = (
                "lifetime counters since engine start are totals, not a hit-rate reading; "
                "a second snapshot (--metrics-later) enables a window")
            unknowns.append("prefix-cache hit rate in a time window (needs --metrics-later)")
        else:
            windows = window_for(mt, mt2)
            for w in windows:
                if w["status"] == "ok":
                    obs.append({"observation": "prefix cache hit rate between the two snapshots was %.1f%% "
                                               "(%d of %d queries) for %s" % (w["hit_pct"], w["dh"], w["dq"], w["series"]),
                                "evidence": "counter delta between snapshot 1 and snapshot 2",
                                "note": "window-based and engine-reported; read it together with the prompt layout"})
                elif w["status"] == "no_traffic":
                    obs.append({"observation": "no prefix-cache queries between the snapshots for %s" % w["series"],
                                "evidence": "query counter delta is 0",
                                "note": "this window carried no traffic; nothing to read"})
                else:
                    unknowns.append("window for %s: INCONCLUSIVE - %s" % (w["series"], w["reason"]))
    if mt is not None and log_text is not None:
        names = set(re.findall(r'model_name="([^"]*)"', " ".join(mt)))
        known = {facts.get("served_model_name"), facts.get("model")} - {None, NA}
        if names and known and not (names & known):
            obs.append({"observation": "WARNING: model name in metrics (%s) does not match the log (%s)"
                                       % (", ".join(sorted(names)), ", ".join(sorted(known))),
                        "evidence": "model_name label vs served_model_name/model in the log",
                        "note": "the log and metrics may come from different servers; do not combine them"})
    if use_smi and log_text is not None:
        obs.append({"observation": "GPU names come from the machine running this report, the log from a file",
                    "evidence": "--with-nvidia-smi together with --log",
                    "note": "only the same host if you ran this on the serving machine; otherwise ignore the GPU names"})
    if not obs:
        obs.append({"observation": "no cache-relevant signal in the provided files",
                    "evidence": "parsed fields are all 'not available' or neutral",
                    "note": "provide the server startup log and/or a /metrics snapshot"})
    return {"facts": facts, "sources": sources, "windows": windows,
            "observations": obs, "unknowns": unknowns,
            "scope_note": ("facts are parsed from the files/settings given; 'not available' "
                           "means the file did not show it. This report does not probe any "
                           "host and does not claim a hardware bottleneck from metadata. "
                           "Parsing targets vLLM's V1 log lines and metric names as seen in "
                           "ONE tested build (0.26.1rc1); other builds may differ.")}

def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="CacheLens serving-environment report")
    ap.add_argument("--log"); ap.add_argument("--metrics"); ap.add_argument("--metrics-later")
    ap.add_argument("--mean-prompt-tokens", type=float)
    ap.add_argument("--with-nvidia-smi", action="store_true")
    a = ap.parse_args(argv)
    log_text = metrics_text = metrics_later = None
    try:
        if a.log:
            log_text = open(a.log, encoding="utf-8").read()
        if a.metrics:
            metrics_text = open(a.metrics, encoding="utf-8").read()
        if a.metrics_later:
            metrics_later = open(a.metrics_later, encoding="utf-8").read()
    except OSError as e:
        print(f"INCONCLUSIVE - cannot read input: {e.strerror or e}")
        return 2
    except UnicodeDecodeError:
        print("INCONCLUSIVE - input is not valid UTF-8 text")
        return 2
    if log_text is None and metrics_text is None and not a.with_nvidia_smi:
        print("INCONCLUSIVE - give --log and/or --metrics (or --with-nvidia-smi on the server host)")
        return 2
    try:
        res = report(log_text, metrics_text, a.mean_prompt_tokens, a.with_nvidia_smi,
                     metrics_later_text=metrics_later)
    except BadInput as e:
        print(f"INCONCLUSIVE - {e}")
        return 2
    print("== CacheLens serving-environment report ==")
    print(json.dumps(res, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
