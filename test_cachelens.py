"""Local tests for CacheLens - no GPU or network required.
Run: python3 test_cachelens.py
"""
import json, os, subprocess, sys, tempfile

import analyze
from workload import build_requests

T = tempfile.mkdtemp(prefix="cachelens-test-")
fails = []

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

def write_metrics(path, qb, qa, hb, ha, key='{engine="0",model_name="muse-glimmer"}'):
    json.dump({"before": {"vllm:prefix_cache_queries_total" + key: str(qb),
                          "vllm:prefix_cache_hits_total" + key: str(hb)},
               "after":  {"vllm:prefix_cache_queries_total" + key: str(qa),
                          "vllm:prefix_cache_hits_total" + key: str(ha)}},
              open(path, "w"))

# 1. null cached_tokens stay UNKNOWN, never 0
rows = [{"prompt_tokens": 10, "cached_tokens": None} for _ in range(3)]
s = analyze.summarize(rows)
check("null cached_tokens reported unavailable, not zero",
      "unavailable" in s.get("per_request_cached_tokens", "") and "total_cached_tokens" not in s)

# 2. divergence examined across ALL requests (a third prompt diverging earlier
#    must pull the reported offset down)
prompts = [{"prompt": p, "prompt_tokens": 1, "cached_tokens": None} for p in
           ["AAAAAxBBBB", "AAAAAyBBBB", "qZZZZZZZZZ"]]
f = analyze.diagnose(prompts)[0]
check("all-prompt divergence finds earliest offset (0, not 5)", f["diverges_at_char"] == 0)

# 3. counter reset rejected
p = os.path.join(T, "reset_metrics.json"); write_metrics(p, 100, 50, 0, 0)
try:
    analyze.metrics_delta(p); check("counter reset rejected", False)
except analyze.InvalidEvidence: check("counter reset rejected", True)

# 4. hits > queries rejected
p = os.path.join(T, "hq_metrics.json"); write_metrics(p, 0, 100, 0, 200)
try:
    analyze.metrics_delta(p); check("hits>queries rejected", False)
except analyze.InvalidEvidence: check("hits>queries rejected", True)

# 5. non-finite rejected
p = os.path.join(T, "nan_metrics.json"); write_metrics(p, 0, float("nan"), 0, 0)
try:
    analyze.metrics_delta(p); check("non-finite counter rejected", False)
except analyze.InvalidEvidence: check("non-finite counter rejected", True)

# 6. unavailable metrics snapshot rejected
p = os.path.join(T, "err_metrics.json")
json.dump({"before": {"error": "metrics unavailable: boom"}, "after": {}}, open(p, "w"))
try:
    analyze.metrics_delta(p); check("unavailable snapshot rejected", False)
except analyze.InvalidEvidence: check("unavailable snapshot rejected", True)

# 7. unequal prompt volumes -> INCONCLUSIVE (rc 2) on real-shaped rows
def fake_rows(variant, n, ptoks):
    return [{"id": f"req-{i}", "variant": variant, "prompt": "x" * 100,
             "prompt_tokens": ptoks, "cached_tokens": None, "error": None}
            for i in range(n)]
d1, d2 = os.path.join(T, "v1"), os.path.join(T, "v2"); os.makedirs(d1); os.makedirs(d2)
for d, v, pt in ((d1, "ts_before", 10), (d2, "ts_after", 10)):
    with open(os.path.join(d, f"{v}.jsonl"), "w") as fh:
        for r in fake_rows(v, 18, pt): fh.write(json.dumps(r) + "\n")
write_metrics(os.path.join(d1, "ts_before_metrics.json"), 0, 180, 0, 2)
write_metrics(os.path.join(d2, "ts_after_metrics.json"), 0, 90, 0, 85)  # half volume
rc = subprocess.run([sys.executable, "analyze.py",
                     os.path.join(d1, "ts_before.jsonl"),
                     os.path.join(d2, "ts_after.jsonl")],
                    capture_output=True, text=True).returncode
check("unequal volumes -> INCONCLUSIVE rc=2", rc == 2)

# 8. missing metrics file -> analyzer failure rc=2, no traceback
r = subprocess.run([sys.executable, "analyze.py",
                    os.path.join(d1, "ts_before.jsonl"),
                    os.path.join(d2, "ts_after.jsonl").replace("ts_after", "nope")],
                   capture_output=True, text=True)
check("missing evidence file handled", r.returncode != 0 and "Traceback" not in r.stderr)

# 9a. counterexamples on the timestamp-move validator
def paired_rows(variant, n, ptoks, idprefix="req-", tstamp=None, layout="move"):
    rows = []
    for i in range(n):
        t = tstamp or f"2026-11-04T09:{15+i:02d}:00+05:30"
        f_ = f"[request_time={t}]"
        if layout == "move":
            p = (f_ + "\n" + "x" * 100) if variant == "ts_before" else ("x" * 100 + "\n" + f_)
        elif layout == "identical":
            p = "x" * 100 + "\n" + f_
        elif layout == "nofield":
            p = "x" * 110
        rows.append({"id": f"{idprefix}{i}", "variant": variant, "prompt": p,
                     "prompt_tokens": ptoks, "cached_tokens": None, "error": None})
    return rows

def run_pair(rows_b, rows_a, qb=1800, hb=10, qa=1800, ha=1700):
    d3, d4 = os.path.join(T, "vx"), os.path.join(T, "vy")
    os.makedirs(d3, exist_ok=True); os.makedirs(d4, exist_ok=True)
    for d, v, rows in ((d3, "ts_before", rows_b), (d4, "ts_after", rows_a)):
        with open(os.path.join(d, f"{v}.jsonl"), "w") as fh:
            for r in rows: fh.write(json.dumps(r) + "\n")
    write_metrics(os.path.join(d3, "ts_before_metrics.json"), 0, qb, 0, hb)
    write_metrics(os.path.join(d4, "ts_after_metrics.json"), 0, qa, 0, ha)
    return subprocess.run([sys.executable, "analyze.py",
                           os.path.join(d3, "ts_before.jsonl"),
                           os.path.join(d4, "ts_after.jsonl")],
                          capture_output=True, text=True)

# truly identical prompts (same layout both files) -> INCONCLUSIVE
rc = run_pair(paired_rows("ts_before", 18, 100, layout="identical"),
              paired_rows("ts_after", 18, 100, layout="identical"))
check("truly identical prompts -> INCONCLUSIVE", rc.returncode == 2 and "identical" in rc.stdout)

# no timestamp field at all -> INCONCLUSIVE
rc = run_pair(paired_rows("ts_before", 18, 100, layout="nofield"),
              paired_rows("ts_after", 18, 100, layout="nofield"))
check("no timestamp field -> INCONCLUSIVE", rc.returncode == 2)

# altered timestamp value between variants -> INCONCLUSIVE
rc = run_pair(paired_rows("ts_before", 18, 100),
              paired_rows("ts_after", 18, 100, tstamp="2020-01-01T00:00:00+05:30"))
check("altered timestamp value -> INCONCLUSIVE", rc.returncode == 2 and "timestamp value differs" in rc.stdout)

# valid move -> OBSERVED INCREASE rc 0
rc = run_pair(paired_rows("ts_before", 18, 100), paired_rows("ts_after", 18, 100))
check("valid timestamp move -> OBSERVED INCREASE", rc.returncode == 0 and "OBSERVED INCREASE" in rc.stdout)

# 9b. unmatched ids -> INCONCLUSIVE
rc = run_pair(paired_rows("ts_before", 18, 100, idprefix="req-"),
              paired_rows("ts_after", 18, 100, idprefix="other-"))
check("unmatched request ids -> INCONCLUSIVE", rc.returncode == 2 and "ids" in rc.stdout)

# 9c. captured usage sum != engine queried -> INCONCLUSIVE
rc = run_pair(paired_rows("ts_before", 18, 100), paired_rows("ts_after", 18, 100),
              qb=1000)
check("usage/counter mismatch -> INCONCLUSIVE", rc.returncode == 2 and "captured usage" in rc.stdout)

# 9d. different models between windows -> INCONCLUSIVE
d3, d4 = os.path.join(T, "vz"), os.path.join(T, "vw")
os.makedirs(d3, exist_ok=True); os.makedirs(d4, exist_ok=True)
for d, v in ((d3, "ts_before"), (d4, "ts_after")):
    with open(os.path.join(d, f"{v}.jsonl"), "w") as fh:
        for r in paired_rows(v, 18, 100): fh.write(json.dumps(r) + "\n")
write_metrics(os.path.join(d3, "ts_before_metrics.json"), 0, 1800, 0, 10)
write_metrics(os.path.join(d4, "ts_after_metrics.json"), 0, 1800, 0, 1700,
              key='{engine="0",model_name="other-model"}')
rc = subprocess.run([sys.executable, "analyze.py",
                     os.path.join(d3, "ts_before.jsonl"),
                     os.path.join(d4, "ts_after.jsonl")],
                    capture_output=True, text=True)
check("different model series across windows -> INCONCLUSIVE", rc.returncode == 2)

# 9e. probe refuses to overwrite existing evidence
out = os.path.join(T, "exists.jsonl"); open(out, "w").write("{}\n")
r = subprocess.run([sys.executable, "probe.py", "--variant", "ts_before",
                    "--out", out, "--server", "http://127.0.0.1:1"],
                   capture_output=True, text=True)
check("probe refuses to overwrite evidence", "refusing to overwrite" in (r.stdout + r.stderr))

# 9f. runner static checks: host pinned + cleanup in finally
src = open("modal_serve.py").read()
check("runner pins vLLM to 127.0.0.1", '"--host", "127.0.0.1"' in src)
check("runner terminates server in finally", "finally:" in src and "srv.terminate()" in src)
check("runner checks analyzer return code", "r.returncode not in (0, 1)" in src)

# 10. workload shapes: 18 requests, single user role, variants differ as designed
rs = build_requests(18)
check("18 requests built", len(rs) == 18)
check("ts_before puts timestamp first", rs[0]["variants"]["ts_before"].startswith("[request_time="))
check("ts_after puts shared text first", rs[0]["variants"]["ts_after"].startswith("You are FieldOps"))

# strict-prefix regression: "ab" vs "abc" must not raise and reports 2
f = analyze.diagnose([{"prompt": "abc", "prompt_tokens": 1, "cached_tokens": None},
                      {"prompt": "ab", "prompt_tokens": 1, "cached_tokens": None}])[0]
check("strict-prefix inputs (ab/abc) handled, common prefix 2", f["diverges_at_char"] == 2)

print()
n = sum(1 for line in open(__file__).read().splitlines() if line.strip().startswith("check("))
print(f"{n} checks; " + ("FAILED: " + ", ".join(fails) if fails else "ALL TESTS PASSED"))
sys.exit(1 if fails else 0)
