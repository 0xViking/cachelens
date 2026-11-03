"""Local tests for CacheLens - no GPU or network required.
Run: python3 test_cachelens.py
"""
import json, os, subprocess, sys, tempfile

import analyze
from workload import build_requests

T = tempfile.mkdtemp(prefix="cachelens-test-")
fails = []
total = 0

def check(name, cond):
    global total
    total += 1
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


# 11. prompt_diag: generic prompt-file diagnostic (text level)
import prompt_diag

def pd_run(path):
    return subprocess.run([sys.executable, "analyze.py", "--prompts", path],
                          capture_output=True, text=True)

def pd_write(name, objs):
    f = os.path.join(T, name)
    with open(f, "w") as fh:
        for o in objs: fh.write(json.dumps(o) + "\n")
    return f

# 11a. bad JSON line -> INCONCLUSIVE rc 2, no traceback
f = os.path.join(T, "bad.jsonl"); open(f, "w").write('{"prompt": "x"}\nnot json\n')
r = pd_run(f)
check("prompt_diag bad JSON -> INCONCLUSIVE rc=2, no traceback",
      r.returncode == 2 and "INCONCLUSIVE" in r.stdout and "Traceback" not in r.stderr)

# 11b. fewer than 2 prompts -> INCONCLUSIVE
r = pd_run(pd_write("one.jsonl", [{"prompt": "only one"}]))
check("prompt_diag single prompt -> INCONCLUSIVE", r.returncode == 2 and "INCONCLUSIVE" in r.stdout)

# 11c. missing file -> INCONCLUSIVE
r = pd_run(os.path.join(T, "does-not-exist.jsonl"))
check("prompt_diag missing file -> INCONCLUSIVE", r.returncode == 2)

# 11d. identical prompts -> finding identical, no blame on layout
res = prompt_diag.diagnose(["same text\nmore"] * 4)
check("prompt_diag identical prompts -> finding identical",
      res["finding"] == "identical" and "server flags" in res["verdict"])

# 11e. early variation: distinct-value count at divergence line is exact
res = prompt_diag.diagnose(["head\nAAA\n" + "body" * 50, "head\nBBB\n" + "body" * 50])
check("prompt_diag counts distinct values at divergence line",
      res["first_divergence"]["distinct_values_at_that_line"] == 2)
check("prompt_diag reports stable text after divergence",
      res["stable_text_after_first_divergence"]["pct_of_first_prompt"] > 80)

# 11f. counterexample: low-cardinality mid-prompt field is not called per-request-unique
res = prompt_diag.diagnose(["sys\n[user-1]\n" + "p" * 200, "sys\n[user-2]\n" + "p" * 200,
                            "sys\n[user-1]\n" + "p" * 200])
note = res["varying_lines_in_first_prompt"][0]["cardinality_note"]
check("prompt_diag distinguishes low-cardinality field from per-request id",
      "small set" in note)

# 11g. counterexample: reordered lines detected only when line SETS are identical
res = prompt_diag.diagnose(["a\nb\nc", "c\nb\na"])
check("prompt_diag reordered-lines finding", res["finding"] == "reordered-lines")
res = prompt_diag.diagnose(["a\nb\nc", "c\nb\nDIFFERENT"])
check("prompt_diag reorder not claimed when content also differs",
      res["finding"] != "reordered-lines")

# 11h. late variation -> layout not blamed
res = prompt_diag.diagnose(["x" * 500 + "\ntail A", "x" * 500 + "\ntail B"])
check("prompt_diag late-variation finding", res["finding"] == "late-variation")

# 11i. messages-mode input renders and diagnoses
res = prompt_diag.diagnose(["system: s\nuser: u1", "system: s\nuser: u2"])
check("prompt_diag messages-rendered text works", res["first_divergence"]["line_index"] == 1)
r = pd_run(pd_write("msgs.jsonl", [{"messages": [{"role": "system", "content": "s"},
                                               {"role": "user", "content": "u1"}]},
                                   {"messages": [{"role": "system", "content": "s"},
                                               {"role": "user", "content": "u2"}]}]))
check("prompt_diag messages file rc=0", r.returncode == 0)

# 11j. empty prompt and malformed message rows fail closed
try:
    prompt_diag.diagnose(["text", ""]); check("prompt_diag empty prompt rejected", False)
except prompt_diag.BadInput: check("prompt_diag empty prompt rejected", True)
try:
    prompt_diag.load(pd_write("badmsg.jsonl", [{"messages": [{"role": "user"}]}]))
    check("prompt_diag malformed message rejected", False)
except prompt_diag.BadInput: check("prompt_diag malformed message rejected", True)

# 11k. strict-prefix prompts (one is a prefix of the other) handled
res = prompt_diag.diagnose(["ab\ncd", "ab"])
check("prompt_diag strict-prefix pair handled", res["leading_shared_chars"] == 2)

# 11l. real example files ship findings as documented
for ex, want in (("examples/user_id_midprompt.jsonl", "early-variation"),
                 ("examples/reordered_tool_lines.jsonl", "reordered-lines")):
    r = pd_run(ex)
    check(f"prompt_diag {ex} -> {want}", r.returncode == 0 and f'"finding": "{want}"' in r.stdout)

# 11m. no engine prediction: output never claims a hit rate or token figure
r = pd_run("examples/user_id_midprompt.jsonl")
check("prompt_diag states text-level scope, no hit-rate claim",
      "text-level" in r.stdout and "hit rate" in r.stdout and "token" not in r.stdout.lower().replace("not tokens", ""))


# 12. env_report: serving-environment collector (local files only)
import env_report

def er_main(args):
    return subprocess.run([sys.executable, "env_report.py"] + args,
                          capture_output=True, text=True)

# 12a. real Glimmer log parses to the verified values
res = env_report.report(open("evidence/vllm.log", errors="replace").read(), None, None, False)
check("env_report parses real vllm.log facts",
      res["facts"]["gpu_kv_cache_tokens"] == 776150 and
      res["facts"]["enable_prefix_caching"] is True and
      res["facts"]["gpu_memory_total_gib"] == 79.27 and
      res["sources"]["gpu_kv_cache_tokens"] == "server log")

# 12b. missing fields are 'not available', never guessed
# partial log: only some settings present -> the rest stay 'not available' and are listed unknown
res = env_report.report("Initializing a V1 LLM engine (v0.26.1) with config: model='m/x', dtype=torch.bfloat16\n",
                        None, None, False)
check("env_report partial log: missing fields are not available and listed as unknown",
      res["facts"]["enable_prefix_caching"] == "not available" and
      "enable_prefix_caching" in res["unknowns"] and res["facts"]["dtype"] == "bfloat16")
try:
    env_report.report("some random log line\n", None, None, False)
    check("env_report unrecognized log fails closed", False)
except env_report.BadInput: check("env_report unrecognized log fails closed", True)

# 12c. prefix caching disabled is called out from config evidence
res = env_report.report(open("examples/env/vllm_prefix_off.log").read(), None, None, False)
check("env_report flags enable_prefix_caching=False",
      any("disabled" in o["observation"] for o in res["observations"]) and
      res["facts"]["tensor_parallel_size"] == 2)

# 12d. single metrics snapshot: usage is a fact, NO eviction/hit-rate claim
res = env_report.report(None, open("examples/env/metrics_snap1.txt").read(), None, False)
text = json.dumps(res)
check("env_report single snapshot makes no eviction/hit-rate claim",
      res["windows"] == [] and "eviction" not in text.lower().replace(
          "cannot separate eviction pressure from a cold start or a burst", "").replace(
          "layout, capacity, eviction or routing", "") and
      "lifetime counters" in res["facts"].get("prefix_cache_counters_note", ""))

# 12e. two snapshots: window hit rate is counter arithmetic only
res = env_report.report(None, open("examples/env/metrics_snap1.txt").read(), None, False,
                        metrics_later_text=open("examples/env/metrics_snap2.txt").read())
check("env_report window hit rate = 50/1000 = 5.0%",
      [w.get("hit_pct") for w in res["windows"]] == [5.0])

# 12f. counterexample: zero query delta -> no hit-rate reading
res = env_report.report(None, open("examples/env/metrics_snap1.txt").read(), None, False,
                        metrics_later_text=open("examples/env/metrics_snap1.txt").read())
check("env_report zero-delta window says no traffic, not 0% hits",
      res["windows"][0]["status"] == "no_traffic" and "hit_pct" not in res["windows"][0] and
      any("no prefix-cache queries" in o["observation"] for o in res["observations"]))

# 12g. capacity ratio only when caller supplies mean prompt tokens
res = env_report.report(open("evidence/vllm.log", errors="replace").read(), None, 5796, False)
check("env_report capacity ratio 776150/5796 ~ 133.9",
      any("133.9" in o["observation"] for o in res["observations"]))
res = env_report.report(open("evidence/vllm.log", errors="replace").read(), None, None, False)
check("env_report no capacity ratio without mean prompt tokens",
      not any("mean-size prompts" in o["observation"] for o in res["observations"]))

# 12h. garbage input -> INCONCLUSIVE rc 2
r = er_main(["--log", os.path.join(T, "bad.jsonl")])  # 'not json' content, no vllm facts
check("env_report unparseable input -> INCONCLUSIVE rc=2",
      r.returncode == 2 and "INCONCLUSIVE" in r.stdout)
r = er_main([])
check("env_report no input -> INCONCLUSIVE rc=2", r.returncode == 2)
r = er_main(["--log", os.path.join(T, "nope.log")])
check("env_report missing file -> INCONCLUSIVE rc=2", r.returncode == 2)

# 12i. labeled example log yields its documented finding via CLI
r = er_main(["--log", "examples/env/vllm_prefix_off.log"])
check("env_report CLI example: prefix-off log flagged",
      r.returncode == 0 and "prefix caching is disabled" in r.stdout)

# 13. nasty counterexamples for env_report (review findings B1-B4, S1-S6) and prompt_diag (P1,P2,P6)
Q, H, U = "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:gpu_cache_usage_perc"
def mtxt(q, h, label='{engine="0",model_name="m"}', extra=""):
    return f"{Q}{label} {q}\n{H}{label} {h}\n{extra}"

def win(a, b):
    return env_report.report(None, a, None, False, metrics_later_text=b)["windows"]

# B1 counter reset / missing counters never read as 'no traffic' or a rate
w = win(mtxt(1000, 100), mtxt(100, 10))
check("env_report counter reset -> inconclusive, no rate", w[0]["status"] == "inconclusive" and "hit_pct" not in w[0])
w = win(mtxt(1000, 100), f"{U}{{engine=\"0\"}} 0.5\n")
check("env_report counters missing in snapshot 2 -> inconclusive",
      all(x["status"] == "inconclusive" for x in w))
# B2 hits > queries in window rejected (no 500%)
w = win(mtxt(100, 0), mtxt(110, 50))
check("env_report hits>queries window -> inconclusive", w[0]["status"] == "inconclusive" and "inconsistent" in w[0]["reason"])
# B3 two engines kept separate, never collapsed
two = (mtxt(1000, 100, '{engine="0"}') + mtxt(500, 50, '{engine="1"}'))
two2 = (mtxt(2000, 600, '{engine="0"}') + mtxt(600, 51, '{engine="1"}'))
w = win(two, two2)
check("env_report two engines -> two separate windows (50.0% and 1.0%)",
      sorted(x["hit_pct"] for x in w) == [1.0, 50.0] and len(w) == 2)
# B3 different series set across snapshots (different pod/labels) -> inconclusive per series
w = win(mtxt(1000, 100, '{engine="0"}'), mtxt(2000, 600, '{engine="9"}'))
check("env_report different label sets across snapshots -> inconclusive",
      all(x["status"] == "inconclusive" for x in w))
# B4 label value containing '}' and trailing timestamp both parse
t = 'vllm:prefix_cache_queries_total{engine="0",model_name="a}b"} 7 1700000000\n'
mt = env_report.parse_metrics(t)
check("env_report parses brace-in-label and trailing timestamp", list(mt.values()) == [7.0])
mt = env_report.parse_metrics('vllm:gpu_prefix_cache_hits_total{e="0"} 3\nvllm:kv_cache_usage_perc{e="0"} 0.4\n')
check("env_report parses alias metric names", len(mt) == 2)
res = env_report.report(None, 'vllm:kv_cache_usage_perc{engine="0"} 0.40\n', None, False)
check("env_report newer kv_cache_usage_perc name is used",
      any("KV cache usage was 40%" in o["observation"] for o in res["observations"]))
# S1 log passed as metrics -> INCONCLUSIVE rc 2; metrics passed as log too
r = er_main(["--metrics", "evidence/vllm.log"])
check("env_report log-as-metrics -> INCONCLUSIVE rc=2", r.returncode == 2 and "INCONCLUSIVE" in r.stdout)
r = er_main(["--log", "examples/env/metrics_snap1.txt"])
check("env_report metrics-as-log -> INCONCLUSIVE rc=2", r.returncode == 2)
# S2 two engine starts: last one wins, warning shown
two_logs = ("Initializing a V1 LLM engine (v0.10.0) with config: model='old/m', enable_prefix_caching=False\n"
            "Initializing a V1 LLM engine (v0.26.1) with config: model='new/m', enable_prefix_caching=True\n")
res = env_report.report(two_logs, None, None, False)
check("env_report two engine starts: last wins + warning",
      res["facts"]["vllm_version"] == "0.26.1" and res["facts"]["enable_prefix_caching"] is True and
      any("2 engine starts" in o["observation"] for o in res["observations"]))
# S3 model name mismatch warned
res = env_report.report("Initializing a V1 LLM engine (v1) with config: model='a/b', served_model_name=svc-a, enable_prefix_caching=True\n",
                        'vllm:gpu_cache_usage_perc{model_name="svc-z"} 0.1\n', None, False)
check("env_report model-name mismatch warned",
      any("does not match" in o["observation"] for o in res["observations"]))
# S4 non-positive / non-finite mean prompt tokens rejected
for bad in (-5, 0, float("nan"), float("inf")):
    try:
        env_report.report(open("evidence/vllm.log", errors="replace").read(), None, bad, False)
        check(f"env_report mean_prompt_tokens={bad} rejected", False)
    except env_report.BadInput: check(f"env_report mean_prompt_tokens={bad} rejected", True)
r = er_main(["--log", "evidence/vllm.log", "--mean-prompt-tokens", "-5"])
check("env_report CLI negative mean -> INCONCLUSIVE rc=2", r.returncode == 2)
# S6 invalid JSON / JSON array / non-finite / non-UTF8 -> INCONCLUSIVE, no traceback
for name, content in (("bad.json", '{"a": '), ("arr.json", "[1,2]"),
                      ("nan.json", '{"vllm:prefix_cache_queries_total{e=\\"0\\"}": NaN}')):
    f = os.path.join(T, name); open(f, "w").write(content)
    r = er_main(["--metrics", f])
    check(f"env_report {name} -> INCONCLUSIVE rc=2 no traceback",
          r.returncode == 2 and "Traceback" not in r.stderr)
f = os.path.join(T, "binary.log"); open(f, "wb").write(b"\xff\xfe\x80\x81 not utf8 \xff")
r = er_main(["--log", f])
check("env_report non-UTF8 log -> INCONCLUSIVE rc=2 no traceback", r.returncode == 2 and "Traceback" not in r.stderr)
r = pd_run(f)
check("prompt_diag non-UTF8 file -> INCONCLUSIVE rc=2 no traceback", r.returncode == 2 and "Traceback" not in r.stderr)
# metrics-later without metrics rejected
try:
    env_report.report(None, None, None, False, metrics_later_text=mtxt(1, 1)); check("env_report later without first rejected", False)
except env_report.BadInput: check("env_report later without first rejected", True)

# P1 content parts + BOM supported; non-text parts rejected clearly
rows = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "hello A"}]}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello B"}]}]}]
f = os.path.join(T, "parts.jsonl")
with open(f, "w", encoding="utf-8-sig") as fh:
    for o in rows: fh.write(json.dumps(o) + "\n")
r = pd_run(f)
check("prompt_diag text content parts + UTF-8 BOM supported", r.returncode == 0)
f = pd_write("img.jsonl", [{"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
                           {"messages": [{"role": "user", "content": "t"}]}])
r = pd_run(f)
check("prompt_diag non-text content part -> INCONCLUSIVE", r.returncode == 2 and "text content parts" in r.stdout)
# P2 --no-examples redacts prompt fragments
f = pd_write("secret.jsonl", [{"prompt": "head\nSECRET-KEY sk-abc123\n" + "z" * 80},
                              {"prompt": "head\nSECRET-KEY sk-zzz999\n" + "z" * 80}])
r = subprocess.run([sys.executable, "analyze.py", "--prompts", f, "--no-examples"], capture_output=True, text=True)
check("prompt_diag --no-examples leaves no prompt text in output",
      r.returncode == 0 and "sk-abc123" not in r.stdout and "SECRET" not in r.stdout and "redacted" in r.stdout)
r = pd_run(f)
check("prompt_diag default output DOES echo fragments (documented)", "SECRET-KEY" in r.stdout)
# P6 two differing prompts are not called a per-request id
res = prompt_diag.diagnose(["h\nA\n" + "p" * 100, "h\nB\n" + "p" * 100])
note = res["varying_lines_in_first_prompt"][0]["cardinality_note"]
check("prompt_diag small-n does not claim per-request id", "does not show a per-request id" in note)
# P4 heuristic is labeled
res = prompt_diag.diagnose(["a" * 10 + "\nX" + "q" * 200, "a" * 10 + "\nY" + "q" * 200])
check("prompt_diag early-variation labeled heuristic", "heuristic" in res)

# 14. reviewer round 2 + combined report (cachelens.py)
# all windows inconclusive -> rc 2 visible INCONCLUSIVE; no misleading "no signal" observation
f1 = os.path.join(T, "m_a.txt"); f2 = os.path.join(T, "m_b.txt")
open(f1, "w").write(mtxt(1000, 100)); open(f2, "w").write(mtxt(100, 10))   # reset
r = er_main(["--metrics", f1, "--metrics-later", f2])
check("env_report all windows inconclusive (reset) -> rc=2 INCONCLUSIVE",
      r.returncode == 2 and "INCONCLUSIVE" in r.stdout and "went down" in r.stdout)
res = env_report.report(None, mtxt(1000, 100), None, False, metrics_later_text=mtxt(100, 10))
check("env_report no misleading 'no cache-relevant signal' when counters were parsed",
      not any("no cache-relevant signal" in o["observation"] for o in res["observations"]))
# identical snapshots (same file twice) are called out
res = env_report.report(None, mtxt(5, 1), None, False, metrics_later_text=mtxt(5, 1))
check("env_report identical snapshots flagged as same file / stalled",
      "identical" in res["windows"][0]["reason"])
# --no-examples works in any argument position
f = pd_write("secret2.jsonl", [{"prompt": "h\nSECRET-A\n" + "z" * 50}, {"prompt": "h\nSECRET-B\n" + "z" * 50}])
r = subprocess.run([sys.executable, "analyze.py", "--no-examples", "--prompts", f], capture_output=True, text=True)
check("analyze.py --no-examples before --prompts also redacts", r.returncode == 0 and "SECRET" not in r.stdout)

def cl(args):
    return subprocess.run([sys.executable, "cachelens.py"] + args, capture_output=True, text=True)
secret = pd_write("secret3.jsonl", [{"prompt": "h\nSECRET-KEY sk-live-111 <script>alert(1)</script>\n" + "z" * 80},
                                    {"prompt": "h\nSECRET-KEY sk-live-222 <script>alert(1)</script>\n" + "z" * 80}])
o1 = os.path.join(T, "rep_default")
r = cl(["--prompts", secret, "--out", o1])
html1 = open(os.path.join(o1, "report.html")).read(); json1 = open(os.path.join(o1, "report.json")).read()
check("cachelens report default: no prompt text in HTML or JSON",
      r.returncode == 0 and "SECRET" not in html1 and "sk-live" not in html1 and "SECRET" not in json1)
o2 = os.path.join(T, "rep_show")
r = cl(["--prompts", secret, "--show-examples", "--out", o2])
html2 = open(os.path.join(o2, "report.html")).read()
check("cachelens --show-examples includes fragments, HTML-escaped, with warning",
      "sk-live-111" in html2 and "<script>alert" not in html2 and "&lt;script&gt;" in html2 and "Do not share" in html2)
import re as _re
check("cachelens HTML is self-contained (no external URLs, no scripts)",
      not _re.findall(r"(?:src|href)=[\"']https?:", html1) and "<script" not in html1 and "<link" not in html1)
# full real example renders every section with real measured numbers
o3 = os.path.join(T, "rep_full")
r = cl(["--prompts", "evidence/ts_before.jsonl", "--log", "evidence/vllm.log",
        "--pair-before", "evidence/ts_before.jsonl", "--pair-after", "evidence/ts_after.jsonl", "--out", o3])
html3 = open(os.path.join(o3, "report.html")).read()
check("cachelens full report: measured 1.0% -> 93.6%, break at char 28, KV 776150",
      r.returncode == 0 and "93.6%" in html3 and "1088" in html3 and "char 28" in html3 and "776150" in html3)
# missing sections are visibly unknown
check("cachelens missing sections show 'unknown'",
      html1.count("unknown") >= 2 and "Not provided" in html1)
# fail closed: no input, bad input, half a pair, bad pair
r = cl([]); check("cachelens no input -> INCONCLUSIVE rc=2", r.returncode == 2)
r = cl(["--prompts", os.path.join(T, "bad.jsonl")]); check("cachelens bad prompts -> INCONCLUSIVE rc=2", r.returncode == 2 and "Traceback" not in r.stderr)
r = cl(["--pair-before", "evidence/ts_before.jsonl"]); check("cachelens half a pair -> INCONCLUSIVE rc=2", r.returncode == 2)
r = cl(["--pair-before", os.path.join(d1, "ts_before.jsonl"), "--pair-after", os.path.join(d2, "ts_after.jsonl")])
check("cachelens inconclusive measured pair -> rc=2 not a report", r.returncode == 2 and "INCONCLUSIVE" in r.stdout)
# reset windows: report still renders with visible banner
o4 = os.path.join(T, "rep_reset")
r = cl(["--metrics", f1, "--metrics-later", f2, "--out", o4])
html4 = open(os.path.join(o4, "report.html")).read()
check("cachelens all-inconclusive windows: visible INCONCLUSIVE banner, no rate",
      r.returncode == 0 and "INCONCLUSIVE:" in html4 and "hit rate" not in html4.split("Findings")[0].lower().replace("window hit rate", ""))

# 15. report provenance, conflicts, synthetic labeling, caveats (main's review of the rendered pages)
ex_args = ["--prompts", "examples/user_id_midprompt.jsonl", "--log", "examples/env/vllm_prefix_off.log",
           "--metrics", "examples/env/metrics_snap1.txt", "--metrics-later", "examples/env/metrics_snap2.txt"]
o5 = os.path.join(T, "rep_conflict"); r = cl(ex_args + ["--out", o5])
h5 = open(os.path.join(o5, "report.html")).read()
j5 = json.load(open(os.path.join(o5, "report.json")))
check("report: cache-off log + hits window -> INPUTS DISAGREE flag, no cross-section cause row",
      "INPUTS DISAGREE" in h5 and "Early layout break AND" not in h5 and "COMBINED" not in h5)
check("report: examples/ inputs are labeled SYNTHETIC EXAMPLE DATA and flagged in JSON",
      "SYNTHETIC EXAMPLE DATA" in h5 and j5["synthetic"] is True)
# real-evidence run is not labeled synthetic; measured caveat names same-server/no cache clear
check("report: real evidence run not labeled synthetic; measured caveat says same-instance NOT verified, next to numbers",
      "SYNTHETIC EXAMPLE DATA" not in html3 and "Not verified here:" in html3 and "can repeat across restarts" in html3
      and html3.index("Not verified here:") < html3.index("Tokens queried") and "18 and 18 requests" in html3)
# legends: not 'reusable across requests'; explicit ALL prompts + measurement caveat
check("report legend: shared by ALL prompts, character-level, needs measurement",
      "shared by ALL" in html3 and "needs measurement" in html3 and "not reusable across these requests" not in html3)
# no causal cross-section row ever; inputs-not-verified note present when several sections
mA = os.path.join(T, "mm1.txt"); mB = os.path.join(T, "mm2.txt")
open(mA, "w").write(mtxt(1000, 100)); open(mB, "w").write(mtxt(2000, 150))
o6 = os.path.join(T, "rep_nolink"); cl(["--prompts", secret, "--metrics", mA, "--metrics-later", mB, "--out", o6])
h6 = open(os.path.join(o6, "report.html")).read()
check("report: prompt file + counters -> 'not verified same server' note, no cross-reading row",
      "Inputs are not verified to describe the same server or traffic" in h6 and "Early layout break AND" not in h6
      and "COMBINED" not in h6)
r = cl(["--prompts", secret, "--same-system", "--out", os.path.join(T, "x")])
check("report: --same-system flag no longer exists", r.returncode != 0)
j6 = json.load(open(os.path.join(o6, "report.json")))
check("report JSON carries input_provenance NOT VERIFIED", "NOT VERIFIED" in j6["input_provenance"])
# F2: pair files without .jsonl suffix -> INCONCLUSIVE rc 2, no traceback (both tools)
nojl = os.path.join(T, "pair_nosuffix"); open(nojl, "w").write("{}\n")
r = cl(["--pair-before", nojl, "--pair-after", nojl])
check("cachelens non-.jsonl pair -> INCONCLUSIVE rc=2 no traceback", r.returncode == 2 and "Traceback" not in r.stderr)
r = subprocess.run([sys.executable, "analyze.py", nojl, nojl], capture_output=True, text=True)
check("analyze.py non-.jsonl pair -> INCONCLUSIVE rc=2 no traceback", r.returncode == 2 and "Traceback" not in r.stderr)
# corrupt metrics JSON next to a valid .jsonl pair
dbad = os.path.join(T, "badmetrics"); os.makedirs(dbad)
for v in ("ts_before", "ts_after"):
    with open(os.path.join(dbad, v + ".jsonl"), "w") as fh:
        for rr in fake_rows(v, 2, 10): fh.write(json.dumps(rr) + "\n")
    open(os.path.join(dbad, v + "_metrics.json"), "w").write("{not json")
r = cl(["--pair-before", os.path.join(dbad, "ts_before.jsonl"), "--pair-after", os.path.join(dbad, "ts_after.jsonl")])
check("cachelens corrupt pair metrics JSON -> INCONCLUSIVE rc=2 no traceback", r.returncode == 2 and "Traceback" not in r.stderr)
r = subprocess.run([sys.executable, "analyze.py", os.path.join(dbad, "ts_before.jsonl"), os.path.join(dbad, "ts_after.jsonl")], capture_output=True, text=True)
check("analyze.py corrupt metrics JSON -> INCONCLUSIVE rc=2 no traceback", r.returncode == 2 and "Traceback" not in r.stderr)
# responsive + legible chart markup present
check("report HTML has <600px stacked-card CSS and a scrollable fixed-min-width chart",
      "@media(max-width:600px)" in html3 and "data-l=" in html3 and 'class="chart"' in html3 and "min-width:640px" in html3)
# F4/F5/F6 wording
check("report: pairwise numbers and first-8-of-N shown; stable-text wording precise",
      "Across prompt PAIRS" in html3 and "first 8 of 18 prompts" in html3 and "appears in every prompt but sits after the first varying line" in html3)
res = env_report.report(open("evidence/vllm.log", errors="replace").read(), None, 5796, False)
check("env_report capacity ratio says the mean prompt tokens were user-supplied",
      any("YOU supplied" in o["evidence"] for o in res["observations"]))

print()
# 13. nvidia-smi collector (fixed command; parsed from file or mocked subprocess; no GPU needed)
import unittest.mock as _mock
_lg = open("evidence/vllm.log", errors="replace").read()
_r = env_report.report(_lg, None, None, False, smi_text=open("examples/env/smi_h100.csv").read())
check("smi file: GPU name/memory/driver parsed, source says unverified host",
      "H100" in _r["facts"]["gpu_hardware"][0] and "550.54.15" in _r["facts"]["gpu_hardware"][0]
      and "cannot verify" in _r["sources"]["gpu_hardware"])
check("smi file consistent with log device memory (79.27 vs 79.65 GiB) says consistent, not proof",
      any("consistent with the log" in o["observation"] and "not proof" in o["note"] for o in _r["observations"])
      and not any("WARNING: nvidia-smi" in o["observation"] for o in _r["observations"]))
check("smi supplied removes GPU-name unknown",
      not any("GPU model name" in u for u in _r["unknowns"]))
_r = env_report.report(_lg, None, None, False, smi_text=open("examples/env/smi_a10.csv").read())
check("smi memory mismatching the log warns different machines",
      any("WARNING: nvidia-smi" in o["observation"] for o in _r["observations"]))
for _bad in ("", "garbage", "0, name, 80000", "x, n, 80000, 1, 5", "0, n, -5, 1, 5", "0, n, 100, 500, 5"):
    try:
        env_report.report(_lg, None, None, False, smi_text=_bad)
        check("smi bad input fails closed: %r" % _bad, False)
    except env_report.BadInput:
        check("smi bad input fails closed: %r" % _bad, True)
_ok = _mock.Mock(returncode=0, stdout=open("examples/env/smi_h100.csv").read())
with _mock.patch("env_report.subprocess.run", return_value=_ok) as _m:
    _r = env_report.report(None, open("examples/env/metrics_snap1.txt").read(), None, True)
    check("local collector runs exactly the fixed read-only argv, no shell",
          _m.call_args[0][0] == env_report.SMI_CMD and not _m.call_args[1].get("shell")
          and "THIS report" in _r["sources"]["gpu_hardware"])
with _mock.patch("env_report.subprocess.run", side_effect=FileNotFoundError):
    _r = env_report.report(None, open("examples/env/metrics_snap1.txt").read(), None, True)
    check("local collector missing nvidia-smi -> unknown, not invented",
          _r["facts"]["gpu_hardware"] == env_report.NA and any("GPU model name" in u for u in _r["unknowns"]))
_rc = subprocess.run([sys.executable, "cachelens.py", "--log", "evidence/vllm.log", "--smi-file", "examples/env/smi_h100.csv", "--out", "/tmp/r_smi"], capture_output=True, text=True)
check("cachelens report with --smi-file shows GPU row", _rc.returncode == 0 and "H100" in open("/tmp/r_smi/report.html").read())
_rc = subprocess.run([sys.executable, "cachelens.py", "--log", "evidence/vllm.log", "--smi-file", "/nonexistent", "--out", "/tmp/r_smi2"], capture_output=True, text=True)
check("cachelens missing smi file fails rc!=0 w/o traceback", _rc.returncode != 0 and "Traceback" not in _rc.stderr)

print(f"{total} checks; " + ("FAILED: " + ", ".join(fails) if fails else "ALL TESTS PASSED"))
sys.exit(1 if fails else 0)
