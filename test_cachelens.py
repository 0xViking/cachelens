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

print()
print(f"{total} checks; " + ("FAILED: " + ", ".join(fails) if fails else "ALL TESTS PASSED"))
sys.exit(1 if fails else 0)
