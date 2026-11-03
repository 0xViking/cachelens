"""CacheLens probe: fire a workload at an OpenAI-compatible vLLM server and
capture per-request prefix-cache evidence.

Usage: python3 probe.py --variant ts_before --out evidence/<run-id>/ts_before.jsonl
       [--server http://127.0.0.1:8000] [--model muse-glimmer] [--run-id RUN]

Captures per request: prompt tokens, cached tokens
(usage.prompt_tokens_details.cached_tokens when the serving build populates
it - the official muse-glimmer build does NOT), wall latency (perf_counter),
finish_reason, completion text, completion tokens, plus /metrics prefix-cache
counter snapshots before and after the run.

Rows are appended incrementally so a crash still leaves partial evidence.
Prompts are synthetic (see workload.py) - do not probe with real user data
without redaction; evidence JSONL contains FULL prompts.
"""
import argparse, json, time, urllib.request
from datetime import datetime, timezone
from workload import build_requests

def chat(server, model, prompt, max_tokens=64):
    body = json.dumps({
        "model": model, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    r = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions",
                               data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.loads(resp.read())

def metrics_snapshot(server):
    """Return prefix-cache counters, or an explicit error object when the
    server does not expose them."""
    try:
        with urllib.request.urlopen(server.rstrip("/") + "/metrics", timeout=30) as resp:
            txt = resp.read().decode()
        rows = {l.split()[0]: l.split()[1] for l in txt.splitlines()
                if l and not l.startswith("#") and "prefix_cache" in l}
        return rows if rows else {"error": "no prefix_cache counters in /metrics"}
    except Exception as e:
        return {"error": f"metrics unavailable: {e}"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--variant", required=True, choices=["clean", "ts_before", "ts_after"])
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--run-id", default=None,
                    help="stamped into every row; use a fresh id per run so "
                         "evidence files are never overwritten")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_id = a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    before = metrics_snapshot(a.server)
    n_rows = 0
    with open(a.out, "w") as f:
        for r in build_requests(a.n):
            prompt = r["variants"][a.variant]
            t0 = time.perf_counter()
            try:
                resp = chat(a.server, a.model, prompt)
                err = None
            except Exception as e:
                resp, err = {}, f"{type(e).__name__}: {e}"
            dt = time.perf_counter() - t0
            u = resp.get("usage") or {}
            det = u.get("prompt_tokens_details") or {}
            choices = resp.get("choices") or [{}]
            row = {
                "id": r["id"], "variant": a.variant, "run_id": run_id,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "latency_s": round(dt, 3),
                "prompt": prompt,
                "prompt_tokens": u.get("prompt_tokens"),
                "cached_tokens": det.get("cached_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "finish_reason": choices[0].get("finish_reason"),
                "completion_text": (choices[0].get("message") or {}).get("content"),
                "error": err,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_rows += 1
            print(f"{r['id']}: prompt={u.get('prompt_tokens')} cached={det.get('cached_tokens')} {dt:.2f}s{(' ERROR ' + err) if err else ''}")
    after = metrics_snapshot(a.server)
    with open(a.out.replace(".jsonl", "_metrics.json"), "w") as f:
        json.dump({"run_id": run_id, "variant": a.variant,
                   "before": before, "after": after}, f, indent=2)
    print(f"wrote {a.out} (+ _metrics.json), {n_rows} rows, run_id={run_id}")

if __name__ == "__main__":
    main()
