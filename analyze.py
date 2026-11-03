"""CacheLens analyzer: read probe evidence and diagnose WHY the prefix cache
is or is not hitting, then verify the fix.

Usage: python3 analyze.py evidence/ts_before.jsonl evidence/ts_after.jsonl
"""
import json, sys

def load(p):
    try:
        return [json.loads(l) for l in open(p)]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"INCONCLUSIVE - cannot read evidence file {p}: {e}")
        sys.exit(2)

class InvalidEvidence(Exception):
    """Raised when metrics evidence fails validation - the analysis must
    fail closed rather than produce numbers from bad data."""

def _counter(m, when, name, key):
    try:
        v = float(m[when][name + key])
    except KeyError:
        raise InvalidEvidence(f"missing counter {name} in {when} snapshot")
    if v != v or v in (float("inf"), float("-inf")):
        raise InvalidEvidence(f"non-finite counter {name} in {when} snapshot")
    return v

def metrics_delta(path):
    """Prefix-cache counter deltas over one probe window, FAIL-CLOSED:
    rejects missing/non-finite counters, counter resets, negative deltas,
    hits > queries, model/engine label mismatches, and absent snapshots."""
    try:
        m = json.load(open(path))
    except FileNotFoundError:
        raise InvalidEvidence(f"metrics file not found: {path}")
    if "error" in m.get("before", {}) or "error" in m.get("after", {}):
        raise InvalidEvidence(f"metrics snapshot unavailable in {path}: "
                              f"{m['before'].get('error') or m['after'].get('error')}")
    keys = [k for k in m["after"] if k.startswith("vllm:prefix_cache_queries_total")]
    if len(keys) != 1:
        raise InvalidEvidence(f"expected exactly 1 engine/model series, found {len(keys)} in {path}")
    key = keys[0][len("vllm:prefix_cache_queries_total"):]
    if "vllm:prefix_cache_queries_total" + key not in m["before"]:
        raise InvalidEvidence("engine/model series differs between before/after snapshots")
    qb = _counter(m, "before", "vllm:prefix_cache_queries_total", key)
    qa = _counter(m, "after", "vllm:prefix_cache_queries_total", key)
    hb = _counter(m, "before", "vllm:prefix_cache_hits_total", key)
    ha = _counter(m, "after", "vllm:prefix_cache_hits_total", key)
    if qa < qb or ha < hb:
        raise InvalidEvidence("counter reset or restart inside the probe window")
    q, h = qa - qb, ha - hb
    if q <= 0:
        raise InvalidEvidence("no prompt tokens queried in window")
    if h < 0:
        raise InvalidEvidence("negative hit delta")
    if h > q:
        raise InvalidEvidence(f"hits ({h}) exceed queries ({q}) - counter mismatch")
    return {"prompt_tokens_queried": int(q), "cached_tokens_hit": int(h),
            "hit_rate_pct": round(100 * h / q, 1)}

def summarize(rows):
    total = sum(r["prompt_tokens"] or 0 for r in rows)
    out = {"requests": len(rows), "total_prompt_tokens": total}
    if all(r["cached_tokens"] is None for r in rows):
        # Do not report 0: the field was never populated, so per-request
        # cache accounting is UNKNOWN, not zero.
        out["per_request_cached_tokens"] = (
            "unavailable - usage.cached_tokens not populated by this "
            "serving build; engine /metrics deltas below are the "
            "measurement of record")
    else:
        cached = [r["cached_tokens"] or 0 for r in rows]
        out["requests_with_cache_hits"] = sum(1 for c in cached if c > 0)
        out["total_cached_tokens"] = sum(cached)
    return out

def diagnose(rows):
    """The CacheLens rule set (v1). Works from the ACTUAL prompts captured by
    the probe, not assumptions: find where any two requests' prompts stop
    being byte-identical, and name what sits at the divergence point."""
    findings = []
    prompts = [r["prompt"] for r in rows if r.get("prompt")]
    if len(prompts) < 2:
        return findings
    # Compare EVERY pair of captured prompts, not just the first two, and
    # take the earliest divergence across all pairs.
    div = None
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            a, b = prompts[i], prompts[j]
            d = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y),
                     min(len(a), len(b)))
            div = d if div is None else min(div, d)
    a = prompts[0]
    total = len(a)
    ctx = a[max(0, div-40):div+40].replace("\n", "\\n")
    early = div < total * 0.5
    findings.append({
        "rule": "first-divergence",
        "diverges_at_char": div,
        "divergence_position": "EARLY (before the shared prefix body)" if early else "late",
        "context": f"...{ctx}...",
        "verdict": ("All %d captured requests differ from character %d of %d "
                    "(character offset, not tokens). A per-request value sits "
                    "near the START of the prompt, so tokens after it cannot "
                    "match cached blocks from other requests - consistent "
                    "with the low measured hit rate." % (len(prompts), div, total)) if early else
                   ("All %d captured requests stay identical until character "
                    "%d of %d - the shared prefix is intact at the prompt "
                    "level, so this rule finds no prompt-layout cause; any "
                    "misses need a different explanation." % (len(prompts), div, total)),
        "fix": "Move per-request fields after the shared prefix, then re-probe "
               "and confirm cached_tokens rises." if early else
               "Check server flags (prefix caching enabled) and engine metrics.",
    })
    return findings

def main():
    before, after = load(sys.argv[1]), load(sys.argv[2])
    # Fail closed on structural problems before printing any numbers.
    if not before or not after:
        print("INCONCLUSIVE - empty evidence file")
        return 2
    vb = {r.get("variant") for r in before}
    va = {r.get("variant") for r in after}
    if vb != {"ts_before"} or va != {"ts_after"}:
        print(f"INCONCLUSIVE - variant labels {vb}/{va} do not match "
              "expected ts_before/ts_after pair")
        return 2
    if len(before) != len(after):
        print(f"INCONCLUSIVE - request count differs ({len(before)} vs {len(after)})")
        return 2
    if any(r.get("error") for r in before + after):
        print("INCONCLUSIVE - one or more probed requests errored")
        return 2
    print("== ts_before (broken) ==", json.dumps(summarize(before), indent=2))
    print("== ts_after (fixed)  ==", json.dumps(summarize(after), indent=2))
    try:
        mb = metrics_delta(sys.argv[1].replace(".jsonl", "_metrics.json"))
        ma = metrics_delta(sys.argv[2].replace(".jsonl", "_metrics.json"))
    except InvalidEvidence as e:
        print(f"\nINCONCLUSIVE - evidence failed validation: {e}")
        return 2
    print("\n== engine /metrics delta: ts_before ==", json.dumps(mb, indent=2))
    print("== engine /metrics delta: ts_after  ==", json.dumps(ma, indent=2))
    # Verification criteria, stated explicitly: both windows valid AND
    # identical queried volume AND the fixed variant's hit rate is high in
    # absolute terms (>= 75%) AND at least 50 points above the broken run.
    if mb["prompt_tokens_queried"] != ma["prompt_tokens_queried"]:
        print(f"\nINCONCLUSIVE: prompt volume differs between variants "
              f"({mb['prompt_tokens_queried']} vs {ma['prompt_tokens_queried']}).")
        return 2
    ok = (ma["hit_rate_pct"] >= 75.0
          and ma["hit_rate_pct"] - mb["hit_rate_pct"] >= 50.0)
    print(f"\n{'FIX VERIFIED' if ok else 'FIX NOT CONFIRMED'}: prefix-cache hit "
          f"rate {mb['hit_rate_pct']}% -> {ma['hit_rate_pct']}% on identical "
          f"prompt volume ({mb['prompt_tokens_queried']} tokens). "
          f"Scope: this single synthetic serial run only.")
    for f in diagnose(before):
        print("\nDIAGNOSIS:", json.dumps(f, indent=2))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
