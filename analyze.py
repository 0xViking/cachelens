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
    hits > queries, model/engine label mismatches, and absent snapshots.
    Returns (delta dict, series key) so callers can compare windows."""
    try:
        m = json.load(open(path))
    except FileNotFoundError:
        raise InvalidEvidence(f"metrics file not found: {path}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise InvalidEvidence(f"metrics file is not valid JSON: {path}")
    if not isinstance(m, dict) or not isinstance(m.get("before"), dict) or not isinstance(m.get("after"), dict):
        raise InvalidEvidence(f"metrics file has no before/after snapshots: {path}")
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
    return ({"prompt_tokens_queried": int(q), "cached_tokens_hit": int(h),
             "hit_rate_pct": round(100 * h / q, 1)}, key)

def summarize(rows):
    for r in rows:
        if not isinstance(r.get("prompt_tokens"), (int, float)) or r["prompt_tokens"] <= 0:
            raise InvalidEvidence(
                f"row {r.get('id')} has missing/non-positive/non-numeric prompt_tokens")
    total = sum(r["prompt_tokens"] for r in rows)
    out = {"requests": len(rows), "total_prompt_tokens": total}
    have = [r["cached_tokens"] for r in rows if r["cached_tokens"] is not None]
    if not have:
        # Do not report 0: the field was never populated, so per-request
        # cache accounting is UNKNOWN, not zero.
        out["per_request_cached_tokens"] = (
            "unavailable - usage.cached_tokens not populated by this "
            "serving build; engine /metrics deltas below are the "
            "measurement of record")
    else:
        out["cached_token_coverage"] = f"{len(have)}/{len(rows)} requests"
        out["requests_with_cache_hits"] = sum(1 for c in have if c > 0)
        out["total_cached_tokens"] = sum(have)
    return out

def diagnose(rows):
    """The CacheLens rule set (v1). Works from the ACTUAL prompts captured by
    the probe, not assumptions: find where any two requests' prompts stop
    being byte-identical, and name what sits at the divergence point."""
    findings = []
    prompts = [r["prompt"] for r in rows if r.get("prompt")]
    if len(prompts) < 2:
        return findings
    # Global common-prefix length across ALL captured prompts (not just the
    # first pair). Character offset, not tokens; safe on prefix-of-another
    # inputs (os.path.commonprefix handles strict-prefix cases).
    import os.path
    div = len(os.path.commonprefix(prompts))
    a = prompts[0]
    total = len(a)
    ctx = a[max(0, div-40):div+40].replace("\n", "\\n")
    early = div < total * 0.5
    findings.append({
        "rule": "first-divergence",
        "diverges_at_char": div,
        "divergence_position": "EARLY (before the shared prefix body)" if early else "late",
        "context": f"...{ctx}...",
        "verdict": ("The common leading prefix shared by ALL %d captured "
                    "requests ends at character %d of %d (character offset, "
                    "not tokens). A per-request value sits near the START of "
                    "the prompt; prefix-cache reuse needs a byte-identical "
                    "LEADING prefix, so early per-request variation leaves "
                    "little shared leading content to reuse - consistent "
                    "with the low measured hit rate." % (len(prompts), div, total)) if early else
                   ("All %d captured requests share a leading prefix of %d "
                    "characters - the shared prefix is intact at the prompt "
                    "level, so this rule finds no prompt-layout cause; any "
                    "misses need a different explanation (variable suffixes, "
                    "block granularity, cold cache, or serving config)." % (len(prompts), div)),
        "fix": "Move per-request fields after the shared prefix, then re-probe "
               "and confirm cached_tokens rises." if early else
               "Check server flags (prefix caching enabled) and engine metrics.",
    })
    return findings

def main():
    if "--prompts" in sys.argv[1:]:
        import prompt_diag
        args = sys.argv[1:]
        redact = "--no-examples" in args
        rest = [x for x in args if x != "--no-examples"]
        if len(rest) != 2 or rest[0] != "--prompts":
            print("usage: analyze.py --prompts prompts.jsonl [--no-examples]")
            return 2
        return prompt_diag.main(rest[1], redact=redact)
    if len(sys.argv) != 3:
        print("usage: analyze.py before.jsonl after.jsonl   (measured Glimmer pair)\n"
              "       analyze.py --prompts prompts.jsonl [--no-examples]   (text-level prompt diagnostic)")
        return 2
    if not (sys.argv[1].endswith(".jsonl") and sys.argv[2].endswith(".jsonl")):
        print("INCONCLUSIVE - evidence files must be named *.jsonl "
              "(their metrics are read from the matching *_metrics.json)")
        return 2
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
        mb, kb = metrics_delta(sys.argv[1].replace(".jsonl", "_metrics.json"))
        ma, ka = metrics_delta(sys.argv[2].replace(".jsonl", "_metrics.json"))
    except InvalidEvidence as e:
        print(f"\nINCONCLUSIVE - evidence failed validation: {e}")
        return 2
    # Cross-window consistency: same engine/model series in both windows.
    if kb != ka:
        print(f"\nINCONCLUSIVE - metrics series differ between windows ({kb} vs {ka})")
        return 2
    print("\n== engine /metrics delta: ts_before ==", json.dumps(mb, indent=2))
    print("== engine /metrics delta: ts_after  ==", json.dumps(ma, indent=2))
    # The captured per-request usage must reconcile with the engine counters:
    # queried-token delta must equal the sum of captured prompt_tokens.
    for label, rows, md in (("ts_before", before, mb), ("ts_after", after, ma)):
        captured = sum(r["prompt_tokens"] for r in rows)
        if md["prompt_tokens_queried"] != captured:
            print(f"\nINCONCLUSIVE - {label}: engine queried "
                  f"{md['prompt_tokens_queried']} tokens but captured usage sums "
                  f"to {captured}")
            return 2
    # Request identities must pair across the two windows.
    if [r.get("id") for r in before] != [r.get("id") for r in after]:
        print("\nINCONCLUSIVE - request ids do not pair across variants")
        return 2
    # Expected variant relationship, non-vacuous: each pair must differ;
    # ts_before must START with exactly one [request_time=...] line; ts_after
    # must contain exactly one occurrence AFTER the common text; and removing
    # that one field must make the pair identical.
    import re
    field = re.compile(r"\[request_time=[^\]]+\]")
    for rb, ra in zip(before, after):
        pb, pa = rb["prompt"], ra["prompt"]
        if pb == pa:
            print(f"\nINCONCLUSIVE - {rb.get('id')}: prompts identical across "
                  "variants; no transformation to validate")
            return 2
        m0 = field.match(pb)  # anchored at position 0
        hits_a = field.findall(pa)
        if not m0 or len(field.findall(pb)) != 1:
            print(f"\nINCONCLUSIVE - {rb.get('id')}: ts_before does not start "
                  "with exactly one [request_time=...] field")
            return 2
        if len(hits_a) != 1 or pa.startswith(m0.group(0)):
            print(f"\nINCONCLUSIVE - {rb.get('id')}: ts_after does not hold the "
                  "timestamp field after the shared text")
            return 2
        # The actual transformation in this workload: ts_before carries
        # "[request_time=X]\n" at the very start; ts_after carries the SAME
        # field as "\n[request_time=X]" later in the prompt. Validate exactly
        # that, and that stripping it leaves identical prompts.
        tb = m0.group(0)
        ma2 = field.search(pa)
        if ma2.group(0) != tb:
            print(f"\nINCONCLUSIVE - {rb.get('id')}: timestamp value differs "
                  "between variants")
            return 2
        if ma2.start() == 0 or pa[ma2.start()-1] != "\n":
            print(f"\nINCONCLUSIVE - {rb.get('id')}: ts_after timestamp is not "
                  "placed after shared text")
            return 2
        stripped_b = pb[len(tb) + 1:]  # drop leading field + its newline
        stripped_a = pa.replace("\n" + tb, "", 1)
        if stripped_b != stripped_a:
            print(f"\nINCONCLUSIVE - {rb.get('id')}: prompts are not the "
                  "expected timestamp-move pair")
            return 2
    if mb["prompt_tokens_queried"] != ma["prompt_tokens_queried"]:
        print(f"\nINCONCLUSIVE: prompt volume differs between variants "
              f"({mb['prompt_tokens_queried']} vs {ma['prompt_tokens_queried']}).")
        return 2
    ok = (ma["hit_rate_pct"] >= 75.0
          and ma["hit_rate_pct"] - mb["hit_rate_pct"] >= 50.0)
    print(f"\n{'OBSERVED INCREASE' if ok else 'THRESHOLD NOT MET'}: "
          f"prefix-cache hit rate {mb['hit_rate_pct']}% -> {ma['hit_rate_pct']}% "
          f"on identical prompt volume ({mb['prompt_tokens_queried']} tokens), "
          f"with the validated timestamp-move transformation applied. "
          f"Scope: this single synthetic serial run only.")
    for f in diagnose(before):
        print("\nDIAGNOSIS:", json.dumps(f, indent=2))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
