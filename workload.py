"""CacheLens workload generator (hackathon build, Nov 3-4 2026).

Generates a SYNTHETIC chat workload (single "user" role per request) that
resembles an agent deployment. It is NOT a captured real trace; all content
is generated here. Use synthetic prompts for public demos; keep any real
prompts private and redact before sharing.
many requests sharing one long system prompt, with realistic cache-busting
mistakes introduced on purpose so CacheLens can find them.

Variants:
  clean      - identical shared prefix every request (prefix cache should hit)
  ts_before  - per-request timestamp placed BEFORE the shared prefix (classic
               cache-buster: everything after the first differing token misses)
  ts_after   - timestamp moved AFTER the shared prefix (the fix CacheLens
               recommends; prefix cache should hit again)
"""

SHARED_PREFIX = """You are FieldOps, an operations copilot for a cold-chain logistics fleet.
You answer dispatcher questions using the runbook below.

RUNBOOK (excerpt, repeat to make the prefix long):
""" + "\n".join(
    f"{i:03d}. When trailer {i%17} reports a door-open event outside a geofence, "
    f"verify the assigned route, ping the driver, then escalate to the regional "
    f"on-duty manager if unanswered within 10 minutes. Document every step."
    for i in range(120)
) + """

Always cite the runbook line numbers you used. Never invent policy.
"""

def build_requests(n: int = 18):
    reqs = []
    for i in range(n):
        q = f"Question {i+1}: what do I do when trailer {i%17} reports door-open outside the geofence at mile {40+i}?"
        ts = f"2026-11-04T09:{15+i:02d}:00+05:30"
        reqs.append({
            "id": f"req-{i+1:02d}",
            "question": q,
            "timestamp": ts,
            "variants": {
                "clean":     f"{SHARED_PREFIX}\n\nUser: {q}",
                "ts_before": f"[request_time={ts}]\n{SHARED_PREFIX}\n\nUser: {q}",
                "ts_after":  f"{SHARED_PREFIX}\n[request_time={ts}]\n\nUser: {q}",
            },
        })
    return reqs

if __name__ == "__main__":
    rs = build_requests()
    print(f"{len(rs)} requests x {len(rs[0]['variants'])} variants")
    print(f"shared prefix chars: {len(SHARED_PREFIX)}")
