# CacheLens measurement notes (for independent review)

## What was measured
- Model: meta-models/Muse-Glimmer-30B (dense 30B, bf16), served by the official
  Meta vLLM build `vllm/vllm-openai:muse-glimmer` (vLLM 0.26.1rc1.dev608+g99a10304d),
  on one Modal A100-80GB. `enable_prefix_caching=True` confirmed in evidence/vllm.log
  ("non-default args"). Weights downloaded from HF in 342.6s (vllm.log).
- Workload: 18 synthetic dispatcher requests (workload.py), each a single chat
  completion, max_tokens=64, temperature default. Prompts are 25,640 chars /
  5,796 tokens each; 104,328 prompt tokens per variant run (18 x 5,796 exactly).
  Variants: ts_before = per-request [request_time=...] placed BEFORE the shared
  runbook prefix; ts_after = identical content with that field moved AFTER the
  prefix. Prompts in both variants are otherwise byte-identical across requests
  except req-id/timestamp content, and every full prompt is logged in the
  evidence JSONL for audit.

## Run order and cache state
- Single server session in one Modal container. ts_before probed FIRST on a
  fresh server (prefix_cache_hits_total = 0 at its "before" snapshot), then
  ts_after on the SAME server with NO cache reset. Counter deltas exclude
  earlier increments, not cache contents: ts_after could in principle reuse
  blocks cached during ts_before. Because the two variants' leading prompt
  content differs, cross-variant reuse is limited to any common rendered/
  token header; we did not measure its exact contribution.
- Metric windows: probe.py snapshots /metrics immediately BEFORE request 1 and
  immediately AFTER the last request of each variant. All reported numbers are
  per-window counter DELTAS (counter increments within each window), which
  remove counts from earlier windows; they do NOT isolate cache state, and no
  cache-state reset was performed between variants.
- Shared-substring note: the analysis only reasons about LEADING prefixes of
  the rendered prompt text. An arbitrary shared substring elsewhere in the
  prompt is NOT prefix-cache reuse.
- Provenance: the committed evidence was captured with the first-version
  probe (no per-row run_id, no completion text/finish_reason fields - those
  fields were added to the probe afterwards). The run identity is the
  cachelens-evidence volume contents of Nov 3, 2026 plus evidence/vllm.log;
  no synthetic run metadata was backfilled.

## Results (engine counters, the measurement of record)
- ts_before: prefix_cache_queries +104,328, hits +1,088 -> 1.0% hit rate.
  The 1,088 hits are a small residual (unexplained at token level; only
  tokens before the divergence point - char 28 - are candidates for
  cross-request reuse).
- ts_after: queries +104,328, hits +97,696 -> 93.6% hit rate. For reference,
  an idealized identical-prompt run in which request 1 populates the cache
  and requests 2-18 fully reuse it would hit 17/18 (~94.4%) of the prompt
  volume; that is an idealized reference point, not a verified bound, and
  the gap to 93.6% is not explained at token level.
- Latency (probe-measured wall time per completion, max_tokens=64 requested;
  completion texts and decoded token counts were NOT recorded, so no
  answer-equivalence or exact decode-length claim is made). Single run per
  variant (n=1, no variance estimate), synthetic workload - NOT a production
  benchmark. Median per-request latency: ts_before 3.6855s -> ts_after
  2.4200s (34.34% lower). Means, all 18 requests: 3.6861s -> 2.4995s
  (32.19%); means excluding each run's first request: 3.6816s -> 2.4202s.

## Instrument limitation
- This vLLM build does NOT populate usage.prompt_tokens_details.cached_tokens
  in chat-completion responses (probe recorded cached=None for all 36
  requests). Per-request cached-token accounting is therefore unavailable;
  engine /metrics counter deltas are used instead and analyze.py says so
  explicitly in its output.

## Diagnosis provenance
- analyze.py's divergence finding (char 28, [request_time=...] field) is
  computed from the ACTUAL prompts captured in evidence/ts_before.jsonl,
  not from workload.py assumptions.

## Caveats
- Synthetic workload, one GPU, one model, single run per variant.
- Hit-rate ceiling is workload-dependent; 93.6% reflects this 18-request
  workload, not a general guarantee.
