# CacheLens - why isn't my prefix cache hitting?

CacheLens diagnoses prefix-cache misses in self-hosted LLM serving (vLLM).
It probes a synthetic agent-like workload, finds what breaks byte-identical
leading prefixes, and measures the effect of a candidate fix with a
controlled replay.

Built during the Meta Global AI Developer Hackathon (Oct 27 - Nov 4 window)
against **Muse Glimmer 30B** (Meta open weights, Apache 2.0) served with
vLLM on a single GPU.

## The demo story (verified on Muse Glimmer 30B, Nov 3 2026)
1. An agent workload (18 dispatcher requests sharing a 25.5KB runbook prefix)
   is probed against vLLM - `probe.py --variant ts_before`.
2. CacheLens diffs all captured prompts and pinpoints the cause: a per-request
   timestamp at char 28, BEFORE the shared prefix - `analyze.py`.
   Engine `/metrics`: 1.0% prefix-cache hit rate (1,088 / 104,328 tokens).
3. The fix (move per-request fields after the shared prefix) is replayed -
   `probe.py --variant ts_after`. Observed hit rate rises to 93.6%
   (97,696 / 104,328 tokens). Median per-request latency drops 34.3%
   (3.6855s -> 2.4200s) - single synthetic run, not a production benchmark.

Scope and honesty notes:
- One synthetic, serial run (18 requests per variant) on one A100-80GB,
  Nov 3, 2026. Not a production benchmark, no TTFT/cost-saving claim, no
  answer-equivalence check, no broad model/engine compatibility claim.
- The official `vllm/vllm-openai:muse-glimmer` build does not populate
  `usage.prompt_tokens_details.cached_tokens`, so the analyzer uses engine
  `/metrics` counter deltas as the measurement of record and says so.
- Provenance: image tag `vllm/vllm-openai:muse-glimmer` and the HF `main`
  revision are mutable; the recorded run used vLLM 0.26.1rc1.dev608+g99a10304d
  (see `evidence/*/vllm.log`). Exact bit-for-bit reproduction is not
  guaranteed by these tags.
- The analyzer fails closed (INCONCLUSIVE) on missing/invalid metrics,
  counter resets, hits>queries, mismatched variants/volumes, or errored
  requests. Local tests: `python3 test_cachelens.py`.

Cost, auth, data retention:
- `modal run --detach modal_serve.py` spends real money: an A100-80GB GPU
  (~$2.50/hr list) plus ~60GB of weights cached on a Modal volume
  ($0.09/GiB/mo). Delete the `cachelens-weights` volume when done
  (`modal volume delete cachelens-weights`); `cachelens-evidence` is small.
- Needs a Modal token in `~/.modal.toml` (`modal token new`).
- Evidence JSONL contains FULL prompts. The committed evidence is 100%
  synthetic (see `workload.py`). If you probe real traffic, keep that
  evidence private and redact before sharing.

## Contents
- `workload.py` - synthetic workload with clean / broken / fixed variants
- `probe.py` - OpenAI-compatible probing + vLLM /metrics capture
- `analyze.py` - measured before/after verification + `--prompts` prompt-file diagnostic
- `prompt_diag.py` - text-level prompt diagnostic used by `analyze.py --prompts`
- `env_report.py` - serving-environment report from local log/metrics files
- `examples/` - labeled synthetic inputs for the prompt diagnostic and env report
- `modal_serve.py` - one-command Muse Glimmer 30B serving on Modal (ephemeral)
- `evidence/` - captured probe output (this run's raw data)

## Use it on your own prompts (no GPU, no server)
`analyze.py --prompts your-prompts.jsonl` runs a text-level diagnostic on prompts one
app already sent to one model. It needs no GPU and no live server. Input is JSONL:
one `{"prompt": "..."}` or `{"messages": [{"role": "...", "content": "..."}]}` per line.

It reports:
- how many characters of leading text all prompts share, and the same statistic
  across prompt pairs (so one outlier cannot hide a systematic early break);
- the first line where prompts differ, with example values and how many distinct
  values that line takes (a per-request id/timestamp looks different from a small
  set like a user or tenant field);
- how much identical text sits AFTER that first difference, which a leading-prefix
  cache cannot reuse across these requests;
- whether the prompts are the same lines in a different order (e.g. reordered
  tool or document lists).

What it does NOT do: it does not measure any engine's cache, does not count tokens
(character offsets only), and does not predict a hit rate. Prompt text can only show
layout-level candidate causes. Server flags, cache size/eviction, block granularity
and routing are separate causes this file cannot see. Moving or reordering prompt
content can change model behavior - the app owner decides and re-checks outputs.

Two labeled examples ship in `examples/` (synthetic, text-level only):
```
python3 analyze.py --prompts examples/user_id_midprompt.jsonl    # mid-prompt account id
python3 analyze.py --prompts examples/reordered_tool_lines.jsonl # same tools, different order
```
Bad input fails closed with `INCONCLUSIVE` (exit code 2) instead of guessing.

## Report on the serving environment (no probing, local files only)
`env_report.py` collects what a serving setup already says about itself and keeps
facts, observations and unknowns separate. It reads only local files the app owner
gives it - it never opens a network connection, never SSHes anywhere, and reports
every field it cannot find as "not available" instead of guessing.

```
python3 env_report.py --log vllm.log --metrics snap1.txt --metrics-later snap2.txt \
    --mean-prompt-tokens 5796
# on the serving host only, add GPU names:
python3 env_report.py --log vllm.log --with-nvidia-smi
```

- `--log`: a vLLM startup/runtime log. Parsed facts include engine version, model,
  `enable_prefix_caching`, dtype, max_seq_len, GPU KV cache size in tokens and GiB,
  total GPU memory, and the gpu_memory_utilization setting - each with its source.
- `--metrics` / `--metrics-later`: two Prometheus `/metrics` snapshots. A hit-rate
  reading is made ONLY from the counter delta between two snapshots; lifetime
  counters and a single usage snapshot are reported as facts with no conclusion
  (one snapshot cannot separate eviction from a cold start or a burst).
- `--mean-prompt-tokens`: enables one arithmetic capacity ratio (KV cache tokens
  vs your mean prompt), labeled as observed/given numbers.
- `--with-nvidia-smi`: queries the machine running the report; only meaningful on
  the serving host, and labeled as such.

Observations are tied to observed values (example: `enable_prefix_caching=False`
means no prompt layout can hit a cache that is off). Static metadata is never used
to claim a hardware bottleneck. Labeled synthetic fixtures live in `examples/env/`.

## Reproduce
```
pip install -r requirements.txt   # needs a Modal token in ~/.modal.toml
modal run --detach modal_serve.py # serves + probes inside the GPU container;
                                  # note the run-id it prints at the end
modal volume get cachelens-evidence <run-id>/ts_before.jsonl evidence/ts_before.jsonl
modal volume get cachelens-evidence <run-id>/ts_after.jsonl  evidence/ts_after.jsonl
modal volume get cachelens-evidence <run-id>/ts_before_metrics.json evidence/ts_before_metrics.json
modal volume get cachelens-evidence <run-id>/ts_after_metrics.json  evidence/ts_after_metrics.json
python3 analyze.py evidence/ts_before.jsonl evidence/ts_after.jsonl
```

## Provenance and what is new

- Before the building period (Oct 26, 2026, kickoff day, ahead of Oct 27 - Nov 4): project idea selection and a separate private
  prototype tested on SmolLM2-135M (vLLM 0.11.0, Modal T4). That prototype is not part of this repository. It showed that
  per-request cached-token fields can be null while engine /metrics still count prefix-cache hits.
- This repository's current implementation was created on Nov 3, 2026: workload.py, probe.py, analyze.py, modal_serve.py,
  tests, and the evidence in evidence/, captured by serving Meta Muse Glimmer 30B (official muse-glimmer vLLM image) on a
  Modal A100-80GB. Every number in evidence/ comes from that Nov 3 run.
- Comparison of the recoverable Oct 26 prototype source against the current code found no substantive shared code. The
  comparison covers only the old files that could be recovered, so missing old edits cannot be excluded. The carried-over
  idea is to treat engine counters as the measurement of record.
- This is a disclosure, not a claim about how judges will treat eligibility.
