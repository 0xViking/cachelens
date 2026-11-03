# CacheLens - why isn't my prefix cache hitting?

CacheLens diagnoses prefix-cache misses in self-hosted LLM serving (vLLM).
It probes a synthetic agent-like workload, finds what breaks byte-identical
leading prefixes, and measures the effect of a candidate fix with a
controlled replay.

Built during the Meta Global AI Developer Hackathon (Oct 27 - Nov 4 window)
against **Muse Glimmer 30B** (Meta open weights, Apache 2.0) served with
vLLM on a single GPU.

## The demo story (verified on Muse Glimmer 30B, Nov 3 2026)
1. An agent workload (18 dispatcher requests sharing a 25.6KB runbook prefix)
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
- `analyze.py` - diagnosis rules + before/after verification
- `modal_serve.py` - one-command Muse Glimmer 30B serving on Modal (ephemeral)
- `evidence/` - captured probe output (this run's raw data)

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
