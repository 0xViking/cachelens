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
- `cachelens.py` - one command: combined visual HTML + JSON report
- `env_report.py` - serving-environment report from local log/metrics files
- `examples/` - labeled synthetic inputs for the prompt diagnostic and env report
- `modal_serve.py` - one-command Muse Glimmer 30B serving on Modal (ephemeral)
- `evidence/` - captured probe output (this run's raw data)

## Use it on your own prompts (no GPU, no server)
`python3 analyze.py --prompts your-prompts.jsonl [--no-examples]` runs a text-level
diagnostic on prompts one app already sent to one model. Input is JSONL, one row per
request: `{"prompt": "..."}` or `{"messages": [{"role": "...", "content": ...}]}`.

It reports the shared leading text (global and across prompt pairs), the first line where
prompts differ with how many distinct values that line takes, how much identical text
sits after that point, and whether prompts are the same lines in a different order.

**Limits - read before trusting it**
- Text only: characters, not tokens, and not an engine measurement. No hit-rate prediction.
- Messages are joined as `role: content`, which is NOT the model's real chat template;
  the template can shift where the true break lies.
- `content` may be a string or a list of text parts. Image/tool parts, null content and
  non-UTF-8 files are rejected with `INCONCLUSIVE` (exit 2), not guessed.
- "Early variation" uses a fixed rule of thumb (shared text ends before 50% of the
  shortest prompt). A later break can still cost reuse.
- `looks_like` labels (timestamp/uuid/hex/number) are simple patterns on one line shape
  (e.g. `YYYY-MM-DD HH:MM`); epoch seconds or other date styles fall through to
  "contains-number" or "free-text". Treat it as a hint, never a culprit.
- Prompt text alone cannot show server flags, cache size/eviction, block granularity or routing.
- **Privacy:** the report echoes up to 90 characters of real prompt text per example.
  Use `--no-examples` before sharing output, and never paste the default output publicly.
  Secrets inside prompts can appear in it.

Two labeled synthetic examples ship in `examples/`:
```
python3 analyze.py --prompts examples/user_id_midprompt.jsonl
python3 analyze.py --prompts examples/reordered_tool_lines.jsonl
```

## Report on the serving environment (local files only, no probing)
`env_report.py` reads local files the app owner provides. It opens no network connection and
never SSHes anywhere. Anything it cannot find is "not available" and listed under unknowns.

```
python3 env_report.py --log vllm.log --metrics snap1.txt --metrics-later snap2.txt --mean-prompt-tokens 5796
python3 env_report.py --log vllm.log --with-nvidia-smi   # on the serving host only
```
- `--log`: vLLM startup log. Parsed: version, model, served name, `enable_prefix_caching`,
  dtype, max_seq_len, tensor parallel size, KV cache tokens/GiB, GPU memory, utilization setting.
  If the log has several engine starts, the LAST one is used and a warning is shown.
- `--metrics` / `--metrics-later`: Prometheus text (or a JSON dict). Every label set (engine,
  model) is kept separate, never merged. A hit-rate window needs two snapshots and is
  INCONCLUSIVE on counter reset, missing series, hits > queries or different label sets.
  A single snapshot or lifetime counters carry no hit-rate or eviction conclusion.
- Supported build: parsing was written against vLLM 0.26.1rc1 log lines and metric names
  (also reads `vllm:kv_cache_usage_perc` and `vllm:gpu_prefix_cache_*` aliases). Other builds
  may differ; unrecognized files fail closed with `INCONCLUSIVE`, they are not guessed.
- `--with-nvidia-smi` names the GPUs of the machine running the report. That is the model
  server only if you run it there; a warning is printed when combined with `--log`.
- Metadata (settings, memory sizes, GPU names) is context, not a diagnosis of a hardware bottleneck.
- Privacy: the output includes the model path/name from the log. Labeled synthetic fixtures
  are in `examples/env/`.

## One command, one visual report
```
python3 cachelens.py --prompts prompts.jsonl --log vllm.log --metrics snap1.txt --metrics-later snap2.txt \
    --pair-before evidence/ts_before.jsonl --pair-after evidence/ts_after.jsonl --out my-report
```
Writes `my-report/report.html` (self-contained: inline CSS/SVG, no scripts, no external assets)
and `my-report/report.json`. Every input is optional, but at least one is required; a missing
section is shown as "unknown", not skipped. The page has a prompt-layout diagram, a server/cache
table, the measured before/after table and a findings table. Each finding carries its evidence
class (TEXT-LEVEL, PARSED SETTINGS, MEASURED) and an unknown / next-check column. Classes are
never merged into one verdict; a "combined check" row appears only when a text-level break and a
valid counter window both exist, and says "consistent with", never "caused by".
- **Prompt text is redacted by default** in both HTML and JSON. `--show-examples` adds fragments
  (HTML-escaped) and a warning banner; do not share that output publicly.
- Any unusable input (bad file, bad JSON, non-UTF-8, an inconclusive measured pair) exits 2 with
  `INCONCLUSIVE`. If every requested cache window is unusable the report still renders, with a
  visible INCONCLUSIVE banner and no hit rate.
- Provenance: CacheLens cannot know that your prompts file, log, metrics and measured pair describe
  the same server or traffic. The report says so, makes no cross-section cause claim, and
  `report.json` carries `input_provenance: NOT VERIFIED`. If the log says prefix caching is off
  but the counters show hits, it flags INPUTS DISAGREE instead.
- Inputs under an `examples/` folder, or `--synthetic`, get a "SYNTHETIC EXAMPLE DATA" banner.
- The measured section is the preset validated pair (the shipped timestamp-move experiment), not
  a measurement of arbitrary prompts. Same server instance / no restart between the two runs is
  NOT verified by the tool; matching metric labels can repeat across restarts.
- On narrow screens (<600px) tables turn into labeled cards and the chart scrolls sideways.
- `analyze.py --prompts ... --no-examples` accepts the flag in any position. The plain CLI prints
  prompt fragments by default (see the privacy note above); `cachelens.py` redacts by default.
- `--with-nvidia-smi` (env_report.py) is a fixed command with a timeout and no shell; it has NOT
  been validated against a real GPU host in this repo's tests.

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

## GPU hardware (nvidia-smi)

Run this fixed read-only command on the model-serving host and save the output:

    nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader,nounits > smi.csv
    python3 cachelens.py --log vllm.log --smi-file smi.csv

CacheLens parses name, total memory and driver. It cannot verify which host produced the file; it only
cross-checks total GPU memory against the log ("consistent with, not proof of" / "WARNING: probably
different machines"). `env_report.py --with-nvidia-smi` runs the same fixed command (no shell, 15 s timeout)
on the machine running the report. Tested with fixtures and a mocked subprocess; NOT validated on a real GPU host yet.
