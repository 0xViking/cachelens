# CacheLens demo video script (captioned screen capture, target ~2:30)

All footage is a labeled replay of the captured Nov 3, 2026 run evidence
(caption: "captured run replay"). Numbers below are the verified results.

## 0:00-0:15 Problem
Caption card: "Repeated prompt processing can waste GPU time. When a shared
prefix stops caching, latency rises - and nothing tells you why."

## 0:15-0:45 The broken workload
Screen: terminal replay of evidence/ts_before.probe.log (18 requests,
prompt=5796 tokens each, ~3.7s per request) + metrics delta 1,088/104,328.
Caption: "18 agent requests share a 25.6KB runbook prefix. vLLM's prefix
cache should absorb it. Measured hit rate: 1.0% (1,088 of 104,328 tokens)."

## 0:45-1:30 Diagnosis
Screen: analyze.py output - diverges_at_char 28, the [request_time=...]
context snippet, verdict.
Caption: "CacheLens diffs the captured prompts and finds the first character
where requests stop matching (char 28): a timestamp placed BEFORE the shared
prefix. Tokens after it cannot match cached blocks from other requests."

## 1:30-2:10 Verified fix
Screen: ts_after metrics delta 97,696/104,328 (93.6%), probe log ~2.4s/req,
OBSERVED INCREASE line.
Caption: "Moving per-request fields after the shared prefix coincided with
the hit rate rising to 93.6% (engine metrics, identical 104,328-token
synthetic workload). Median latency: -34% (3.69s -> 2.42s). Single synthetic
serial run on one A100 - not a production benchmark."

## 2:10-2:30 Close
Card: "CacheLens - capture, diagnose, verify prefix caching.
Muse Glimmer 30B (Meta open weights) + vLLM on a single A100.
Built during the Meta Global AI Developer Hackathon."
