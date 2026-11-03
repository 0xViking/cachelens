# CacheLens - voiceover script (silent video v2, 2:20)

Read at a relaxed pace (~130 words/min). Timings are scene starts in the video; each block fits its scene with a little room.

**1. 0:00-0:10 - problem**
"Agents resend the same long prompt every request. If one per-request value, like a timestamp, sits near the start, the prefix cache can't reuse it. CacheLens finds that."

**2. 0:10-0:22 - broken run**
"On a real Muse Glimmer 30B engine, with a timestamp first, the engine's own counters show a 1.0% prefix-cache hit rate. 1,088 of 104,328 tokens reused."

**3. 0:22-0:42 - your own prompt file (example data)**
"You can give it your own prompt file. This example is made up, and it's labeled that way. The report draws every prompt and marks where the shared prefix ends. Here it ends at character 118, so most of each prompt can't be shared as a leading prefix."

**4. 0:42-0:54 - real log, settings**
"From the real server log it reads the settings: vLLM version, model, prefix caching enabled."

**5. 0:54-1:02 - real log, unknowns**
"GPU KV cache is 776,150 tokens. But the log doesn't say which GPU. We didn't capture nvidia-smi, so the report says unknown instead of guessing."

**6. 1:02-1:22 - measured result**
"After moving the timestamp behind the shared prefix, the same workload goes from 1.0% to 93.6% hit rate, from engine counters. This is one serial run of our synthetic workload, 18 requests per variant. We don't claim the same server instance was verified, and no latency, cost or answer-quality claim."

**7. 1:22-1:40 - findings table**
"Each finding says what kind of evidence it is, and what's still unknown or needs checking."

**8. 1:40-1:54 - nvidia-smi**
"For hardware, run one fixed read-only nvidia-smi command on the serving host and pass the file in. CacheLens can't verify which host it came from. It's tested on fixtures, not yet on a real GPU."

**9. 1:54-2:12 - provenance**
"On provenance: the idea and a separate small prototype came before the building period, and aren't in this repo. The code and the Glimmer runs are new for this repo. A comparison of the recoverable old source found no substantive shared code, though missing old edits can't be excluded."

**10. 2:12-2:20 - close**
"CacheLens: find where your prompts break prefix caching."

Word count about 290 - if you run long, drop the last sentence of scene 3 or 9. Every number on screen is real except scene 3 (labeled SYNTHETIC EXAMPLE).
