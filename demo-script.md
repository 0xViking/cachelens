Hi, I'm Praneeth. I'm a backend engineer, and I'm learning inference engineering by building with it. This project is called CacheLens.

Serving engines like vLLM can reuse work for the start of a prompt. That's prefix caching. Whether it actually helps depends on three things: the exact text of your prompts, the server settings, and the engine's metrics. They live in different places, so when the hit rate is low, it's easy to end up guessing. CacheLens puts them side by side in one visual report, and labels what was measured and what was only read from a file.

Here's an example. If a value that changes on every request, like a timestamp, sits at the start of the prompt, the cache can miss. On a real Muse Glimmer thirty-billion engine, with the timestamp first, the engine's own counters showed about one percent of prompt tokens reused.

You can give CacheLens your own prompt file. This example is made up, and labeled that way. It marks where the first difference shows up across all the prompts. Some pairs may share more, so treat it as a lead to check.

For the real run, it reads the server log: the model, the vLLM version, and that prefix caching was on. The log doesn't name the GPU, and we didn't capture nvidia-smi, so it says unknown instead of guessing.

After we moved the timestamp behind the shared text, the hit rate went from about one percent to about ninety-four percent, from engine counters. That's one serial run of our synthetic workload. Both runs went back to back on one engine start, but the analyzer can't verify that from the files alone. We make no claim about speed, cost, or answer quality.

For hardware, you run one fixed read-only nvidia-smi command on the serving host and pass the file in. It's tested on sample files, not yet on a real GPU.

The idea and a small separate prototype came before the building period and aren't in this repo. The code and the Glimmer runs are new here. We found no substantive shared code in the old source we could recover, though we can't rule out missing old edits.

CacheLens. Evidence instead of guessing.
