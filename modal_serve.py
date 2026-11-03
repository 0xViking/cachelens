"""Modal app: serve Muse Glimmer 30B (official vLLM build) and run CacheLens
probes INSIDE the same container against localhost - no public endpoint.

COST WARNING: this runs a paid A100-80GB GPU (~$2.50/hr at list prices) and
writes ~60GB of model weights to a Modal volume ($0.09/GiB/mo while kept).
Check your workspace usage/spend limits first, and delete the
cachelens-weights volume when done.

AUTH: requires a Modal token in ~/.modal.toml (`modal token new`).

RUN:      modal run --detach modal_serve.py
EVIDENCE: written to the cachelens-evidence volume under <run-id>/;
          download with `modal volume get cachelens-evidence <run-id>/<file> <local>`.

Verified against https://dev.meta.ai/docs/muse-glimmer/deploy (Nov 3, 2026):
  image vllm/vllm-openai:muse-glimmer; model meta-models/Muse-Glimmer-30B
  (dense 30B, bf16 ~60GB weights, ~72GB VRAM -> A100-80GB); launch flags below.
PROVENANCE NOTE: the image tag and the HF `main` revision are mutable, so an
exact bit-for-bit rerun is not guaranteed. The Nov 3, 2026 evidence run used
vLLM 0.26.1rc1.dev608+g99a10304d (see evidence vllm.log).
"""
import modal

MODEL = "meta-models/Muse-Glimmer-30B"

image = (
    modal.Image.from_registry("vllm/vllm-openai:muse-glimmer", add_python="3.12")
    .entrypoint([])  # the vllm image's ENTRYPOINT would eat Modal's bootstrap args
    .add_local_file("workload.py", "/app/workload.py")
    .add_local_file("probe.py", "/app/probe.py")
    .add_local_file("analyze.py", "/app/analyze.py")
)

weights = modal.Volume.from_name("cachelens-weights", create_if_missing=True)
evidence = modal.Volume.from_name("cachelens-evidence", create_if_missing=True)

app = modal.App("cachelens-glimmer", image=image)

@app.function(gpu="A100-80GB",
              volumes={"/root/.cache/huggingface": weights,
                       "/evidence": evidence},
              timeout=5400)
def run_probes(run_id: str = ""):
    import subprocess, sys, time, urllib.request, os
    os.chdir("/app")
    if not run_id:
        from datetime import datetime, timezone
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = f"/evidence/{run_id}"
    os.makedirs(outdir)  # fail loudly if run-id dir already exists
    srv = subprocess.Popen([
        "vllm", "serve", MODEL,
        "--served-model-name", "muse-glimmer",
        "--tensor-parallel-size", "1",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "muse_glimmer",
        "--reasoning-parser", "muse_glimmer",
        "--generation-config", "auto",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--max-model-len", "32768",
        "--enable-prefix-caching",
        "--gpu-memory-utilization", "0.92",
    ], stdout=open(f"{outdir}/vllm.log", "w"), stderr=subprocess.STDOUT)
    failed = None
    try:
        deadline = time.time() + 3000
        while time.time() < deadline:
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
                break
            except Exception:
                if srv.poll() is not None:
                    failed = f"vLLM exited early, see {outdir}/vllm.log"
                    break
                time.sleep(10)
        else:
            failed = f"vLLM health timeout, see {outdir}/vllm.log"

        if not failed:
            for variant in ("ts_before", "ts_after"):
                out = f"{outdir}/{variant}.jsonl"
                r = subprocess.run([sys.executable, "probe.py",
                                    "--variant", variant, "--run-id", run_id,
                                    "--out", out],
                                   capture_output=True, text=True)
                open(f"{outdir}/{variant}.probe.log", "w").write(r.stdout + r.stderr)
                if r.returncode != 0:
                    failed = f"probe {variant} failed, see {outdir}/{variant}.probe.log"
                    break
        if not failed:
            r = subprocess.run([sys.executable, "analyze.py",
                                f"{outdir}/ts_before.jsonl", f"{outdir}/ts_after.jsonl"],
                               capture_output=True, text=True)
            open(f"{outdir}/analyze.log", "w").write(r.stdout + r.stderr)
            if r.returncode not in (0, 1):  # 1 = fix not confirmed, still a valid run
                failed = f"analyzer error rc={r.returncode}, see {outdir}/analyze.log"
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except Exception:
            srv.kill()
    if failed:
        raise RuntimeError(failed)
    print(f"DONE - evidence on cachelens-evidence volume under {run_id}/")
