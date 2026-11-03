"""CacheLens combined report: one command, one visual HTML page + machine-readable JSON.

Composes the three CacheLens pieces on YOUR files. Each section keeps its own
evidence class and nothing is merged into a single verdict:
  TEXT-LEVEL        prompt layout from your prompts file (prompt_diag.py)
  PARSED SETTINGS   serving settings / cache counters from files (env_report.py)
  MEASURED          the validated before/after engine-counter pair (analyze.py)

Self-contained output: inline CSS and SVG only, no external assets, no network.
Prompt text is NOT included unless --show-examples is given.

Usage:
  python3 cachelens.py --prompts p.jsonl [--log vllm.log] [--metrics s1.txt
      [--metrics-later s2.txt]] [--mean-prompt-tokens N]
      [--pair-before B.jsonl --pair-after A.jsonl] [--show-examples] [--out DIR]
"""
import argparse, html, json, os, subprocess, sys

import prompt_diag, env_report, analyze

E = html.escape
UNKNOWN = "unknown"

CLASS_STYLE = {
    "TEXT-LEVEL": "#2563eb", "PARSED SETTINGS": "#7c3aed",
    "MEASURED": "#059669", "INPUT CHECK": "#b45309",
}

def build(args):
    paths = [x for x in (args.prompts, args.log, args.metrics, args.metrics_later,
                         args.pair_before, args.pair_after, args.smi_file) if x]
    fix = [x for x in paths if "examples" in os.path.normpath(os.path.abspath(x)).split(os.sep)]
    synth = args.synthetic or bool(fix)
    mixed = bool(fix) and len(fix) < len(paths) and not args.synthetic
    out = {"redacted_examples": not args.show_examples, "sections": {},
           "synthetic": bool(synth) and not mixed, "mixed_inputs": mixed,
           "fixture_inputs": [os.path.basename(x) for x in fix],
           "input_provenance": ("NOT VERIFIED: CacheLens cannot tell whether the prompts file, log, "
                                "metrics and measured pair describe the same server or traffic")}
    prompts = None
    if args.prompts:
        prompts = prompt_diag.load(args.prompts)
        out["sections"]["prompts"] = prompt_diag.diagnose(prompts, redact=not args.show_examples)
        out["_lens"] = [len(p) for p in prompts[:8]]
    if args.log or args.metrics or args.metrics_later or args.smi_file:
        lt = open(args.log, encoding="utf-8").read() if args.log else None
        m1 = open(args.metrics, encoding="utf-8").read() if args.metrics else None
        m2 = open(args.metrics_later, encoding="utf-8").read() if args.metrics_later else None
        st = open(args.smi_file, encoding="utf-8").read() if args.smi_file else None
        out["sections"]["environment"] = env_report.report(
            lt, m1, args.mean_prompt_tokens, False, metrics_later_text=m2, smi_text=st,
            skip_smi_crosscheck=env_report.fixture_mix_reason(args.log, args.smi_file))
        if args.smi_file and os.path.basename(args.smi_file) in out["fixture_inputs"] and "gpu_hardware" in out["sections"]["environment"]["sources"]:
            out["sections"]["environment"]["sources"]["gpu_hardware"] = "FIXTURE (made up): " + out["sections"]["environment"]["sources"]["gpu_hardware"]
    if args.pair_before or args.pair_after:
        if not (args.pair_before and args.pair_after):
            raise prompt_diag.BadInput("--pair-before and --pair-after must be given together")
        if not (args.pair_before.endswith(".jsonl") and args.pair_after.endswith(".jsonl")):
            raise prompt_diag.BadInput("--pair-before/--pair-after must be *.jsonl files "
                                       "(metrics are read from the matching *_metrics.json)")
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py"),
                            args.pair_before, args.pair_after], capture_output=True, text=True)
        if r.returncode not in (0, 1):
            first = (r.stdout.strip().splitlines() or ["INCONCLUSIVE"])[-1] if r.stdout.strip() else "INCONCLUSIVE"
            inc = [l for l in r.stdout.splitlines() if l.startswith("INCONCLUSIVE") or "INCONCLUSIVE" in l]
            raise prompt_diag.BadInput("measured pair is INCONCLUSIVE: " + (inc[0] if inc else first))
        mb, _ = analyze.metrics_delta(args.pair_before.replace(".jsonl", "_metrics.json"))
        ma, _ = analyze.metrics_delta(args.pair_after.replace(".jsonl", "_metrics.json"))
        verdict = [l for l in r.stdout.splitlines() if l.startswith(("OBSERVED INCREASE", "THRESHOLD NOT MET"))]
        nreq = [sum(1 for ln in open(x, encoding="utf-8") if ln.strip()) for x in (args.pair_before, args.pair_after)]
        out["sections"]["measured"] = {"before": mb, "after": ma, "requests_per_variant": nreq,
                                       "verdict": verdict[0] if verdict else UNKNOWN,
                                       "analyzer_rc": r.returncode}
    if not out["sections"]:
        raise prompt_diag.BadInput("give at least one of --prompts, --log/--metrics, --pair-before/--pair-after")
    out["findings"] = findings(out["sections"])
    return out

def findings(sec):
    rows = []
    p, e, m = sec.get("prompts"), sec.get("environment"), sec.get("measured")
    if p:
        fd = p.get("finding")
        if fd == "identical":
            f, ev, nxt = ("Prompts are byte-identical: prompt text is not the cause of any prefix miss",
                          "all %d prompts identical" % p["prompts"],
                          "check server settings, cache size and routing")
        elif fd == "reordered-lines":
            f, ev, nxt = ("Same lines, different order: ordering ends the shared start",
                          "shared leading text %d chars (%s%% of mean prompt)" % (p["leading_shared_chars"], p["leading_shared_pct_of_mean_prompt"]),
                          "app owner decides whether a fixed order is safe; may change model behavior")
        elif fd == "early-variation":
            fdv = p["first_divergence"]
            f, ev, nxt = ("Shared leading text ends early (%s)" % fdv["looks_like"],
                          "break at char %d, line %d; %d distinct value(s) there; %s%% of the first prompt is text that appears in every prompt but sits after the first varying line"
                          % (p["leading_shared_chars"], fdv["line_index"], fdv["distinct_values_at_that_line"],
                             p["stable_text_after_first_divergence"]["pct_of_first_prompt"]),
                          "a candidate cause only; moving it may change model behavior; confirm with a measured window")
        else:
            f, ev, nxt = ("Shared start is long; layout is not an obvious cause",
                          "shared leading text %s%% of mean prompt" % p["leading_shared_pct_of_mean_prompt"],
                          "look at server settings and cache capacity")
        rows.append((f, "TEXT-LEVEL", ev, nxt))
    if e:
        for o in e["observations"]:
            cls = "PARSED SETTINGS"
            rows.append((o["observation"], cls, o["evidence"], o["note"]))
        simple = [w for w in e["unknowns"] if " " not in w]
        other = [w for w in e["unknowns"] if " " in w]
        if simple:
            rows.append(("UNKNOWN settings (%d): %s" % (len(simple), ", ".join(simple)), "PARSED SETTINGS",
                         "not present in the provided files", "provide the vLLM startup log or leave as unknown"))
        for w in other:
            rows.append(("UNKNOWN: " + w, "PARSED SETTINGS", "not present in the provided files",
                         "provide it or leave as unknown"))
    if m:
        rows.append((m["verdict"].split(":")[0] + ": hit rate %s%% -> %s%%" % (m["before"]["hit_rate_pct"], m["after"]["hit_rate_pct"]),
                     "MEASURED",
                     "engine counters: %d/%d tokens hit before, %d/%d after (same %d-token volume)"
                     % (m["before"]["cached_tokens_hit"], m["before"]["prompt_tokens_queried"],
                        m["after"]["cached_tokens_hit"], m["after"]["prompt_tokens_queried"],
                        m["before"]["prompt_tokens_queried"]),
                     "preset experiment only; same-instance / no-restart NOT verified; no TTFT, cost or answer-equivalence claim"))
    # no causal cross-class reading is ever made; only an input-disagreement flag
    if p and e:
        oks = [w for w in e.get("windows", []) if w["status"] == "ok"]
        if e["facts"].get("enable_prefix_caching") is False and any(w["hit_pct"] > 0 for w in oks):
            rows.append(("INPUTS DISAGREE: the log says prefix caching is disabled, but a window shows hits",
                         "INPUT CHECK",
                         "enable_prefix_caching=False (log) vs hit rate %.1f%% (counters)" % max(w["hit_pct"] for w in oks),
                         "the log and the counters may come from different servers or runs; do not rely on either until checked"))
    return rows

def svg_layout(p, lens):
    cp = p["leading_shared_chars"]
    n = len(lens)
    W, H, pad, bh, gap = 760, 40 + n * 28, 190, 18, 10
    mx = max(lens)
    sc = (W - pad - 20) / mx
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" role="img" '
             'aria-label="Prompt layout: shared leading text versus the rest">' % (W, H)]
    for i, L in enumerate(lens):
        y = 28 + i * (bh + gap)
        parts.append('<text x="8" y="%d" font-size="12" fill="#374151">prompt %d (%d chars)</text>' % (y + 13, i + 1, L))
        sw = max(2, cp * sc) if cp else 0
        if cp:
            parts.append('<rect x="%d" y="%d" width="%.1f" height="%d" fill="#10b981"/>' % (pad, y, min(sw, L * sc), bh))
        rest = (L - cp) * sc
        if rest > 0:
            parts.append('<rect x="%.1f" y="%d" width="%.1f" height="%d" fill="#f59e0b"/>' % (pad + cp * sc, y, rest, bh))
    bx = pad + cp * sc
    parts.append('<line x1="%.1f" y1="18" x2="%.1f" y2="%d" stroke="#dc2626" stroke-width="2" stroke-dasharray="4 3"/>' % (bx, bx, H - 6))
    parts.append('<text x="%d" y="12" font-size="12" fill="#dc2626">red line: first difference at char %d</text>' % (pad, cp))
    parts.append('</svg>')
    legend = ('<div class="legend"><span><i style="background:#10b981"></i>leading text shared by ALL these prompts (character-level; real reuse depends on token blocks and needs measurement)</span>'
              '<span><i style="background:#f59e0b"></i>text after the first difference: not shared by ALL prompts as a leading prefix. Some prompt pairs may share more (see pairwise numbers); real token/cache behavior needs measurement</span></div>')
    return "".join(parts) + legend

def badge(cls):
    return '<span class="badge" style="background:%s">%s</span>' % (CLASS_STYLE.get(cls, "#6b7280"), E(cls))

def render(res):
    sec = res["sections"]
    h = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CacheLens report</title>'
         '<meta name="viewport" content="width=device-width,initial-scale=1"><style>'
         'body{font:15px/1.45 system-ui,Segoe UI,Arial,sans-serif;color:#111827;margin:0;background:#f3f4f6}'
         'main{max-width:900px;margin:0 auto;padding:20px}h1{font-size:22px;margin:0 0 4px}'
         'h2{font-size:16px;margin:0 0 10px}.sub{color:#4b5563;margin-bottom:16px}'
         '.card{background:#fff;border-radius:10px;padding:16px;margin:14px 0;box-shadow:0 1px 2px rgba(0,0,0,.08)}'
         '.badge{color:#fff;font-size:11px;font-weight:700;padding:2px 8px;border-radius:99px;white-space:nowrap}'
         '*{box-sizing:border-box}table{border-collapse:collapse;width:100%;table-layout:auto}.swipe{display:none}.card{overflow-x:auto}th,td{overflow-wrap:anywhere;text-align:left;padding:7px 8px;border-bottom:1px solid #e5e7eb;vertical-align:top;font-size:14px}'
         'th{font-size:12px;color:#4b5563;text-transform:uppercase;letter-spacing:.04em}'
         '.unk{color:#b91c1c;font-weight:700}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:#374151;margin-top:8px}'
         '.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}'
         '.na{color:#6b7280;font-style:italic}.foot{font-size:12px;color:#4b5563}.chart{overflow-x:auto}.chart svg{min-width:640px}'
         '@media(max-width:600px){main{padding:10px}.swipe{display:block}table,thead,tbody,tr,th,td{display:block;width:100%}'
         'tr:first-child:has(th){display:none}tr{border:1px solid #e5e7eb;border-radius:8px;margin:8px 0;padding:4px 8px}'
         'td{border:none;padding:4px 0}td[data-l]:before{content:attr(data-l);display:block;font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.04em}}'
         '</style></head><body><main>',
         '<h1>CacheLens report' + (' - SYNTHETIC EXAMPLE' if res.get('synthetic') else (' - MIXED INPUTS' if res.get('mixed_inputs') else '')) + '</h1><div class="sub">Why is my prefix cache not hitting? Three evidence classes, kept separate: ',
         badge("TEXT-LEVEL"), " ", badge("PARSED SETTINGS"), " ", badge("MEASURED"), '</div>']
    # 1 prompt layout
    h.append('<div class="card"><h2>1. Prompt layout %s</h2>' % badge("TEXT-LEVEL"))
    if "prompts" in sec:
        p = sec["prompts"]
        if p.get("finding") == "identical":
            h.append('<p>All %d prompts are byte-identical. Nothing to diagram.</p>' % p["prompts"])
        else:
            h.append('<p class="foot swipe">Swipe the chart sideways to see all of it.</p><div class="chart">' + svg_layout(p, res["_lens"]) + '</div>')
            if not res["redacted_examples"] and p.get("first_divergence"):
                h.append('<table><tr><th>Values at the first differing line (prompt text)</th></tr>')
                for ex in p["first_divergence"]["examples"]:
                    h.append('<tr><td data-l="Value"><code>%s</code></td></tr>' % E(ex))
                h.append('</table>')
            pw = p.get("pairwise_common_prefix_chars", {})
            h.append('<p class="foot">Character offsets, not tokens; token-block alignment is not modeled. Shared by ALL %d prompts: %d chars (%s%% of the mean prompt). '
                     'Across prompt PAIRS (sampled): min %s, median %s, max %s chars. Diagram shows the first %d of %d prompts.</p>'
                     % (p["prompts"], p["leading_shared_chars"], p["leading_shared_pct_of_mean_prompt"],
                        pw.get("min_chars", "?"), pw.get("median_chars", "?"), pw.get("max_chars", "?"),
                        len(res["_lens"]), p["prompts"]))
    else:
        h.append('<p class="na">Not provided (no --prompts file): <span class="unk">unknown</span></p>')
    h.append('</div>')
    if res.get("mixed_inputs"):
        h.insert(8, '<div class="card" style="border-left:6px solid #7c3aed"><b>MIXED INPUTS.</b> These files are '
                    'MADE-UP FIXTURES from examples/: <b>%s</b>. Every other input is as you supplied it. '
                    'Fixture values do not describe the real system; do not read them together.</div>'
                    % E(", ".join(res["fixture_inputs"])))
    if res.get("synthetic"):
        h.insert(8, '<div class="card" style="border-left:6px solid #7c3aed"><b>SYNTHETIC EXAMPLE DATA.</b> '
                    'These numbers come from made-up demo inputs, not from a real system.</div>')
    # banner when every requested window was unusable
    if "environment" in sec and sec["environment"]["windows"] and \
            all(w["status"] == "inconclusive" for w in sec["environment"]["windows"]):
        h.insert(8, '<div class="card" style="border-left:6px solid #b91c1c"><b class="unk">INCONCLUSIVE:</b> '
                    'every requested cache window was unusable (see table 2). No hit-rate reading is made.</div>')
    # 2 server table
    h.append('<div class="card"><h2>2. Server and cache settings %s</h2>' % badge("PARSED SETTINGS"))
    if "environment" in sec:
        e = sec["environment"]
        setting_rows = [(k, v) for k, v in e["facts"].items()
                        if not (k.startswith("vllm:") or k in ("engine_inits_in_log", "prefix_cache_counters_note"))]
        if all(v == env_report.NA for k, v in setting_rows):
            h.append('<p class="na">No server log provided: all server settings are <span class="unk">unknown</span>.</p>')
            setting_rows = []
        else:
            h.append('<table><tr><th>Setting</th><th>Value</th><th>Source</th></tr>')
        for k, v in setting_rows:
            unk = v == env_report.NA
            h.append('<tr><td data-l="Setting">%s</td><td data-l="Value" class="%s">%s</td><td data-l="Source">%s</td></tr>'
                     % (E(k), "unk" if unk else "", E(UNKNOWN if unk else ("; ".join(v) if isinstance(v, list) else str(v))), E(e["sources"].get(k, ""))))
        if setting_rows:
            h.append('</table>')
        if e["windows"]:
            h.append('<table style="margin-top:12px"><tr><th>Cache series</th><th>Window hit rate</th><th>Status</th></tr>')
            for w in e["windows"]:
                val = ("%.1f%% (%d of %d queries)" % (w["hit_pct"], w["dh"], w["dq"])) if w["status"] == "ok" else "-"
                st = "ok" if w["status"] == "ok" else ('<span class="unk">%s</span>' % E(w.get("reason", w["status"])))
                h.append('<tr><td data-l="Cache series">%s</td><td data-l="Window hit rate">%s</td><td data-l="Status">%s</td></tr>' % (E(w["series"]), E(val), st))
            h.append('</table>')
        h.append('<p class="foot">Context from the provided files, not a diagnosis. Metadata never shows a hardware bottleneck.</p>')
    else:
        h.append('<p class="na">Not provided (no --log / --metrics): <span class="unk">unknown</span></p>')
    h.append('</div>')
    # 3 measured
    h.append('<div class="card"><h2>3. Measured before / after %s</h2>' % badge("MEASURED"))
    if "measured" in sec:
        m = sec["measured"]
        h.append('<p class="foot"><b>Scope:</b> the preset validated timestamp-move experiment only (ts_before / ts_after pair, '
                 '%d and %d requests, serial). <b>Not verified here:</b> that both runs used the same server instance with no restart '
                 'or cache clear between them (the analyzer checks that the metric series labels match and counters did not go down; '
                 'matching labels can repeat across restarts). No TTFT, cost or answer-equivalence claim.</p>'
                 % tuple(m["requests_per_variant"]))
        h.append('<table><tr><th></th><th>Tokens queried</th><th>Tokens hit</th><th>Hit rate</th></tr>')
        for lab, k in (("before (timestamp first)", "before"), ("after (timestamp moved)", "after")):
            d = m[k]
            h.append('<tr><td data-l="Run">%s</td><td data-l="Tokens queried">%d</td><td data-l="Tokens hit">%d</td><td data-l="Hit rate"><b>%s%%</b></td></tr>' % (E(lab), d["prompt_tokens_queried"], d["cached_tokens_hit"], d["hit_rate_pct"]))
        h.append('</table><p class="foot">%s Scope: one serial run of our synthetic workload on a real Muse Glimmer 30B engine, for the preset experiment only.</p>'
                 % E(m["verdict"].split(" Scope:")[0]))
    else:
        h.append('<p class="na">No measured pair provided: <span class="unk">unknown</span>. Measuring needs a served model and a before/after run.</p>')
    h.append('</div>')
    # 4 findings
    h.append('<div class="card"><h2>4. Findings, evidence and what to check next</h2><table>'
             '<tr><th>Finding</th><th>Class</th><th>Evidence</th><th>Unknown / next check</th></tr>')
    for f, cls, ev, nxt in res["findings"]:
        fc = ' class="unk"' if f.startswith("UNKNOWN") else ""
        h.append('<tr><td data-l="Finding"%s>%s</td><td data-l="Class">%s</td><td data-l="Evidence">%s</td><td data-l="Unknown / next check">%s</td></tr>' % (fc, E(f), badge(cls), E(ev), E(nxt)))
    h.append('</table></div>')
    if sum(1 for k in ("prompts", "environment", "measured") if k in sec) > 1:
        h.append('<p class="foot"><b>Inputs are not verified to describe the same server or traffic.</b> '
                 'Each section stands on its own evidence; CacheLens makes no cross-section cause claim.</p>')
    h.append('<p class="foot">%s Generated locally from the files you gave it; no network, no external assets. '
             'Machine-readable copy: report.json.</p>'
             % ("Prompt text is redacted in this report." if res["redacted_examples"] else
                "WARNING: this report contains fragments of prompt text (--show-examples). Do not share it publicly."))
    h.append('</main></body></html>')
    return "".join(h)

def main(argv):
    ap = argparse.ArgumentParser(description="CacheLens combined visual report")
    ap.add_argument("--prompts"); ap.add_argument("--log"); ap.add_argument("--metrics")
    ap.add_argument("--metrics-later"); ap.add_argument("--smi-file"); ap.add_argument("--mean-prompt-tokens", type=float)
    ap.add_argument("--pair-before"); ap.add_argument("--pair-after")
    ap.add_argument("--show-examples", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="label the report as SYNTHETIC example data")
    ap.add_argument("--out", default="cachelens-report")
    a = ap.parse_args(argv)
    try:
        res = build(a)
    except (prompt_diag.BadInput, env_report.BadInput, analyze.InvalidEvidence, ValueError, KeyError) as e:
        print(f"INCONCLUSIVE - {e}")
        return 2
    except OSError as e:
        print(f"INCONCLUSIVE - cannot read input: {e.strerror or e}")
        return 2
    except UnicodeDecodeError:
        print("INCONCLUSIVE - an input is not valid UTF-8 text")
        return 2
    os.makedirs(a.out, exist_ok=True)
    pub = {k: v for k, v in res.items() if not k.startswith("_")}
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(pub, f, indent=2)
    with open(os.path.join(a.out, "report.html"), "w", encoding="utf-8") as f:
        f.write(render(res))
    print("wrote %s and %s" % (os.path.join(a.out, "report.html"), os.path.join(a.out, "report.json")))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
