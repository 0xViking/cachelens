"""CacheLens prompt-file diagnostic (text level, no server needed).

Reads a JSONL file of prompts that one app sent to one model and reports where
their shared leading text ends and what varies around that point.

This is a TEXT-LEVEL diagnosis. It does not measure the engine's cache, does
not count tokens (character offsets only), and does not predict a hit rate.
Prefix caches reuse a byte-identical LEADING prefix, so content that varies
early in the prompt limits how much of the later, stable text can be reused.

Usage: python3 analyze.py --prompts prompts.jsonl
Input rows: {"prompt": "text"}  or  {"messages": [{"role": "...", "content": "text"}, ...]}
"""
import json, os.path, re

class BadInput(Exception):
    pass

_TS = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_HEX = re.compile(r"\b[0-9a-fA-F]{12,}\b")
_NUM = re.compile(r"\d+")

def _content_text(c, i):
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c:
        texts = []
        for part in c:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            else:
                raise BadInput(f"row {i}: only text content parts are supported "
                               "(found a non-text part such as an image or tool call)")
        return "\n".join(texts)
    raise BadInput(f"row {i}: message content must be a string or a list of text parts")

def render(row, i):
    if not isinstance(row, dict):
        raise BadInput(f"row {i}: expected a JSON object")
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            if not (isinstance(m, dict) and isinstance(m.get("role"), str)):
                raise BadInput(f"row {i}: each message needs a string role")
            parts.append(f"{m['role']}: {_content_text(m.get('content'), i)}")
        return "\n".join(parts)
    raise BadInput(f"row {i}: need a string 'prompt' or a non-empty 'messages' list")

def load(path):
    prompts = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        prompts.append(render(json.loads(line), i))
                    except json.JSONDecodeError as e:
                        raise BadInput(f"line {i}: invalid JSON ({e.msg})")
    except OSError as e:
        raise BadInput(f"cannot read {path}: {e.strerror}")
    except UnicodeDecodeError:
        raise BadInput(f"{path} is not valid UTF-8 text")
    return prompts

def looks_like(line):
    if _TS.search(line): return "timestamp-like"
    if _UUID.search(line): return "uuid-like"
    if _HEX.search(line): return "long-hex-id-like"
    if _NUM.search(line): return "contains-number"
    return "free-text"

def _clip(s, n=90):
    s = s.replace("\t", " ")
    return s if len(s) <= n else s[:n - 3] + "..."

def pairwise_prefix_stats(prompts):
    """Common-prefix length (chars) across prompt pairs. Sampled if large."""
    vals = []
    step = max(1, len(prompts) // 40)
    for i in range(0, len(prompts), step):
        for j in range(i + 1, len(prompts), step):
            vals.append(len(os.path.commonprefix([prompts[i], prompts[j]])))
    vals.sort()
    if not vals:
        return None
    return {"min_chars": vals[0], "median_chars": vals[len(vals) // 2],
            "max_chars": vals[-1], "pairs_sampled": len(vals)}

def diagnose(prompts, redact=False):
    if len(prompts) < 2:
        raise BadInput("need at least 2 prompts to compare")
    if any(not p for p in prompts):
        raise BadInput("empty prompt found")
    n = len(prompts)
    lens = [len(p) for p in prompts]
    cp = len(os.path.commonprefix(prompts))
    pw = pairwise_prefix_stats(prompts)
    out = {"prompts": n, "mean_chars": round(sum(lens) / n, 1),
           "leading_shared_chars": cp,
           "leading_shared_pct_of_mean_prompt": round(100.0 * cp / (sum(lens) / n), 1),
           "unit_note": "character offsets, not tokens; text-level, not a measured cache hit rate"}
    if pw:
        out["pairwise_common_prefix_chars"] = pw
        out["pairwise_common_prefix_chars"]["meaning"] = (
            "how much leading text any two prompts share; a low median means the "
            "break is early for most pairs, not just one outlier")
    if all(p == prompts[0] for p in prompts):
        out["finding"] = "identical"
        out["verdict"] = ("All prompts are byte-identical, so prompt text gives no reason "
                          "for prefix misses. Look at server flags, cache size/eviction, "
                          "block granularity, or a cold cache.")
        return out
    lines = [p.split("\n") for p in prompts]
    d = 0
    while all(len(l) > d for l in lines) and len({l[d] for l in lines}) == 1:
        d += 1
    div_vals = []
    for l in lines:
        v = l[d] if d < len(l) else "<prompt ends>"
        if v not in div_vals:
            div_vals.append(v)
    out["first_divergence"] = {
        "line_index": d, "char_offset": cp,
        "distinct_values_at_that_line": n if len(div_vals) >= n else len(div_vals),
        "examples": (["[redacted by --no-examples]"] * min(4, len(div_vals))
                     if redact else [_clip(v) for v in div_vals[:4]]),
        "looks_like": looks_like(div_vals[0]),
    }
    stable = set.intersection(*[set(l) for l in lines]) - {""}
    l0 = lines[0]
    after_stable = sum(len(x) + 1 for x in l0[d + 1:] if x in stable)
    out["stable_text_after_first_divergence"] = {
        "chars_in_first_prompt": after_stable,
        "pct_of_first_prompt": round(100.0 * after_stable / len(prompts[0]), 1),
        "meaning": ("lines present in every prompt that sit after the first differing "
                    "line, so a leading-prefix cache cannot reuse them across these requests"),
    }
    varying = [(i, x) for i, x in enumerate(l0) if x not in stable and x != ""]
    def distinct_at(idx):
        vals = {l[idx] for l in lines if idx < len(l)}
        return len(vals)
    out["varying_lines_in_first_prompt"] = [
        {"line_index": i, "looks_like": looks_like(x),
         "example": "[redacted by --no-examples]" if redact else _clip(x),
         "distinct_values": distinct_at(i),
         "cardinality_note": (("one value per prompt (consistent with a per-request id or timestamp)"
                              if n >= 5 else
                              "one value per prompt, but with only %d prompts this does not show a per-request id" % n)
                              if distinct_at(i) >= n else
                              "%d distinct values across %d prompts (a small set, e.g. a user or tenant field)"
                              % (distinct_at(i), n))}
        for i, x in varying[:6]]
    if len(varying) > 6:
        out["more_varying_lines_not_shown"] = len(varying) - 6
    same_set = all(sorted(l) == sorted(lines[0]) for l in lines)
    out["same_lines_different_order"] = bool(same_set)
    if same_set:
        out["finding"] = "reordered-lines"
        out["verdict"] = ("Every prompt has the same lines, but not in the same order. "
                          "Ordering alone ends the shared leading text at char %d." % cp)
    elif cp < 0.5 * min(lens):
        out["finding"] = "early-variation"
        out["heuristic"] = ("'early' means the shared text ends before 50% of the shortest prompt; "
                            "a fixed rule of thumb, not a measured threshold - a later break can still cost reuse")
        out["verdict"] = ("The shared leading text ends early (char %d, line %d): that line "
                          "takes %d distinct value(s) across these prompts, and %.1f%% of the "
                          "first prompt is stable text that follows it. If this server also "
                          "misses its prefix cache, this layout is one candidate cause to "
                          "check - alongside server flags, cache size/eviction and block "
                          "granularity, which prompt text alone cannot show." %
                          (cp, d, out["first_divergence"]["distinct_values_at_that_line"],
                           out["stable_text_after_first_divergence"]["pct_of_first_prompt"]))
    else:
        out["finding"] = "late-variation"
        out["verdict"] = ("The prompts share %.1f%% of their text as a leading prefix; the first "
                          "difference is late (char %d, line %d). The prompt layout is not an "
                          "obvious cause of misses." % (out["leading_shared_pct_of_mean_prompt"], cp, d))
    out["caution"] = ("This does not prove why a server missed its cache and does not say what "
                      "the app should change. Moving or reordering prompt content can change "
                      "model behavior: the app owner must decide and re-check outputs.")
    return out

def main(path, redact=False):
    import sys
    try:
        res = diagnose(load(path), redact=redact)
    except BadInput as e:
        print(f"INCONCLUSIVE - {e}")
        return 2
    print("== CacheLens prompt diagnostic (text level) ==")
    print(json.dumps(res, indent=2))
    return 0
