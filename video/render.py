"""Render CacheLens demo scenes as PNGs (captured-run replay, captioned)."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG = (13, 17, 23); PANEL = (22, 27, 34); FG = (201, 209, 217)
ACCENT = (88, 166, 255); GREEN = (63, 185, 80); RED = (248, 81, 73)
DIM = (139, 148, 158); YELLOW = (210, 153, 34)

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
def f(sz, bold=False): return ImageFont.truetype(MONO_B if bold else MONO, sz)

def scene(term_lines, caption_lines, tag="CAPTURED RUN REPLAY - Nov 3, 2026"):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # terminal panel
    d.rounded_rectangle([40, 60, W-40, H-150], 10, fill=PANEL, outline=(48,54,61))
    for i, c in enumerate([(248,81,73),(210,153,34),(63,185,80)]):
        d.ellipse([60+24*i, 75, 74+24*i, 89], fill=c)
    d.text((120, 73), "cachelens - captured run replay", font=f(15), fill=DIM)
    y = 110
    for text, color, bold in term_lines:
        d.text((65, y), text, font=f(16, bold), fill=color)
        y += 26
    # caption bar
    d.rectangle([0, H-120, W, H], fill=(1, 4, 9))
    y = H - 108
    for line in caption_lines:
        d.text((50, y), line, font=f(22, True), fill=FG)
        y += 34
    d.text((W-470, H-34), tag, font=f(13), fill=DIM)
    return img

# Scene 1 - problem
scene([
    ("$ python3 probe.py --variant ts_before", DIM, False),
    ("", FG, False),
    ("18 agent requests. One shared 25.5KB runbook prefix.", FG, True),
    ("The prefix cache should absorb it.", FG, False),
    ("It doesn't - and nothing tells you why.", RED, False),
], ["Repeated prompt processing can waste GPU time.", "When a shared prefix stops caching, nothing tells you why."]).save("s1.png")

# Scene 2 - broken probe
rows = open("../evidence/ts_before.probe.log").read().splitlines()[:6]
tl = [("$ python3 probe.py --variant ts_before --out evidence/ts_before.jsonl", DIM, False)] + \
     [(r, FG, False) for r in rows] + [("... (18 requests)", DIM, False), ("", FG, False),
     ("engine /metrics: prefix-cache hits +1,088 of +104,328 tokens  =  1.0%", RED, True)]
scene(tl, ["Broken variant: 1.0% prefix-cache hit rate.", "104,328 prompt tokens processed, 1,088 reused."]).save("s2.png")

# Scene 3 - diagnosis
scene([
    ("$ python3 analyze.py evidence/ts_before.jsonl evidence/ts_after.jsonl", DIM, False),
    ("", FG, False),
    ('DIAGNOSIS:', YELLOW, True),
    ('  "rule": "first-divergence",', FG, False),
    ('  "diverges_at_char": 28,', ACCENT, True),
    ('  "context": "...[request_time=2026-11-04T09:15:00+05:30]', FG, False),
    ('                 You are FieldOps, an operati...",', FG, False),
    ('', FG, False),
    ('  The common leading prefix shared by ALL 18 requests', FG, False),
    ('  ends at character 28: a per-request timestamp sits', FG, False),
    ('  BEFORE the shared prefix.', RED, True),
], ["CacheLens diffs the captured prompts: the common prefix", "ends at char 28 - a timestamp placed BEFORE the prefix."]).save("s3.png")

# Scene 4 - fixed replay
scene([
    ("$ python3 probe.py --variant ts_after --out evidence/ts_after.jsonl", DIM, False),
    ("req-01: prompt=5796 cached=None 3.85s", FG, False),
    ("req-02: prompt=5796 cached=None 2.42s", FG, False),
    ("req-03: prompt=5796 cached=None 2.42s", FG, False),
    ("... (18 requests)", DIM, False),
    ("", FG, False),
    ("engine /metrics: prefix-cache hits +97,696 of +104,328 tokens = 93.6%", GREEN, True),
    ("", FG, False),
    ("OBSERVED INCREASE: 1.0% -> 93.6% on identical prompt volume", GREEN, True),
    ("(single synthetic serial run)", DIM, False),
], ["Move the timestamp AFTER the shared prefix:", "hit rate 1.0% -> 93.6%, median latency -34%."]).save("s4.png")

# Scene 5 - close
scene([
    ("CacheLens", ACCENT, True),
    ("", FG, False),
    ("capture  -  diagnose  -  verify prefix caching", FG, False),
    ("", FG, False),
    ("Muse Glimmer 30B (Meta open weights) + vLLM on one A100", FG, False),
    ("Built during the Meta Global AI Developer Hackathon", DIM, False),
], ["CacheLens - find why your prefix cache isn't hitting."]).save("s5.png")
print("rendered 5 scenes")
