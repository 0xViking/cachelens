from PIL import Image, ImageDraw, ImageFont
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
MONO_R = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W,H = 1280,720
BG=(13,17,23); FG=(230,237,243); DIM=(139,148,158); GREEN=(63,185,80); RED=(248,81,73); YEL=(210,153,34); CYAN=(96,192,255)
def scene(lines, cap, dur, name, mfs=26, cfs=26, badge=True):
    img = Image.new("RGB",(W,H),BG); d = ImageDraw.Draw(img)
    d.rectangle([16,16,W-16,H-16],outline=(48,54,61),width=2)
    if badge:
        d.text((40,30),"CACHELENS - CAPTURED RUN REPLAY - Nov 3, 2026",font=ImageFont.truetype(SANS_B,20),fill=YEL)
        d.text((40,60),"Evidence: single synthetic serial run, Modal A100-80GB (GPU OFF in this replay)",font=ImageFont.truetype(SANS,17),fill=DIM)
    ty = 118
    for text,color,mf in lines:
        if mf: d.text((44,ty),text,font=ImageFont.truetype(MONO if color!=DIM else MONO_R,mfs),fill=color); ty += mfs+9
        else: d.text((44,ty),text,font=ImageFont.truetype(SANS,24),fill=color); ty += 36
    cy = H - (len(cap)*(cfs+10)) - 22
    for c in cap: d.text((40,cy),c,font=ImageFont.truetype(SANS_B,cfs),fill=FG); cy += cfs+10
    img.save(name); return name,dur

S=[]
# 1. Opening: name, model, problem, real layout (10s)
S.append(scene([
 ("CacheLens", CYAN, True),
 ("Cache-aware prompt-layout test harness", FG, False),
 ("for Meta Muse Glimmer 30B (open weights) on vLLM", FG, False),
 ("", FG, False),
 ("THE PROBLEM", YEL, False),
 ("Agent workloads resend the same long system prompt every request.", FG, False),
 ("A per-request timestamp placed BEFORE the shared prefix breaks", FG, False),
 ("prefix caching (prompts here: 25,640 chars / 5,796 tokens each):", FG, False),
 ("  [request_time] + system prompt + task  ->  cache misses", RED, True)],[],10,"v4b_s1.png"))
# 2. Broken variant: real command + real counters (14s)
S.append(scene([
 ("$ python3 probe.py --variant ts_before --out evidence/ts_before.jsonl", DIM, True),
 ("18 requests x 5,796 prompt tokens (104,328 queried)", FG, True),
 ("per-request cached=None: this vLLM build does not populate", DIM, True),
 ("usage.cached_tokens -> engine /metrics deltas are the measurement", DIM, True),
 ("", FG, False),
 ("ENGINE /metrics DELTA over the run:", FG, False),
 ("prefix-cache hits +1,088 of +104,328 queried  =  1.0% hit rate", RED, True),
 ("Muse Glimmer 30B, official muse-glimmer vLLM image, one A100-80GB", DIM, True)],
 ["Broken layout: 1.0% prefix-cache hit rate.", "Only 1,088 of 104,328 prompt tokens reused."],14,"v4b_s2.png"))
# 3. Analyzer output (10s)
S.append(scene([
 ("$ python3 analyze.py evidence/ts_before.jsonl evidence/ts_after.jsonl", DIM, True),
 ("", FG, False),
 ("first-divergence: char 28 of 25,640 (characters, not tokens)", CYAN, True),
 ("context: ...[request_time=2026-11-04T09:15:00+05:30]...", YEL, True),
 ("verdict: per-request value near the START of the prompt;", FG, True),
 ("prefix-cache reuse needs a byte-identical LEADING prefix", FG, True),
 ("diagnosis computed from the captured prompts themselves", DIM, True)],
 ["CacheLens finds where the shared prefix breaks,", "measured from the actual captured prompts."],10,"v4b_s3.png"))
# 4. The fix: real layouts both variants (10s)
S.append(scene([
 ("The fix: move the per-request field AFTER the shared prefix.", FG, False),
 ("", FG, False),
 ("ts_before:  [request_time] + system prompt + task", RED, True),
 ("ts_after:   system prompt + [request_time] + task", GREEN, True),
 ("", FG, False),
 ("Identical content - only the ordering changes.", FG, False)],
 ["Same workload, same model, same server - re-probe as ts_after."],10,"v4b_s4.png"))
# 5. Engine counter result: real numbers (14s)
S.append(scene([
 ("$ python3 probe.py --variant ts_after --out evidence/ts_after.jsonl", DIM, True),
 ("", FG, False),
 ("ENGINE /metrics DELTA: hits +97,696 of +104,328 queried", GREEN, True),
 ("hit rate: 1.0% -> 93.6%", GREEN, True),
 ("median per-request completion wall time: 3.69s -> 2.42s (34% lower,", GREEN, True),
 ("single run per variant, no TTFT)", GREEN, True),
 ("OBSERVED INCREASE in prefix-cache hit rate.", FG, True),
 ("This measurement does not prove a fix.", DIM, True)],
 ["93.6% hit rate from engine counter deltas,", "identical prompt volume both runs (104,328 tokens)."],14,"v4b_s5.png"))
# 6. Honest limits + close (12s)
S.append(scene([
 ("Honest limits of this evidence:", YEL, False),
 ("  one synthetic serial run, 18 requests per variant, one model/GPU", FG, True),
 ("  no cache reset between variants; no TTFT, cost, or answer-equivalence", FG, True),
 ("  cached=None = metric not populated by this build, not zero caching", FG, True),
 ("", FG, False),
 ("CacheLens: find where your prompts break prefix caching.", CYAN, False),
 ("Built for the Meta AI Developer Hackathon (Muse track).", FG, False)],[],12,"v4b_s6.png",mfs=24))

S.append(scene([
 ("Provenance", YEL, False),
 ("", FG, False),
 ("Before the building period (Oct 26):", FG, False),
 ("  idea + a separate SmolLM2 prototype (not in this repo)", DIM, True),
 ("", FG, False),
 ("Built for this repo (Nov 3):", FG, False),
 ("  new code, Muse Glimmer 30B runs, all evidence shown", GREEN, True),
 ("", FG, False),
 ("Comparison of recoverable old source found no substantive", DIM, False),
 ("shared code; missing old edits cannot be excluded.", DIM, False)],[],12,"v4_s7.png",mfs=26))
# put provenance before the closing scene
S=S[:-2]+[S[-1],S[-2]] if False else S
with open("concat.txt","w") as f:
    f.write("ffconcat version 1.0\n")
    order=S[:5]+[S[6],S[5]]
    for name,dur in order: f.write(f"file '{name}'\nduration {dur}\n")
    f.write(f"file '{order[-1][0]}'\n")
print(sum(d for _,d in order))
