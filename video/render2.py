"""Silent demo video v2. Frames: top source strip, real report crops (headless Chrome renders), caption."""
from PIL import Image, ImageDraw, ImageFont
import os
C = "crops/"
SANS="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; SANS_B="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO="/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"; MONO_R="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
W,H=1280,720
BG=(13,17,23); FG=(230,237,243); DIM=(139,148,158); GREEN=(63,185,80); RED=(248,81,73); YEL=(210,153,34); CYAN=(96,192,255); PURP=(167,139,250)
F=lambda p,s: ImageFont.truetype(p,s)
def frame(name, source, srccol, cap, img=None, lines=None, crop=None, mfs=24):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    d.rectangle([0,0,W,64],fill=(22,27,34))
    d.text((32,10),"CACHELENS",font=F(SANS_B,20),fill=CYAN)
    d.text((32,36),source,font=F(SANS_B,17),fill=srccol)
    capH=len(cap)*40+22
    top=64; bottom=H-capH
    if img:
        pic=Image.open(C+img).convert("RGB")
        if crop: pic=pic.crop(crop)
        avail_h=bottom-top-24; avail_w=W-60
        s=min(avail_w/pic.width, avail_h/pic.height)
        pic=pic.resize((int(pic.width*s),int(pic.height*s)),Image.LANCZOS)
        im.paste(pic,((W-pic.width)//2, top+12+(avail_h-pic.height)//2))
    if lines:
        y=top+30
        for t,col,mono in lines:
            if t=="": y+=18; continue
            if mono: d.text((50,y),t,font=F(MONO if col!=DIM else MONO_R,mfs),fill=col); y+=mfs+12
            else: d.text((50,y),t,font=F(SANS,27),fill=col); y+=42
    cy=H-capH+12
    d.rectangle([0,H-capH,W,H],fill=(22,27,34))
    for c in cap: d.text((32,cy),c,font=F(SANS_B,28),fill=FG); cy+=40
    im.save(name); return name
S=[]
REAL=("REAL captured evidence + real report render",GREEN)
S.append((frame("w1.png","Meta Global AI Developer Hackathon - Muse Glimmer 30B",YEL,
 ["Find why your prefix cache is not hitting."],
 lines=[("CacheLens",CYAN,True),("A diagnostic report for prefix-cache problems on vLLM",FG,False),("",FG,False),
 ("The problem",YEL,False),("Agents resend the same long system prompt every request.",FG,False),
 ("One per-request value near the START of the prompt breaks the reuse:",FG,False),
 ("[request_time] + system prompt + task   ->   cache misses",RED,True)]),10))
S.append((frame("w2.png","MEASURED on a real Muse Glimmer 30B engine (captured Nov 3)",GREEN,
 ["Broken layout: 1.0% prefix-cache hit rate.","1,088 of 104,328 prompt tokens reused."],
 lines=[("ts_before: timestamp first, 18 requests x 5,796 tokens",FG,True),("",FG,False),
 ("engine counter delta:",DIM,True),("prefix-cache hits +1,088 of +104,328 queried",RED,True),("= 1.0% hit rate",RED,True),("",FG,False),
 ("Measured synthetic workload on a real engine.",DIM,False)]),12))
S.append((frame("w3.png","SYNTHETIC EXAMPLE - the report on your own prompts file",PURP,
 ["Give it your own prompt file: it shows where the","shared prefix ends. This example is made up."],
 img="user_banner_card.png"),20))
S.append((frame("w4.png","REAL log from the captured Glimmer run (vllm.log) - parsed settings",GREEN,
 ["Server facts come from the log. Nothing is guessed:","GPU model is not in the log, so it stays unknown."],
 img="full_2.png",crop=(0,0,1404,560)),12))
S.append((frame("w5.png","REAL log - cache capacity and what stays unknown",GREEN,
 ["GPU KV cache: 776,150 tokens. Prefix caching: enabled.","No nvidia-smi was captured, so GPU model: unknown."],
 img="full_2.png",crop=(0,480,1404,818)),8))
S.append((frame("w6.png","MEASURED - real engine counters, same synthetic workload",GREEN,
 ["Moving the timestamp after the shared prefix:","1.0% -> 93.6% prefix-cache hit rate."],
 img="full_3.png"),20))
S.append((frame("w7.png","Report findings - evidence class and limits on every row",GREEN,
 ["Each finding names its evidence and what is not verified.","No cost, TTFT or answer-equivalence claim."],
 img="full_4.png"),18))
S.append((frame("w8.png","nvidia-smi: supported as an input file, NOT run on a GPU here",YEL,
 ["Run one fixed read-only command on the serving host,","pass the file in. Host identity stays unverified."],
 lines=[("nvidia-smi --query-gpu=index,name,memory.total,",FG,True),("   memory.used,driver_version --format=csv,noheader,nounits",FG,True),
 ("   > smi.csv",FG,True),("python3 cachelens.py --log vllm.log --smi-file smi.csv",CYAN,True),("",FG,False),
 ("It checks GPU memory against the log: consistent with, not proof of,",FG,False),("the same host. Tested on fixtures, not validated on a real GPU yet.",DIM,False)]),14))
S.append((frame("w9.png","Provenance and limits",YEL,
 ["Honest scope: one serial run per variant, 18 requests,","one model. Diagnosis only."],
 lines=[("Before the building period (Oct 26):",FG,False),("  the idea and a separate SmolLM2 prototype, not in this repo",DIM,True),
 ("Built for this repo: new code and the Muse Glimmer 30B runs",GREEN,False),
 ("Comparison of recoverable old source found no substantive shared",DIM,False),("code; missing old edits cannot be excluded.",DIM,False),("",FG,False),
 ("Both runs ran back-to-back in one server session (one engine start,",FG,False),("no cache-clear step). The analyzer cannot verify that from files alone.",FG,False)]),18))
S.append((frame("w10.png","CacheLens - Meta Global AI Developer Hackathon",CYAN,
 ["CacheLens: find where your prompts break prefix caching."],
 lines=[("Muse Glimmer 30B + vLLM prefix caching",FG,False),("1.0% -> 93.6%, measured from engine counters",GREEN,False),
 ("Report: prompt layout, server facts, measured result, next checks",FG,False)]),8))
with open("concat2.txt","w") as f:
    f.write("ffconcat version 1.0\n")
    for n,dur in S: f.write(f"file '{n}'\nduration {dur}\n")
    f.write(f"file '{S[-1][0]}'\n")
print(sum(d for _,d in S))
