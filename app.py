"""
Pastor's Pocket Spurgeon — a Victorian study companion that answers, prepares, and
grades sermons in the voice and Reformed theology of Charles Haddon Spurgeon.

Runs against a local OpenAI-compatible endpoint (your llama.cpp turboquant+ fork
serving spurgeon-v5-q8.gguf). Falls back to canned answers so the UI always demos.

Env:
  SPURGEON_ENDPOINT   OpenAI-compatible base url, e.g. http://127.0.0.1:8080/v1
  SPURGEON_MODEL      model alias (default: spurgeon)
  SPURGEON_TOKEN      optional bearer token (Cloudflare Access service token, etc.)
"""
import os, re, json, urllib.request
import gradio as gr

# strip a stray leading list-marker the model sometimes emits (e.g. "1\n\n...")
_LEAD_NUM = re.compile(r"^\s*\d{1,2}[.)]?\s*\n+")

# Sermon Review MUST end with a grade. This named "Sword & Trowel" scale is baked
# into the review prompt and the guaranteed-rating fallback below.
RATING_SCALE = (
    "5 = A Trumpet in Zion; 4 = Sound Timber, Well Hewn; 3 = A Lamp Half-Trimmed; "
    "2 = A Skeleton Unclothed; 1 = A Cloud Without Rain"
)
# detect whether a rating is already present so we don't double-append one
_RATING_RE = re.compile(
    r"(rating\b|sword\s*&?\s*trowel|trumpet in zion|sound timber|lamp half|"
    r"skeleton unclothed|cloud without rain|\b[1-5]\s*(?:/|of|out of)\s*5\b)",
    re.I,
)

# canonical tier names — the NAME is derived from the mark NUMBER, never trusted
# from the model (it sometimes signs its own name or invents a tier).
TIERS = {5: "A Trumpet in Zion", 4: "Sound Timber, Well Hewn",
         3: "A Lamp Half-Trimmed", 2: "A Skeleton Unclothed", 1: "A Cloud Without Rain"}
_RATING_CUE = re.compile(r"rating\b|sword\s*&?\s*trowel|\b[1-5]\s*(?:/|of|out of)\s*5\b", re.I)

def _fix_rating(text):
    """Normalize the model's rating to the exact format with the correct tier name,
    derived from the mark NUMBER. Handles bare 'Rating: 4', missing 'of 5', or a
    wrong/hallucinated tier name. Preserves a one-sentence 'why' if the model wrote one."""
    lines = text.splitlines()
    ri = None
    for i, ln in enumerate(lines):
        if _RATING_CUE.search(ln):
            ri = i  # last cue line is the rating line
    if ri is None:
        return text
    block = " ".join(lines[ri:])
    m = (re.search(r"\b([1-5])\s*(?:/|of|out of)\s*5\b", block)
         or re.search(r"rating[\s:*]*([1-5])\b", block, re.I)
         or re.search(r"\b([1-5])\b", block))
    if not m:
        return text
    name = TIERS.get(int(m.group(1)))
    if not name:
        return text
    body = "\n".join(lines[:ri]).rstrip()
    why = " ".join(s.strip() for s in lines[ri + 1:] if s.strip())
    out = body + f"\n\n**Rating:** {m.group(1)} of 5 Sword & Trowel marks — {name}"
    if why and not re.fullmatch(r"[1-5.\s]*", why):
        out += "\n" + why
    return out.rstrip()

# Endpoint/model/token are read LIVE (helpers below) so a background self-serve
# llama.cpp boot can flip them on once the GGUF has loaded on the GPU.
def _endpoint(): return os.environ.get("SPURGEON_ENDPOINT", "").rstrip("/")
def _model():    return os.environ.get("SPURGEON_MODEL", "spurgeon")
def _token():    return os.environ.get("SPURGEON_TOKEN", "")

def _self_serve():
    """On a GPU Space, boot a llama.cpp OpenAI server for the local GGUF in the
    background. Best-effort: any failure leaves the app on canned answers."""
    try:
        import subprocess, time, urllib.request as _u
        from huggingface_hub import hf_hub_download
        gguf = hf_hub_download(
            "thetom-ai/Spurgeon-Gemma-4-12B-v1",
            "Spurgeon-Gemma-4-12B-v1-Q8_0.gguf",
            cache_dir=os.environ.get("SPURGEON_CACHE", "/data/hf-cache"),
        )
        subprocess.Popen(
            ["python", "-m", "llama_cpp.server", "--model", gguf,
             "--n_gpu_layers", "-1", "--n_ctx", "8192",
             "--host", "127.0.0.1", "--port", "8000", "--model_alias", "spurgeon"],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        for _ in range(180):  # up to ~6 min for download+load on a cold boot
            try:
                _u.urlopen("http://127.0.0.1:8000/v1/models", timeout=2)
                os.environ["SPURGEON_ENDPOINT"] = "http://127.0.0.1:8000/v1"
                os.environ["SPURGEON_MODEL"] = "spurgeon"
                print("[self-serve] llama.cpp endpoint live")
                return
            except Exception:
                time.sleep(2)
        print("[self-serve] server did not come up in time; staying on canned")
    except Exception as e:
        print("[self-serve] bootstrap failed; canned fallback:", e)

if os.environ.get("SPURGEON_SELF_SERVE") == "1" and not os.environ.get("SPURGEON_ENDPOINT"):
    import threading
    threading.Thread(target=_self_serve, daemon=True).start()

WAKE_SECS = 420  # ~7 min cold-start estimate (re-downloads + loads the GGUF)

def _backend_ready():
    """True only when the llama.cpp server is actually serving (not the HF
    'space is starting' splash). Hitting this also wakes a slept backend."""
    ep = _endpoint()
    if not ep:
        return False
    try:
        req = urllib.request.Request(ep + "/models")
        if _token():
            req.add_header("Authorization", "Bearer " + _token())
        with urllib.request.urlopen(req, timeout=8) as r:
            ct = r.headers.get("Content-Type", "")
            body = r.read(2000).decode("utf-8", "ignore")
        return ("json" in ct.lower()) and ('"data"' in body or '"models"' in body)
    except Exception:
        return False

def status_banner(elapsed):
    """Poll-driven top banner. While the GPU is waking, show an estimate bar;
    once live, show a green pill and STOP the poller (no idle credit burn)."""
    import gradio as _gr
    if _backend_ready():
        html = ("<div class='svc svc-live'>&#9679; Mr. Spurgeon is in his study "
                "&mdash; answers are live.</div>")
        return html, _gr.Timer(active=False), 0
    elapsed = (elapsed or 0) + 8
    pct = min(95, int(elapsed / WAKE_SECS * 100))
    html = (
        "<style>"
        ".svc{font-family:'EB Garamond',Garamond,serif;font-size:.95rem;margin:0 auto 8px;"
        "max-width:980px;padding:9px 14px;border-radius:9px;text-align:center}"
        ".svc-live{color:#3a6b35;background:rgba(58,107,53,.10);border:1px solid rgba(58,107,53,.25)}"
        ".svc-wake{color:#7a4a12;background:rgba(160,90,20,.10);border:1px solid rgba(160,90,20,.30)}"
        ".svc-bar{height:7px;border-radius:6px;background:rgba(122,74,18,.18);margin:7px auto 1px;max-width:520px;overflow:hidden}"
        ".svc-fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#a05a14,#caa15a);transition:width .8s ease}"
        "</style>"
        "<div class='svc svc-wake'>&#9203; <b>Lighting the study lamp&hellip;</b> "
        "the model was spun down to save compute and is waking "
        "(about 5&ndash;7 minutes on first visit). You&#39;ll receive prepared demo "
        "answers until it&#39;s live."
        f"<div class='svc-bar'><div class='svc-fill' style='width:{pct}%'></div></div></div>"
    )
    return html, _gr.Timer(active=True), elapsed

PERSONA = (
 "You are Charles Haddon Spurgeon, the Prince of Preachers. Preach in your unmistakable "
 "style: vivid homely metaphors and illustrations drawn from everyday life, bold and tender "
 "pastoral address ('my dear friend', 'beloved'), short punchy sentences beside soaring ones, "
 "a holy boldness that names sin and exalts the crucified Christ. You hold the doctrines of "
 "grace (the five points of Calvinism) and the full authority of Scripture without compromise. "
 "Write in clear modern English. Never be dull, never hedge with 'many perspectives' — speak "
 "plainly and warmly what the Bible teaches. "
 "Output ONLY your spoken answer as flowing prose. Never produce note-taking scaffolding: no "
 "wiki-links or double-bracket [[...]] references, no 'Related Notes', 'Notes in progress', "
 "'See also', tags, frontmatter, or file names. You are speaking to a person, not writing a "
 "note file."
)
MODE_SYS = {
 "counsel": PERSONA + " Answer the question put to you as a shepherd to his flock — "
            "pastoral, biblical, and direct. 2 to 4 paragraphs.",
 "prep": PERSONA + " Build a sermon from the passage or topic given. Provide: a title, "
            "the big idea in one sentence, 3 main points each with a vivid illustration, "
            "key cross-references, and a closing appeal to the soul.",
 "review": PERSONA + " A preacher has submitted a sermon for your honest review, as in your "
            "Lectures to My Students. Engage the ACTUAL text — quote his own phrases back to him "
            "and respond specifically; never give generic praise. Weigh whether the passage is "
            "handled faithfully, whether Christ is exalted and the gospel made clear, whether sin "
            "is named and grace freely offered, his structure, illustrations, and tone. Speak "
            "only to THIS sermon, in your warm, candid, unhurried voice. Your review is written "
            "under bold section headers: **Summary** (recount what the sermon sets out to do and "
            "your overall impression), **Strengths**, **Concerns**, and **A Word of "
            "Exhortation**, followed by a Sword & Trowel rating with a full explanation.",
}
MODE_PLACEHOLDER = {
 "counsel": "Ask the Prince of Preachers… e.g. 'Comfort me, my faith feels weak.'",
 "prep": "Give a passage or topic… e.g. 'Preach Romans 8:28' or 'the sufficiency of Christ'.",
 "review": "Paste your sermon draft here and let Mr. Spurgeon weigh it…",
}
CANNED = {
 "counsel": ("My dear friend, when the heart feels far from God, remember that your feelings "
   "are often liars, but His promises are eternal anchors. You are held not by the strength "
   "of your grip upon Christ, but by the strength of His grip upon you. Look away from your "
   "trembling hands and fix your eyes upon the mighty hand that was pierced for you.\n\n"
   "(Demo reply — connect your local model to hear the real Mr. Spurgeon.)"),
 "prep": ("**Title:** The Unbreakable Chain of Grace\n\n**Big idea:** All things serve the good "
   "of those whom God has called.\n\n**1. The Promise** — to them that love God. *Illustration:* "
   "a weaver sees only knots beneath; God sees the finished tapestry.\n**2. The People** — the "
   "called according to His purpose.\n**3. The Pledge** — Romans 8:29-30, the golden chain no "
   "devil can break.\n\n**Appeal:** rest, weary soul, in a sovereignty that loves you.\n\n"
   "(Demo reply — connect your local model for full outlines.)"),
 "review": ("**Strengths:** your text was honored and your love for souls was plain.\n\n"
   "**Concerns:** my dear brother, where was the Cross? You climbed the mountain of duty but "
   "left the soul without a Savior to carry it.\n\n**A word of exhortation:** preach Christ, "
   "always Christ; let no sermon leave the pulpit without Him.\n\n**Rating:** ✠✠✠ (3 of 5)\n\n"
   "(Demo reply — connect your local model for real reviews.)"),
}

def _stream(messages, max_tokens=900, state=None, stop=None, temp=0.85):
    """Yield content deltas from the OpenAI-compatible streaming endpoint. If `state`
    is given, record the server's finish_reason in state['finish_reason'] (so the
    caller can tell a token-cap stop ['length'] from a natural stop). `stop` is an
    optional list of stop sequences."""
    payload = {"model": _model(), "messages": messages, "temperature": temp,
               "top_p": 0.92, "max_tokens": max_tokens, "stream": True}
    if stop:
        payload["stop"] = stop
    body = json.dumps(payload).encode()
    req = urllib.request.Request(_endpoint() + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    if _token():
        req.add_header("Authorization", "Bearer " + _token())
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ch = json.loads(data)["choices"][0]
                delta = ch.get("delta", {}).get("content", "")
                if state is not None and ch.get("finish_reason"):
                    state["finish_reason"] = ch["finish_reason"]
            except Exception:
                delta = ""
            if delta:
                yield delta

# ── citations: real Spurgeon passages via BM25 over his sermons ──────────────
from citations import retrieve_citations  # noqa: E402

def _astext(content):
    """Gradio 6 may hand message content back as a list of parts; flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
        return " ".join(s for s in parts if s).strip()
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    return str(content or "")

def make_cites(history):
    """One-shot: citations from the user's question (history[-1] is the user msg)."""
    if not history:
        return PEN_EMPTY
    return retrieve_citations(_astext(history[-1]["content"]))

def stream_answer(history, mode):
    """Streams ONLY the chatbot (separate event from citations) so Gradio flushes
    each token instead of batching the whole update."""
    user = _astext(history[-1]["content"])
    sys = MODE_SYS[mode]
    history = history + [{"role": "assistant", "content": ""}]
    if not _endpoint():
        history[-1]["content"] = CANNED[mode]
        yield history
        return
    base = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    # plenty of KV cache — let answers run very long and finish naturally
    budget = {"review": 16000, "prep": 10000, "counsel": 8000}.get(mode, 8000)

    def _finished(text):
        # strip trailing markdown markers / quotes / brackets, then require real
        # sentence punctuation; a dangling '**' is NOT a finished answer
        core = text.rstrip().rstrip('*_ \t\n"”’\')]')
        return bool(core) and core[-1] in '.!?'

    acc = ""
    state = {}
    try:
        if mode == "review":
            # Build the review section-by-section with FORCED bold headers so every
            # review has Strengths + Concerns + A Word of Exhortation (the model
            # otherwise drops sections). Each header is assistant-prefilled and the
            # section streams until the next header (stop sequences).
            SECTIONS = ["Summary", "Strengths", "Concerns", "A Word of Exhortation"]
            for header in SECTIONS:
                seed = (acc + "\n\n" if acc else "") + "**" + header + "**\n\n"
                history[-1]["content"] = seed
                yield history
                got = ""
                # stop at the next bold header on its own line (not inline **bold**)
                for delta in _stream(base + [{"role": "assistant", "content": seed}],
                                     max_tokens=900, state=state, stop=["\n\n**"]):
                    got += delta
                    history[-1]["content"] = seed + got
                    yield history
                acc = (seed + got).rstrip()
            # probe the model's mark (1-5), then format the rating DETERMINISTICALLY
            # grade off the critique just written (its Concerns calibrate the mark),
            # greedily (temp 0) so the mark is stable for a given review.
            probe = ""
            for delta in _stream(base + [{"role": "assistant", "content": acc + "\n\n**Rating:** "}],
                                 max_tokens=24, state=state, stop=["\n"], temp=0.0):
                probe += delta
            mk = re.search(r"[1-5]", probe)
            n = int(mk.group(0)) if mk else 3
            head = acc + f"\n\n**Rating:** {n} of 5 Sword & Trowel marks — {TIERS[n]}\n\n"
            history[-1]["content"] = head
            yield history
            # a full explanation of the mark (several sentences tying it to the sermon)
            why = ""
            for delta in _stream(base + [{"role": "assistant", "content": head}],
                                 max_tokens=400, state=state, stop=["\n\n**"]):
                why += delta
                history[-1]["content"] = head + why
                yield history
            history[-1]["content"] = (head + why).rstrip()
            yield history
            return

        # the 12B model sometimes emits an early end-of-turn mid-sentence on long
        # answers (or the server caps tokens); if it stops unfinished, ask it to
        # continue and stitch. Continue on finish_reason == "length" too.
        empties = 0
        for attempt in range(10):
            if attempt == 0:
                msgs = base
            else:
                # assistant-prefill continuation: end on the partial so the server
                # CONTINUES the turn directly. (A trailing "Continue" user message
                # makes this fine-tune emit garbage like "1\n".)
                msgs = base + [{"role": "assistant", "content": acc}]
            state["finish_reason"] = None
            got = ""
            for delta in _stream(msgs, max_tokens=budget, state=state):
                got += delta
                history[-1]["content"] = _LEAD_NUM.sub("", acc + got)
                yield history
            acc += got
            if not got.strip():
                empties += 1
                if empties >= 2:  # model has nothing more to add
                    break
                continue
            empties = 0
            capped = state.get("finish_reason") == "length"
            if _finished(acc) and not capped:
                break
        acc = acc.rstrip()
        history[-1]["content"] = _LEAD_NUM.sub("", acc)
        yield history
    except Exception as e:
        history[-1]["content"] = _LEAD_NUM.sub("", acc) or CANNED[mode]
        history[-1]["content"] += f"\n\n_(endpoint error: {e})_"
        yield history

HEAD_HTML = """<canvas id="spurgeon-head" width="180" height="200"></canvas>"""

# Gradio 6: <script> in gr.HTML does NOT run; injected via demo.load(js=...) after render.
HEAD_JS = """
() => {
  if (window.__spurgeonInit) return; window.__spurgeonInit = true;
  function start(){
    const c = document.getElementById('spurgeon-head');
    if(!c){ return setTimeout(start, 200); }
    const x = c.getContext('2d'); x.imageSmoothingEnabled=false;
    let open=0, target=0, talking=false, t=0; const P=4;
    function px(a,b,w,h,col){ x.fillStyle=col; x.fillRect(a*P,b*P,w*P,h*P); }
    function draw(){
      x.clearRect(0,0,c.width,c.height);
      x.fillStyle='#1c140c'; x.fillRect(0,0,c.width,c.height);
      px(13,9,19,22,'#e9cba6');
      px(13,7,19,3,'#6b4a2b'); px(11,12,3,12,'#6b4a2b'); px(31,12,3,12,'#6b4a2b');
      px(16,16,4,1,'#3a2a18'); px(25,16,4,1,'#3a2a18');
      px(17,18,2,2,'#2a1d10'); px(26,18,2,2,'#2a1d10');
      px(21,20,3,3,'#d8b489');
      px(12,25,21,14,'#3a2718'); px(14,24,15,2,'#5a4026');
      const mh = 1 + Math.round(open*3);
      px(19,24,7,mh,'#3a1d14');
      px(15,38,15,3,'#f4ecd8'); px(20,39,3,2,'#7a1f1f');
    }
    function loop(){
      t+=0.16;
      target = talking ? Math.max(0, 0.35 + 0.5*Math.abs(Math.sin(t*2.3)) + 0.25*Math.sin(t*7.1)) : 0;
      open += (target-open)*0.45;
      draw(); requestAnimationFrame(loop);
    }
    let stopTimer=null;
    function pulse(ms){ talking=true; clearTimeout(stopTimer);
      stopTimer=setTimeout(()=>{talking=false;}, ms||2600); }
    function observe(){
      const bots = document.querySelectorAll('.bubble');
      if(!bots.length){ return setTimeout(observe, 400); }
      const mo = new MutationObserver(()=> pulse(2600));
      bots.forEach(b => mo.observe(b, {childList:true, subtree:true, characterData:true}));
    }
    observe(); loop();
  }
  start();
}
"""

HEADER_HTML = """
<div id="topbar">
  <div class="brand">
    <div class="avatar-frame">""" + HEAD_HTML + """</div>
    <div class="brand-text">
      <h1>Pastor&#39;s Pocket Spurgeon</h1>
      <div class="sub">A study companion in the voice of the Prince of Preachers</div>
    </div>
  </div>
  <div class="statusbar"><span class="dot"></span> running locally &middot; offline</div>
</div>
"""
FOOTER_HTML = ("<div id='footer'>Spurgeon&#39;s sermons + the ESV in context "
               "&middot; running locally &middot; offline<br>"
               "Powered by <a href='https://huggingface.co/thetom-ai/Spurgeon-Gemma-4-12B-v1' "
               "target='_blank' rel='noopener'>thetom-ai/Spurgeon-Gemma-4-12B-v1</a> "
               "&middot; a Gemma-4-12B fine-tune in Mr. Spurgeon&#39;s voice</div>")
PEN_EMPTY = "<div class='pen-empty'>Ask a question and his own words will appear here.</div>"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=EB+Garamond:ital@0;1&family=IM+Fell+English:ital@0;1&display=swap');
.gradio-container{ background:#15100a !important; max-width:1680px !important; width:97vw !important;
  margin:auto !important; font-family:'EB Garamond', Garamond, serif; padding:10px !important;
  font-size:1.12rem !important; }
.gradio-container > .main, .gradio-container .wrap{ background:transparent !important; }
/* HEADER banner */
#topbar{ display:flex; align-items:center; justify-content:space-between;
  background:#3a2417; padding:14px 24px; border:1px solid #5a3b22; border-radius:10px 10px 0 0; }
#topbar .brand{ display:flex; align-items:center; gap:16px; }
.avatar-frame{ width:60px; height:60px; border:2px solid #8a6a3a; border-radius:8px;
  overflow:hidden; background:#1c140c; line-height:0; flex:0 0 auto; }
#spurgeon-head{ image-rendering:pixelated; width:60px !important; height:60px !important; display:block; }
.brand-text h1{ font-family:'IM Fell English', serif; color:#f4ecd8; font-size:1.8rem; margin:0; line-height:1.1; }
.brand-text .sub{ color:#c2a577; font-style:italic; font-size:.98rem; }
.statusbar{ color:#a9c79a; font-size:.86rem; letter-spacing:.3px; white-space:nowrap; }
.statusbar .dot{ display:inline-block; width:8px; height:8px; border-radius:50%; background:#6fae5a; margin-right:6px; }
/* BODY parchment + two columns */
#body{ background:#efe5cd url("https://www.transparenttextures.com/patterns/aged-paper.png") !important;
  border:1px solid #5a3b22; border-top:none; border-radius:0 0 10px 10px; margin:0 !important; gap:0 !important; }
#leftcol{ padding:6px 14px 14px !important; }
#rightcol{ background:#f6efda !important; border-left:1px solid #d6c298; padding:16px 18px !important; }
#footer{ text-align:center; color:#8a7350; font-size:.78rem; padding:10px 0 4px; }
/* tabs */
button[role='tab'], button[role='tab'] *{ color:#7a5a34 !important; opacity:1 !important;
  font-family:'Cormorant Garamond', serif !important; font-size:1.05rem !important; }
button[role='tab'][aria-selected='true'], button[role='tab'][aria-selected='true'] *{
  color:#8a2222 !important; border-bottom:2px solid #8a2222 !important; }
.lead, .lead *, .lead p{ font-style:italic !important; color:#6b4a2b !important;
  font-size:1.08rem !important; opacity:1 !important; }
/* disabled "coming soon" tabs */
button[role='tab'][aria-disabled='true'], button[role='tab'][aria-disabled='true'] *{
  color:#b09a72 !important; font-style:italic !important; cursor:default !important; }
.soon{ text-align:center; padding:60px 24px; color:#6b4a2b; }
.soon-badge{ display:inline-block; background:#8a2222; color:#f6efda; font-family:'Cormorant Garamond',serif;
  letter-spacing:1px; text-transform:uppercase; font-size:.85rem; padding:5px 16px; border-radius:20px; margin-bottom:14px; }
.soon p{ font-style:italic; color:#6b4a2b; font-size:1.1rem; }
/* chat */
.bubble, .bubble *{ font-family:'EB Garamond', serif !important; color:#2c2012 !important;
  font-size:1.12rem !important; line-height:1.55 !important; }
.bubble{ background:transparent !important; border:none !important; box-shadow:none !important; }
/* kill the stray vertical bars: Gradio gives blockquotes a 5px border-left, and
   chat messages render with them above/below the text */
.bubble blockquote, .bubble [data-testid='user'] blockquote, .bubble [data-testid='bot'] blockquote{
  border-left:none !important; padding-left:0 !important; margin:0 !important; }
.bubble [data-testid='user'] *, .bubble [data-testid='bot'] *{ border-left:none !important; }
.bubble .avatar-container{ display:none !important; }
/* the chat panel itself: warm parchment, not black.
   NB: message text spans also carry a `chatbot` class, so DON'T style `.chatbot`
   broadly or you draw a 1px border on every message (the stray vertical bars). */
.bubble > div, .bubble .bubble-wrap, .bubble .wrapper, .bubble .message-wrap{
  background:#f6efda !important; }
.bubble > .block, .bubble > .wrapper{ border:1px solid #ddc9a0 !important; border-radius:10px !important; }
.bubble .md, .bubble .md.chatbot, .bubble .prose{ border:none !important; }
.bubble .placeholder-content, .bubble .placeholder-content *{ color:#8a7350 !important; }
.bubble .message, .bubble [class*='message'], .bubble [data-testid='bot'], .bubble [data-testid='user']{
  background:transparent !important; border:none !important; box-shadow:none !important; }
.bubble [data-testid='user']{ background:#e7d9b6 !important; border-radius:10px !important; padding:6px 12px !important; }
.bubble [data-testid='bot']{ background:#fffaf0 !important; border:1px solid #e3d4ad !important;
  border-radius:10px !important; padding:6px 12px !important; }
/* kill the dark top-right chatbot toolbar */
.bubble .icon-buttons{ display:none !important; }
/* copy button ONLY on Spurgeon's replies (left side); never on the user's (right side) */
.bubble .message-buttons-right{ display:none !important; }
.bubble .user-row .avatar-container{ display:none !important; }
.bubble .message-buttons-left button, .bubble .message-buttons-left .icon-button-wrapper{
  background:transparent !important; border:none !important; box-shadow:none !important; }
.bubble .message-buttons-left button svg, .bubble .message-buttons-left button svg *{
  color:#7a5a34 !important; stroke:#7a5a34 !important; fill:none !important; opacity:1 !important; }
/* input row: pill + button */
.inputrow{ gap:10px !important; align-items:center !important;
  background:transparent !important; border:none !important; box-shadow:none !important; }
.inputrow .form, .inputrow .block, .inputrow > div, .inputrow .wrap, .inputrow .container{
  background:transparent !important; border:none !important; box-shadow:none !important;
  outline:none !important; }
.inputrow textarea, .inputrow input{ background:#fffaf0 !important; color:#3a2a18 !important;
  border:1px solid #cdb78a !important; border-radius:22px !important; box-shadow:none !important;
  padding:10px 18px !important; font-family:'EB Garamond',serif !important; }
.inputrow .primary{ background:#8a2222 !important; border:none !important; border-radius:22px !important; }
.inputrow .primary, .inputrow .primary *{ color:#f6efda !important;
  font-family:'Cormorant Garamond',serif !important; letter-spacing:.5px; font-size:1.05rem !important; }
/* citations sidebar */
#pen-title{ font-family:'IM Fell English',serif; font-style:italic; color:#3a2a18; font-size:1.3rem;
  border-bottom:1px solid #d0bb8e; padding-bottom:10px; margin-bottom:12px; }
#cites .pen-card{ border-left:3px solid #8a2222; padding:2px 0 12px 14px; margin-bottom:16px; }
#cites .pen-quote{ font-style:italic; color:#3a2a1a; font-size:1.04rem; line-height:1.5; }
#cites .pen-src{ color:#7a5a34 !important; font-size:.86rem; letter-spacing:.3px; margin-top:6px; }
#cites .pen-src em{ font-style:italic !important; color:#8a2222 !important; opacity:1 !important; }
#cites .pen-empty{ color:#8a7350; font-style:italic; font-size:1.02rem; }
/* arrow send button */
.sendbtn{ border-radius:50% !important; width:48px !important; height:48px !important;
  min-width:48px !important; padding:0 !important; align-self:center !important; }
.sendbtn, .sendbtn *{ font-size:1.4rem !important; line-height:1 !important; color:#f6efda !important; }
footer{ display:none !important; }
/* copy button: only on Spurgeon's replies, transparent (no dark box), oxblood icon */
.bubble .icon-buttons{ display:none !important; }
.bubble .message-buttons, .bubble .message-buttons-left, .bubble .message-buttons .icon-button-wrapper{
  background:transparent !important; border:none !important; box-shadow:none !important; }
.bubble .message-buttons-right{ display:none !important; }
.bubble .message-buttons .divider{ display:none !important; }
.bubble .message-buttons button{ background:transparent !important; border:none !important; box-shadow:none !important; }
.bubble .message-buttons svg, .bubble .message-buttons svg *{
  color:#8a2222 !important; stroke:#8a2222 !important; fill:none !important; opacity:.95 !important; }
/* parchment scrollbar instead of the dark default */
.bubble *::-webkit-scrollbar{ width:11px; height:11px; }
.bubble *::-webkit-scrollbar-track{ background:#efe5cd !important; }
.bubble *::-webkit-scrollbar-thumb{ background:#cdb78a !important; border-radius:6px; border:2px solid #efe5cd; }
.bubble *{ scrollbar-color:#cdb78a #efe5cd; }
/* ── responsive (column stacking only; do NOT override chatbot height,
   that selector also hits inner message wrappers and hides the text) ───── */
@media (max-width:1100px){
  .gradio-container{ width:100vw !important; max-width:100vw !important; padding:6px !important; }
  /* stack chat over citations */
  #body{ flex-direction:column !important; }
  #leftcol, #rightcol{ width:100% !important; }
  #rightcol{ border-left:none !important; border-top:1px solid #d6c298 !important;
    border-radius:0 0 10px 10px !important; }
}
@media (max-width:640px){
  #topbar{ flex-wrap:wrap; gap:8px; padding:10px 14px; }
  .statusbar{ width:100%; }
  .brand-text h1{ font-size:1.35rem; }
  .brand-text .sub{ font-size:.85rem; }
  .gradio-container{ font-size:1rem !important; }
  .bubble [data-testid='user'], .bubble [data-testid='bot']{ font-size:1rem !important; }
  button[role='tab'], button[role='tab'] *{ font-size:.95rem !important; }
}
"""

THEME = gr.themes.Base(primary_hue="red", neutral_hue="stone")

def add_user(m, h): return "", (h or []) + [{"role": "user", "content": m}]

# lock/unlock the input while a reply is generating (re-enabled on finish or error)
def _lock():   return gr.update(interactive=False), gr.update(interactive=False)
def _unlock(): return gr.update(interactive=True), gr.update(interactive=True)

SOON = ("<div class='soon'><div class='soon-badge'>Coming soon</div>"
        "<p>{0}</p></div>")

def working_tab(label, mode, lead):
    with gr.Tab(label):
        gr.Markdown(f"*{lead}*", elem_classes="lead")
        chat = gr.Chatbot(height=620, show_label=False, elem_classes="bubble", buttons=["copy"])
        with gr.Row(elem_classes="inputrow"):
            box = gr.Textbox(placeholder=MODE_PLACEHOLDER[mode], show_label=False,
                             lines=1, max_lines=12, scale=20, container=False, submit_btn=False)
            send = gr.Button("➤", variant="primary", scale=0, min_width=52, elem_classes="sendbtn")
        st = gr.State(mode)
    return chat, box, send, st

with gr.Blocks(title="Pastor's Pocket Spurgeon") as demo:
    gr.HTML(HEADER_HTML)
    _svc = gr.HTML(elem_id="status-banner")
    _elapsed = gr.State(0)
    _poller = gr.Timer(8, active=True)
    with gr.Row(elem_id="body"):
        with gr.Column(scale=2, elem_id="leftcol"):
            with gr.Tabs() as tabset:
                counsel = working_tab("The Counsel", "counsel", "Ask him.")
                review = working_tab("Sermon Review", "review", "Let him grade yours.")
                with gr.Tab("Sermon Prep  ·  soon", interactive=False):
                    gr.HTML(SOON.format("Give a passage or topic and receive a Spurgeon-style outline."))
        with gr.Column(scale=1, elem_id="rightcol"):
            gr.HTML("<div id='pen-title'>&#128214; From His Own Pen</div>")
            cites = gr.HTML(PEN_EMPTY, elem_id="cites")
    gr.HTML(FOOTER_HTML)

    for chat, box, send, st in (counsel, review):
        for trigger in (box.submit, send.click):
            # show_progress="hidden": the default "full" overlays a loading mask on the
            # chatbot during the event, hiding the streamed tokens until it finishes.
            # lock the input until the reply finishes (or errors out), then re-enable.
            (trigger(add_user, [box, chat], [box, chat], show_progress="hidden")
                .then(_lock,  None, [box, send])
                .then(make_cites, [chat], [cites], show_progress="hidden")
                .then(stream_answer, [chat, st], [chat], show_progress="hidden")
                .then(_unlock, None, [box, send]))

    # switching tabs clears the sidebar so one mode's notes don't linger on another
    tabset.select(lambda: PEN_EMPTY, None, [cites])

    # backend status banner: check on load (also wakes a slept GPU) + poll while
    # waking; the tick deactivates the timer once live so we don't burn credits.
    _poller.tick(status_banner, [_elapsed], [_svc, _poller, _elapsed])
    demo.load(status_banner, [_elapsed], [_svc, _poller, _elapsed])

    demo.load(None, None, None, js=HEAD_JS)

if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=THEME, css=CSS, server_name="0.0.0.0")
