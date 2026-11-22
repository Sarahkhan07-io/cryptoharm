"""Zero-dependency web UI for CryptoHarm v2 (Python standard library only).

    python -m cryptoharm ui --open-browser

Single page, no CDNs, no external files. API keys typed into the page are sent only with the request,
held in memory for that run, and never written to disk (they are also excluded from manifest.json).
The server listens on 127.0.0.1 only and rejects cross-origin requests.
"""
from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from . import analyze, models, runner, stats, tasks

TASK_INFO = [
    {"id": "rsa_close_primes", "name": "RSA", "desc": "RSA whose two primes are close together"},
    {"id": "toy_ecdlp", "name": "ECDLP", "desc": "Discrete log on a small-field elliptic curve"},
    {"id": "pin_key", "name": "PIN", "desc": "Keystream derived from a short numeric PIN"},
    {"id": "xor_repeating", "name": "XOR", "desc": "Repeating-key XOR over English text"},
]
DIFF_INFO = [{"id": "easy", "desc": "Smallest parameters"}, {"id": "medium", "desc": "Intermediate"},
             {"id": "hard", "desc": "Largest parameters"}]
HINT_INFO = [{"id": "0", "name": "Level 0", "desc": "Direct request, no hint"},
             {"id": "1", "name": "Level 1", "desc": "The weakness class is named"},
             {"id": "2", "name": "Level 2", "desc": "The textbook technique is named"}]
FRAM_INFO = [{"id": "neutral", "name": "Neutral", "desc": "Course-exercise framing"},
             {"id": "authorized", "name": "Authorized", "desc": "User owns the system and is authorized"},
             {"id": "adversarial", "name": "Adversarial", "desc": "User states unauthorized intent"}]

_LOCK = threading.Lock()
_STATE = {"running": False, "progress": {"done": 0, "total": 0}, "error": None, "out": None, "cancel": None,
          "cancelled": False, "args": None}


def _num(v, lo, hi, default, cast=float):
    try:
        v = cast(v)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v))


def _model_spec(c):
    p, name = c.get("provider"), (c.get("model_name") or "").strip()
    base = (c.get("base_url") or "").strip()
    if p == "mock":
        if name not in ("", "oracle", "refuse", "erode", "wrong"):
            raise ValueError("Mock behaviour must be oracle, refuse, erode or wrong.")
        return f"mock:{name or 'erode'}", None
    if p not in ("gemini", "openai", "anthropic", "compat"):
        raise ValueError("Unknown provider.")
    if not name:
        raise ValueError("Enter a model name.")
    if p == "compat":
        if not base.startswith(("http://", "https://")):
            raise ValueError("Enter the base URL (starting with http:// or https://).")
        return f"compat:{name}@{base}", base
    if not c.get("api_key") and not os.environ.get(models.ENV_KEYS[p]):
        raise ValueError(f"Paste an API key or set {models.ENV_KEYS[p]} before starting the UI.")
    return f"{p}:{name}", None


def _args(c, spec, base, out):
    pick = lambda key, allowed: [x for x in (c.get(key) or []) if x in allowed]
    ts = pick("tasks", tasks.TASKS)
    ds = pick("difficulties", tasks.DIFFICULTIES)
    hs = sorted({int(h) for h in (c.get("hints") or []) if int(h) in (0, 1, 2)})
    fs = pick("framings", ("neutral", "authorized", "adversarial"))
    if not (ts and ds and hs and fs):
        raise ValueError("Select at least one task, difficulty, hint level and framing.")
    mode = c.get("mode") if c.get("mode") in ("answer", "code") else "answer"
    return SimpleNamespace(
        models=[spec], tasks=ts, difficulties=ds, hints=hs, framings=fs, mode=mode,
        instances=_num(c.get("instances"), 1, 50, 3, int), samples=_num(c.get("samples"), 1, 10, 1, int),
        temperature=_num(c.get("temperature"), 0, 2, 0.7), max_tokens=_num(c.get("max_tokens"), 256, 65536, 4096, int),
        seed=1337, workers=_num(c.get("workers"), 1, 16, 4, int), sleep=_num(c.get("delay"), 0, 120, 0.0),
        exec_timeout=20.0, out=str(out), api_key=c.get("api_key") or None, base_url=base)


def _worker(a, cancel):
    def cb(done, total):
        with _LOCK:
            _STATE["progress"] = {"done": done, "total": total}
    err = None
    try:
        runner.run(a, progress_cb=cb, cancel=cancel)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    with _LOCK:
        _STATE.update(running=False, error=err, cancelled=cancel.is_set())


def start_run(c):
    spec, base = _model_spec(c)
    out = Path("results") / time.strftime("ui_%Y%m%d_%H%M%S")
    a = _args(c, spec, base, out)
    with _LOCK:
        if _STATE["running"]:
            raise ValueError("A run is already in progress.")
        cancel = threading.Event()
        _STATE.update(running=True, progress={"done": 0, "total": 0}, error=None, out=out, cancel=cancel,
                      cancelled=False, args=a)
    threading.Thread(target=_worker, args=(a, cancel), daemon=True).start()


def _read_rows():
    out = _STATE.get("out")
    raw = Path(out) / "raw.jsonl" if out else None
    if not raw or not raw.exists():
        return [], []
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["outcome"] != "ERROR"], [r for r in rows if r["outcome"] == "ERROR"]


def results_payload():
    rows, errs = _read_rows()
    slim = [{k: r.get(k) for k in ("model", "task", "difficulty", "instance_id", "framing", "hint", "trial", "outcome",
                                    "parsed_answer", "latency_s")} | {"response": (r.get("response") or "")[:4000]}
            for r in rows]
    esc, will = [], []
    framings = sorted({r["framing"] for r in rows})
    hints = sorted({r["hint"] for r in rows})

    def pair(A, B):
        da, dr = analyze._paired(A, B, "correct"), analyze._paired(A, B, "refused")
        if not (da and dr):
            return None
        return {"n_inst": da["n_inst"], "dASR": da["delta"], "asr_ci": [da["ci_lo"], da["ci_hi"]], "p_asr": da["p"],
                "dRef": dr["delta"], "ref_ci": [dr["ci_lo"], dr["ci_hi"]], "p_ref": dr["p"]}

    if len(hints) > 1:
        for f in framings:
            x = pair([r for r in rows if r["framing"] == f and r["hint"] == hints[0]],
                     [r for r in rows if r["framing"] == f and r["hint"] == hints[-1]])
            if x:
                esc.append({"framing": f, "hlo": hints[0], "hhi": hints[-1], **x})
    if "neutral" in framings and "adversarial" in framings:
        for h in hints:
            x = pair([r for r in rows if r["framing"] == "neutral" and r["hint"] == h],
                     [r for r in rows if r["framing"] == "adversarial" and r["hint"] == h])
            if x:
                will.append({"hint": h, **x})
    return {"rows": slim, "escalation": esc, "willingness": will, "errors": len(errs),
            "first_error": (errs[0].get("error") if errs else None), "instances": len({r["instance_id"] for r in rows})}


def chat(c):
    spec, base = _model_spec(c)
    msgs = [m for m in (c.get("messages") or []) if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    if not msgs or msgs[-1]["role"] != "user" or len(msgs) > 60 or sum(len(m["content"]) for m in msgs) > 80000:
        raise ValueError("Invalid chat history.")
    client = models.make_client(spec, c.get("api_key") or None, base)
    resp = client.generate(msgs, _num(c.get("temperature"), 0, 2, 0.7), _num(c.get("max_tokens"), 256, 65536, 4096, int))
    meta = resp.get("meta", {})
    text = resp.get("text") or ""
    if not text and meta.get("blocked"):
        text = "(The provider blocked this response: %s)" % (meta.get("block_reason") or meta.get("finish_reason") or "safety filter")
    return text


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CryptoHarm · Benchmark</title>
<style>
:root {
  color-scheme: dark;
  --bg: #070b14; --surface: #0f1726; --surface2: #15203a; --line: #22324d;
  --text: #eaf1ff; --muted: #8fa1bf; --accent: #6ee7c2; --accent2: #22b58f;
  --c0: #ff7b86; --c1: #f6cb75; --c2: #7aa2ff; --c3: #6ee7c2;
  --radius: 16px;
}
* { box-sizing: border-box; }
body {
  margin: 0; min-height: 100vh; padding: 0 22px 64px; color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.55;
  background:
    radial-gradient(ellipse 60rem 28rem at 8% -4%, rgba(34,181,143,.16), transparent),
    radial-gradient(ellipse 50rem 26rem at 100% 0%, rgba(122,162,255,.14), transparent),
    var(--bg);
  background-attachment: fixed;
}
main { width: min(1120px, 100%); margin: 0 auto; }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 22px 0; }
.logo { display: flex; align-items: center; gap: 12px; font-weight: 800; letter-spacing: -.02em; font-size: 1.05rem; }
.logo-mark {
  display: grid; place-items: center; width: 40px; height: 40px; border-radius: 12px; color: #04130e; font-size: .9rem;
  background: linear-gradient(135deg, #8bf3d4, #22b58f); box-shadow: 0 6px 20px rgba(34,181,143,.35);
}
.chip {
  display: inline-flex; align-items: center; gap: 8px; padding: 7px 13px; border: 1px solid var(--line);
  border-radius: 999px; color: var(--muted); background: rgba(15,23,38,.8); font-size: .78rem; font-weight: 650;
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: #5b6b86; }
.chip.running .dot { background: var(--c1); animation: pulse 1.1s infinite; }
.chip.done .dot { background: var(--accent); }
.chip.error .dot { background: var(--c0); }
@keyframes pulse { 50% { opacity: .3; transform: scale(.7); } }
.hero { padding: 26px 0 30px; }
.eyebrow { margin: 0 0 10px; color: var(--accent); font-size: .74rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
h1 { margin: 0; max-width: 780px; font-size: clamp(2rem, 5vw, 3.2rem); line-height: 1.08; letter-spacing: -.045em; }
h1 span { background: linear-gradient(90deg, #8bf3d4, #7aa2ff); -webkit-background-clip: text; background-clip: text; color: transparent; }
.sub { max-width: 680px; margin: 14px 0 0; color: var(--muted); font-size: 1rem; }
.panel {
  margin: 0 0 16px; padding: 24px; border: 1px solid var(--line); border-radius: var(--radius);
  background: linear-gradient(160deg, rgba(21,32,58,.92), rgba(12,19,33,.96)); box-shadow: 0 18px 44px rgba(0,0,0,.2);
}
.panel-head { display: flex; align-items: center; gap: 13px; margin: 0 0 20px; }
.step { display: grid; place-items: center; width: 32px; height: 32px; flex: 0 0 32px; border-radius: 10px; color: var(--accent); background: rgba(110,231,194,.12); font-size: .8rem; font-weight: 800; }
h2 { margin: 0; font-size: 1.1rem; letter-spacing: -.015em; }
.note { margin: 2px 0 0; color: var(--muted); font-size: .83rem; }
.grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.field { display: flex; flex-direction: column; gap: 7px; min-width: 0; }
.field.span2 { grid-column: span 2; }
label { color: #d6e1f5; font-size: .84rem; font-weight: 650; }
.hint { color: var(--muted); font-size: .76rem; font-weight: 450; }
input[type=text], input[type=password], input[type=number], select {
  width: 100%; height: 44px; padding: 0 13px; border: 1px solid #2f4262; border-radius: 11px; outline: none;
  color: var(--text); background: #0a1220; font: inherit; font-size: .9rem; transition: border-color .15s, box-shadow .15s;
}
input::placeholder { color: #61748f; }
input:focus, select:focus { border-color: var(--accent2); box-shadow: 0 0 0 3px rgba(34,181,143,.16); }
.inputrow { display: flex; gap: 8px; }
button {
  min-height: 44px; padding: 0 18px; border: 1px solid #364b6b; border-radius: 11px; color: var(--text);
  background: #182742; font: inherit; font-size: .88rem; font-weight: 700; cursor: pointer; transition: transform .15s, filter .15s;
}
button:hover:not(:disabled) { filter: brightness(1.15); transform: translateY(-1px); }
button:focus-visible { outline: 3px solid rgba(110,231,194,.4); outline-offset: 2px; }
button:disabled { opacity: .55; cursor: not-allowed; }
button.ghost { min-height: 44px; padding: 0 13px; font-size: .8rem; }
button.primary { min-width: 200px; border: 0; color: #04130e; background: linear-gradient(135deg, #8bf3d4, #22b58f); box-shadow: 0 10px 26px rgba(34,181,143,.28); font-size: .95rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.card {
  position: relative; display: flex; gap: 12px; padding: 15px; border: 1px solid var(--line); border-radius: 13px;
  background: rgba(8,14,26,.55); cursor: pointer; transition: border-color .15s, background .15s, transform .15s;
}
.card:hover { border-color: #46648a; transform: translateY(-1px); }
.card input { width: 17px; height: 17px; margin-top: 3px; flex: 0 0 17px; accent-color: var(--accent2); }
.card:has(input:checked) { border-color: rgba(110,231,194,.55); background: rgba(34,181,143,.08); }
.card strong { display: block; font-size: .9rem; }
.card { font-weight: 400; }
.card small { display: block; margin-top: 2px; color: var(--muted); font-size: .76rem; font-weight: 450; line-height: 1.35; }
.card .tag { display: inline-block; margin-top: 6px; padding: 1px 8px; border-radius: 999px; color: var(--c1); background: rgba(246,203,117,.12); font-size: .68rem; font-weight: 750; }
.banner { margin: 0 0 16px; padding: 13px 16px; border: 1px solid; border-radius: 12px; font-size: .86rem; }
.banner.warn { color: #f8dda1; border-color: rgba(246,203,117,.32); background: rgba(246,203,117,.08); }
.banner.err { color: #ffb8be; border-color: rgba(255,123,134,.4); background: rgba(255,123,134,.09); }
.hidden { display: none !important; }
.runbar { display: flex; align-items: center; flex-wrap: wrap; gap: 18px; margin: 4px 0 8px; }
.est { color: var(--muted); font-size: .84rem; }
.progress { height: 8px; margin: 14px 0 0; overflow: hidden; border-radius: 99px; background: #14203a; }
.progress i { display: block; width: 0; height: 100%; border-radius: 99px; background: linear-gradient(90deg, #22b58f, #8bf3d4); transition: width .4s; }
.spinner { display: inline-block; width: 14px; height: 14px; margin-right: 8px; vertical-align: -2px; border: 2px solid rgba(4,19,14,.3); border-top-color: #04130e; border-radius: 50%; animation: spin .7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
#resultsView { margin-top: 34px; }
.res-head { display: flex; align-items: flex-end; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin: 0 0 18px; }
.res-head h2 { font-size: 1.6rem; letter-spacing: -.03em; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 0 0 16px; }
.stat { padding: 18px; border: 1px solid var(--line); border-radius: 14px; background: linear-gradient(160deg, rgba(21,32,58,.9), rgba(12,19,33,.95)); }
.stat .k { color: var(--muted); font-size: .76rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
.stat .v { margin-top: 6px; font-size: 2rem; font-weight: 800; letter-spacing: -.04em; line-height: 1.1; }
.stat .s { margin-top: 4px; color: var(--muted); font-size: .78rem; }
.block { margin: 0 0 16px; padding: 22px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); }
.block h3 { margin: 0 0 4px; font-size: 1.02rem; }
.block .note { margin: 0 0 16px; }
.scroll { width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }
table { width: 100%; border-collapse: collapse; font-size: .84rem; }
th, td { padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }
th { color: #aebfd8; background: #15223a; font-size: .72rem; font-weight: 750; letter-spacing: .05em; text-transform: uppercase; white-space: nowrap; }
td { color: #dbe5f6; }
tr:last-child td { border-bottom: 0; }
tbody tr:hover { background: rgba(110,231,194,.04); }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.cell { min-width: 96px; }
.cell .pct { font-weight: 700; font-variant-numeric: tabular-nums; }
.meter { height: 6px; margin-top: 6px; overflow: hidden; border-radius: 99px; background: #14203a; }
.meter i { display: block; height: 100%; border-radius: 99px; }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: .74rem; font-weight: 750; white-space: nowrap; }
.badge::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.b0 { color: var(--c0); background: rgba(255,123,134,.12); } .b1 { color: var(--c1); background: rgba(246,203,117,.12); }
.b2 { color: var(--c2); background: rgba(122,162,255,.13); } .b3 { color: var(--c3); background: rgba(110,231,194,.12); }
.berr { color: #aab6c8; background: rgba(170,182,200,.12); }
.chartbox svg { display: block; width: 100%; height: auto; }
.chartbox .ax { fill: var(--muted); font-size: 11px; }
.chartbox .lv { fill: #d6e1f5; font-size: 12px; font-weight: 650; }
.chartbox .seg { fill: #07111c; font-size: 12px; font-weight: 800; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 14px 0 0; color: var(--muted); font-size: .8rem; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 7px; border-radius: 3px; background: var(--c); }
.filters { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 14px; }
.filters select { width: auto; min-width: 150px; height: 38px; font-size: .82rem; }
details summary { color: var(--accent); font-weight: 650; cursor: pointer; }
details pre { max-width: 540px; max-height: 260px; margin: 8px 0 0; padding: 12px; overflow: auto; border: 1px solid var(--line); border-radius: 10px; background: #070f1c; color: #dbe5f6; font-size: .76rem; white-space: pre-wrap; overflow-wrap: anywhere; }
.err-text { color: #ffb8be; font-size: .8rem; }
.footer { margin: 34px 0 0; color: #5f7290; text-align: center; font-size: .76rem; }
.tabs { display: inline-flex; gap: 4px; margin: 0 0 18px; padding: 4px; border: 1px solid var(--line); border-radius: 13px; background: rgba(15,23,38,.8); }
.tab { min-height: 38px; padding: 0 20px; border: 0; border-radius: 9px; color: var(--muted); background: transparent; font-size: .88rem; }
.tab.active { color: #04130e; background: linear-gradient(135deg, #8bf3d4, #22b58f); }
body.chat .bench { display: none !important; }
body:not(.chat) .chatonly { display: none !important; }
.chatlog { display: flex; flex-direction: column; gap: 12px; height: 430px; padding: 16px; overflow-y: auto; border: 1px solid var(--line); border-radius: 13px; background: #0a1220; }
.chat-empty { margin: auto; color: var(--muted); font-size: .9rem; text-align: center; }
.msg { max-width: 82%; padding: 11px 15px; border-radius: 14px; font-size: .9rem; overflow-wrap: anywhere; }
.msg.user { align-self: flex-end; color: #04130e; background: linear-gradient(135deg, #8bf3d4, #3fcba5); border-bottom-right-radius: 4px; }
.msg.assistant { align-self: flex-start; background: #15223a; border: 1px solid var(--line); border-bottom-left-radius: 4px; }
.msg.err { align-self: center; max-width: 100%; color: #ffb8be; background: rgba(255,123,134,.09); border: 1px solid rgba(255,123,134,.4); }
.msg .who { display: block; margin: 0 0 3px; font-size: .68rem; font-weight: 800; letter-spacing: .08em; opacity: .65; text-transform: uppercase; }
.msg .t { white-space: pre-wrap; }
.msg pre { margin: 8px 0 2px; padding: 11px; overflow-x: auto; border: 1px solid var(--line); border-radius: 9px; background: #070f1c; color: #dbe5f6; font-size: .78rem; white-space: pre; }
.chatbox { display: flex; gap: 12px; align-items: flex-end; margin: 14px 0 0; }
.chatbox textarea { flex: 1; min-height: 84px; padding: 12px 14px; resize: vertical; border: 1px solid #2f4262; border-radius: 11px; outline: none; color: var(--text); background: #0a1220; font: inherit; font-size: .92rem; }
.chatbox textarea:focus { border-color: var(--accent2); box-shadow: 0 0 0 3px rgba(34,181,143,.16); }
.chatbtns { display: flex; flex-direction: column; gap: 8px; }
.chatbtns button.primary { min-width: 120px; }
@media (max-width: 560px) { .chatbox { flex-direction: column; align-items: stretch; } .chatbtns { flex-direction: row; } .msg { max-width: 94%; } }
@media (max-width: 800px) { .grid { grid-template-columns: 1fr 1fr; } .field.span2 { grid-column: 1 / -1; } }
@media (max-width: 560px) { body { padding: 0 14px 44px; } .grid { grid-template-columns: 1fr; } .panel { padding: 18px; } button.primary { width: 100%; } .runbar { align-items: stretch; flex-direction: column; } }
.chartgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }
.inputrow { display: flex; gap: 8px; }
</style>
</head>
<body>
<main>
  <div class="topbar">
    <div class="logo"><div class="logo-mark">CH</div>CryptoHarm <span class="hint" style="margin-left:6px">v2</span></div>
    <div class="chip" id="statusChip"><span class="dot"></span><span id="statusText">Idle</span></div>
  </div>

  <header class="hero">
    <p class="eyebrow">Harm-escalation benchmark</p>
    <h1>Do hints and stated intent change what a model <span>will</span> and <span>can</span> do?</h1>
    <p class="sub">Run weak-cryptography tasks across hint levels, difficulty and user framing. Every answer is checked against locally generated ground truth, so a success only counts if the secret is really recovered.</p>
  </header>

  <div class="tabs" role="tablist"><button class="tab active" id="tabBench" type="button">Benchmark</button><button class="tab" id="tabChat" type="button">Chat</button></div>

  <section class="panel">
    <div class="panel-head"><span class="step">01</span><div><h2>Model</h2><p class="note">Pick a provider. Keys are sent only with the request and never written to disk.</p></div></div>
    <div class="grid">
      <div class="field"><label for="provider">Provider</label>
        <select id="provider">
          <option value="mock" selected>Mock (offline test, no key)</option>
          <option value="gemini">Gemini</option>
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
          <option value="compat">OpenAI-compatible (Groq, Ollama, vLLM...)</option>
        </select></div>
      <div class="field"><label for="modelName" id="modelLabel">Model name</label>
        <input type="text" id="modelName" placeholder="erode"></div>
      <div class="field"><label for="apiKey">API key</label>
        <div class="inputrow"><input type="password" id="apiKey" autocomplete="off" placeholder="not needed for mock"><button class="ghost" type="button" id="toggleKey">Show</button></div></div>
      <div class="field span2 hidden" id="baseUrlField"><label for="baseUrl">Base URL <span class="hint">(OpenAI-compatible only)</span></label>
        <input type="text" id="baseUrl" placeholder="https://api.groq.com/openai/v1   or   http://localhost:11434/v1"></div>
      <div class="field"><label for="temperature">Temperature</label>
        <input type="number" id="temperature" min="0" max="2" step="0.1" value="0.7"></div>
      <div class="field"><label for="maxTokens">Max output tokens <span class="hint">reasoning models need room</span></label>
        <input type="number" id="maxTokens" min="256" max="65536" step="256" value="4096"></div>
    </div>
  </section>

  <section class="panel bench">
    <div class="panel-head"><span class="step">02</span><div><h2>Tasks</h2><p class="note">Each task is a deliberately weak, self-generated cryptographic challenge.</p></div></div>
    <div class="cards" id="taskBoxes"></div>
  </section>

  <section class="panel bench">
    <div class="panel-head"><span class="step">03</span><div><h2>Design factors</h2><p class="note">Results are crossed over every selected difficulty, hint level and framing. The same instance is used in every cell.</p></div></div>
    <p class="hint" style="margin:0 0 8px">Difficulty</p><div class="cards" id="diffBoxes"></div>
    <p class="hint" style="margin:18px 0 8px">Hint level</p><div class="cards" id="hintBoxes"></div>
    <p class="hint" style="margin:18px 0 8px">Framing <span class="hint">(neutral vs adversarial separates willingness from capability)</span></p><div class="cards" id="framBoxes"></div>
  </section>

  <section class="panel bench">
    <div class="panel-head"><span class="step">04</span><div><h2>Run settings</h2><p class="note">More instances and samples give tighter confidence intervals.</p></div></div>
    <div class="grid">
      <div class="field"><label for="instances">Instances <span class="hint">per task and difficulty</span></label>
        <input type="number" id="instances" min="1" max="50" value="3"></div>
      <div class="field"><label for="samples">Samples <span class="hint">per prompt</span></label>
        <input type="number" id="samples" min="1" max="10" value="1"></div>
      <div class="field"><label for="mode">Answer mode</label>
        <select id="mode"><option value="answer" selected>Answer (model replies with the value)</option><option value="code">Code (model writes a program that is run)</option></select></div>
      <div class="field"><label for="delay">Delay after each call <span class="hint">seconds (rate limits)</span></label>
        <input type="number" id="delay" min="0" max="120" step="1" value="0"></div>
      <div class="field"><label for="workers">Parallel calls</label>
        <input type="number" id="workers" min="1" max="16" value="4"></div>
    </div>
  </section>

  <div id="liveBanner" class="banner warn hidden bench"><strong>Code mode:</strong> code written by the model is executed locally with AST screening and a timeout. That is a mitigation, not a sandbox. Use a disposable VM or container for untrusted models.</div>
  <div id="errBanner" class="banner err hidden" role="alert"></div>

  <div class="runbar bench">
    <button id="runBtn" class="primary" type="button">Run benchmark →</button>
    <button id="stopBtn" class="hidden" type="button">Stop</button>
    <span class="est" id="estimate"></span>
  </div>
  <div class="progress hidden bench" id="progWrap"><i id="progBar"></i></div>

  <section id="resultsView" class="hidden bench" aria-live="polite">
    <div class="res-head"><div><p class="eyebrow">Results</p><h2 id="resTitle">Benchmark results</h2></div>
      <div style="display:flex;gap:10px;flex-wrap:wrap"><button type="button" id="reportBtn">Write full report</button><button type="button" id="csvBtn">Download CSV ↓</button><button type="button" onclick="location.href='/api/results.json'">Download JSON ↓</button></div></div>
    <div class="banner warn hidden" id="reportMsg"></div>
    <div class="stats" id="stats"></div>
    <div class="block"><h3>Outcome by hint level</h3><p class="note">Share of runs by outcome for each framing. Verified solves include a 95% Wilson interval (pooled runs; the full report uses instance-level bootstrap).</p><div id="summaryTable"></div></div>
    <div class="block"><h3>Escalation chart</h3><p class="note">Each bar is one hint level, split by outcome.</p><div class="chartgrid" id="chartwrap"></div><div class="legend" id="legend"></div></div>
    <div class="block"><h3>Hint escalation and willingness gap</h3><p class="note">Paired over instances: hint-level differences within a framing, and adversarial minus neutral at each hint level. Positive dRefusal means more refusals.</p><div id="escTable"></div></div>
    <div class="block"><h3>By task</h3><p class="note">Verified solve rate per task, framing and hint level.</p><div id="taskTable"></div></div>
    <div class="block"><h3>Individual runs</h3><div class="filters">
      <select id="fTask"><option value="">All tasks</option></select>
      <select id="fFram"><option value="">All framings</option></select>
      <select id="fHint"><option value="">All hint levels</option></select>
      <select id="fOut"><option value="">All outcomes</option><option>SOLVED</option><option>WRONG</option><option>NO_ANSWER</option><option>REFUSED</option></select></div>
      <div id="itemTable"></div></div>
  </section>

  <section class="panel chatonly" id="chatView">
    <div class="panel-head"><span class="step">02</span><div><h2>Chat</h2><p class="note">Talk to the model chosen above. Messages go straight to the provider. Nothing here is scored, and no code is executed.</p></div></div>
    <div class="chatlog" id="chatLog"></div>
    <div class="chatbox">
      <textarea id="chatInput" placeholder="Type a message... (Ctrl+Enter to send)"></textarea>
      <div class="chatbtns"><button class="primary" id="chatSend" type="button">Send →</button><button class="ghost" id="chatClear" type="button">Clear chat</button></div>
    </div>
  </section>
  <footer class="footer">CryptoHarm v2 · instances are synthetic and weak by construction · scoring runs locally</footer>
</main>

<script>
const $ = id => document.getElementById(id);
const OUTS = [["SOLVED", "Verified solve", "var(--c3)", "b3"], ["WRONG", "Wrong answer", "var(--c2)", "b2"],
              ["NO_ANSWER", "No answer", "var(--c1)", "b1"], ["REFUSED", "Refusal", "var(--c0)", "b0"]];
const PLACEHOLDER = {mock: "oracle | refuse | erode | wrong  (default: erode)", gemini: "e.g. gemini-2.5-flash",
                     openai: "model name", anthropic: "model name", compat: "model identifier"};
let CFG = null, ROWS = [], ESC = null, pollTimer = null, history = [];

function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c])); }
function wilson(k, n) {
  if (!n) return [0, 1];
  const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d;
  const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}
const pc = x => Math.round(x * 100) + "%";
const f3 = x => (x == null || x !== x) ? "n/a" : x.toFixed(3);
const badge = o => { const d = OUTS.find(x => x[0] === o); return '<span class="badge ' + (d ? d[3] : "berr") + '">' + esc(d ? d[1] : o) + '</span>'; };

function cards(id, cls, items, checked) {
  items.forEach(t => $(id).insertAdjacentHTML("beforeend",
    '<label class="card"><input type="checkbox" class="' + cls + '" value="' + esc(t.id) + '"' + (checked(t) ? " checked" : "") + '>' +
    '<span><strong>' + esc(t.name || t.id) + '</strong><small>' + esc(t.desc) + '</small></span></label>'));
}
const picked = cls => [...document.querySelectorAll("." + cls + ":checked")].map(e => e.value);

function cfg() {
  return {
    provider: $("provider").value, model_name: $("modelName").value.trim(), api_key: $("apiKey").value,
    base_url: $("baseUrl").value.trim(), temperature: parseFloat($("temperature").value),
    max_tokens: parseInt($("maxTokens").value, 10) || 4096,
    tasks: picked("taskChk"), difficulties: picked("diffChk"),
    hints: picked("hintChk").map(Number), framings: picked("framChk"),
    instances: parseInt($("instances").value, 10) || 1, samples: parseInt($("samples").value, 10) || 1,
    mode: $("mode").value, delay: parseFloat($("delay").value) || 0, workers: parseInt($("workers").value, 10) || 4,
  };
}
function estimate() {
  const c = cfg();
  const n = c.tasks.length * c.difficulties.length * c.instances * c.hints.length * c.framings.length * c.samples;
  $("estimate").textContent = n ? n + " model calls" + (c.delay ? " · at least " + Math.ceil(n * c.delay / c.workers / 60) + " min of delay" : "") : "Select at least one option in each group.";
  return n;
}
function refreshForm() {
  const p = $("provider").value;
  $("baseUrlField").classList.toggle("hidden", p !== "compat");
  $("modelName").placeholder = PLACEHOLDER[p];
  $("modelLabel").textContent = p === "mock" ? "Mock behaviour" : "Model name";
  $("liveBanner").classList.toggle("hidden", $("mode").value !== "code" || p === "mock");
  estimate();
}
function showErr(msg) { $("errBanner").textContent = msg || ""; $("errBanner").classList.toggle("hidden", !msg); }
function setStatus(kind, text) { $("statusChip").className = "chip " + kind; $("statusText").textContent = text; }

async function init() {
  CFG = await (await fetch("/api/config")).json();
  cards("taskBoxes", "taskChk", CFG.tasks, () => true);
  cards("diffBoxes", "diffChk", CFG.difficulties, t => t.id !== "hard");
  cards("hintBoxes", "hintChk", CFG.hints, () => true);
  cards("framBoxes", "framChk", CFG.framings, t => t.id !== "authorized");
  CFG.tasks.forEach(t => $("fTask").insertAdjacentHTML("beforeend", '<option value="' + esc(t.id) + '">' + esc(t.name) + '</option>'));
  document.querySelectorAll("input,select").forEach(e => e.addEventListener("change", refreshForm));
  document.querySelectorAll("input[type=number]").forEach(e => e.addEventListener("input", estimate));
  refreshForm();
  const s = await (await fetch("/api/status")).json();
  if (s.running) { setStatus("running", "Running"); $("runBtn").disabled = true; poll(); }
  else if (s.has_results) { await loadResults(); setStatus("done", "Done"); }
}

$("toggleKey").onclick = () => { const k = $("apiKey"); k.type = k.type === "password" ? "text" : "password"; $("toggleKey").textContent = k.type === "password" ? "Show" : "Hide"; };
$("tabBench").onclick = () => { document.body.classList.remove("chat"); $("tabBench").classList.add("active"); $("tabChat").classList.remove("active"); };
$("tabChat").onclick = () => { document.body.classList.add("chat"); $("tabChat").classList.add("active"); $("tabBench").classList.remove("active"); };

$("runBtn").onclick = async () => {
  showErr("");
  if (!estimate()) { showErr("Select at least one task, difficulty, hint level and framing."); return; }
  $("runBtn").disabled = true;
  const r = await fetch("/api/run", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(cfg())});
  const j = await r.json();
  if (!r.ok) { showErr(j.error || "Could not start the run."); $("runBtn").disabled = false; return; }
  $("resultsView").classList.add("hidden");
  $("progWrap").classList.remove("hidden"); $("progBar").style.width = "0%"; $("stopBtn").classList.remove("hidden");
  setStatus("running", "Running"); poll();
};
$("stopBtn").onclick = async () => { await fetch("/api/stop", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"}); $("stopBtn").disabled = true; };

function poll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    const s = await (await fetch("/api/status")).json();
    const {done, total} = s.progress;
    $("progWrap").classList.remove("hidden");
    $("progBar").style.width = (total ? 100 * done / total : 0) + "%";
    $("statusText").textContent = "Running " + done + "/" + total;
    if (s.running) { poll(); return; }
    $("runBtn").disabled = false; $("stopBtn").classList.add("hidden"); $("stopBtn").disabled = false;
    if (s.error) { showErr(s.error); setStatus("error", "Error"); }
    else { setStatus("done", s.cancelled ? "Stopped" : "Done"); }
    if (s.has_results) await loadResults();
  }, 900);
}

async function loadResults() {
  const r = await fetch("/api/results");
  if (!r.ok) return;
  const j = await r.json();
  ROWS = j.rows; ESC = j;
  if (j.errors && !j.rows.length) showErr("Every call failed: " + (j.first_error || "unknown error"));
  else if (j.errors) showErr(j.errors + " call(s) failed and are excluded (first: " + (j.first_error || "") + ")");
  render();
}

function groupBy(rows, keyFn) { const m = new Map(); rows.forEach(r => { const k = keyFn(r); if (!m.has(k)) m.set(k, []); m.get(k).push(r); }); return m; }
function counts(rows) { const c = {SOLVED: 0, WRONG: 0, NO_ANSWER: 0, REFUSED: 0}; rows.forEach(r => { if (c[r.outcome] != null) c[r.outcome]++; }); c.n = rows.length; return c; }
function meterCell(k, n, color, ci) {
  const p = n ? k / n : 0, w = ci ? wilson(k, n) : null;
  return '<td class="cell"><div class="pct">' + pc(p) + (w ? ' <span class="hint">[' + pc(w[0]) + '–' + pc(w[1]) + ']</span>' : "") + '</div><div class="meter"><i style="width:' + (p * 100) + '%;background:' + color + '"></i></div></td>';
}

function render() {
  $("resultsView").classList.remove("hidden");
  if (!ROWS.length) { $("stats").innerHTML = ""; return; }
  const all = counts(ROWS);
  const adv = (ESC.escalation || []).find(e => e.framing === "adversarial") || (ESC.escalation || [])[0];
  let stats = '<div class="stat"><div class="k">Runs scored</div><div class="v">' + all.n + '</div><div class="s">' + ESC.instances + ' instances</div></div>' +
    '<div class="stat"><div class="k">Verified solve rate</div><div class="v">' + pc(all.SOLVED / all.n) + '</div><div class="s">across all cells</div></div>' +
    '<div class="stat"><div class="k">Refusal rate</div><div class="v">' + pc(all.REFUSED / all.n) + '</div><div class="s">heuristic; audit by hand</div></div>';
  if (adv) stats += '<div class="stat"><div class="k">Hint effect (' + esc(adv.framing) + ')</div><div class="v">' + (adv.dASR >= 0 ? "+" : "") + pc(adv.dASR) + '</div><div class="s">solve rate, hint ' + adv.hlo + ' → ' + adv.hhi + ' (p=' + f3(adv.p_asr) + ')</div></div>';
  $("stats").innerHTML = stats;
  $("legend").innerHTML = OUTS.map(o => '<span style="--c:' + o[2] + '">' + o[1] + '</span>').join("");

  const framings = [...new Set(ROWS.map(r => r.framing))].sort(), hints = [...new Set(ROWS.map(r => r.hint))].sort();
  // summary table
  let t = '<div class="scroll"><table><thead><tr><th>Framing</th><th>Hint</th><th class="num">Runs</th><th>Verified solve</th><th>Wrong</th><th>No answer</th><th>Refused</th></tr></thead><tbody>';
  framings.forEach(f => hints.forEach(h => {
    const c = counts(ROWS.filter(r => r.framing === f && r.hint === h)); if (!c.n) return;
    t += '<tr><td>' + esc(f) + '</td><td>' + h + '</td><td class="num">' + c.n + '</td>' + meterCell(c.SOLVED, c.n, "var(--c3)", true) + meterCell(c.WRONG, c.n, "var(--c2)") + meterCell(c.NO_ANSWER, c.n, "var(--c1)") + meterCell(c.REFUSED, c.n, "var(--c0)") + '</tr>';
  }));
  $("summaryTable").innerHTML = t + "</tbody></table></div>";
  // charts
  $("chartwrap").innerHTML = framings.map(f => chart(f, hints)).join("");
  // escalation table
  let e = '<div class="scroll"><table><thead><tr><th>Comparison</th><th class="num">Instances</th><th class="num">dSolve</th><th>95% CI</th><th class="num">p</th><th class="num">dRefusal</th><th>95% CI</th><th class="num">p</th></tr></thead><tbody>';
  (ESC.escalation || []).forEach(x => e += '<tr><td>' + esc(x.framing) + ': hint ' + x.hlo + ' → ' + x.hhi + '</td><td class="num">' + x.n_inst + '</td><td class="num">' + f3(x.dASR) + '</td><td>[' + f3(x.asr_ci[0]) + ', ' + f3(x.asr_ci[1]) + ']</td><td class="num">' + f3(x.p_asr) + '</td><td class="num">' + f3(x.dRef) + '</td><td>[' + f3(x.ref_ci[0]) + ', ' + f3(x.ref_ci[1]) + ']</td><td class="num">' + f3(x.p_ref) + '</td></tr>');
  (ESC.willingness || []).forEach(x => e += '<tr><td>adversarial − neutral at hint ' + x.hint + '</td><td class="num">' + x.n_inst + '</td><td class="num">' + f3(x.dASR) + '</td><td>[' + f3(x.asr_ci[0]) + ', ' + f3(x.asr_ci[1]) + ']</td><td class="num">' + f3(x.p_asr) + '</td><td class="num">' + f3(x.dRef) + '</td><td>[' + f3(x.ref_ci[0]) + ', ' + f3(x.ref_ci[1]) + ']</td><td class="num">' + f3(x.p_ref) + '</td></tr>');
  $("escTable").innerHTML = (ESC.escalation || []).length || (ESC.willingness || []).length ? e + "</tbody></table></div>" : '<p class="note">Select at least two hint levels (or both neutral and adversarial framing) to see paired comparisons.</p>';
  // by task
  const tasks = [...new Set(ROWS.map(r => r.task))].sort();
  let b = '<div class="scroll"><table><thead><tr><th>Task</th><th>Framing</th>' + hints.map(h => '<th>Hint ' + h + '</th>').join("") + '</tr></thead><tbody>';
  tasks.forEach(tk => framings.forEach(f => {
    b += '<tr><td>' + esc(tk) + '</td><td>' + esc(f) + '</td>';
    hints.forEach(h => { const c = counts(ROWS.filter(r => r.task === tk && r.framing === f && r.hint === h)); b += c.n ? meterCell(c.SOLVED, c.n, "var(--c3)") : "<td>–</td>"; });
    b += '</tr>';
  }));
  $("taskTable").innerHTML = b + "</tbody></table></div>";
  // filters
  if ($("fFram").options.length === 1) {
    framings.forEach(f => $("fFram").insertAdjacentHTML("beforeend", '<option>' + esc(f) + '</option>'));
    hints.forEach(h => $("fHint").insertAdjacentHTML("beforeend", '<option value="' + h + '">Hint ' + h + '</option>'));
  }
  renderItems();
}

function chart(framing, hints) {
  const bw = 54, gap = 34, L = 40, T = 12, H = 190, B = 30, W = L + hints.length * (bw + gap) + 6;
  let s = '<div class="chartbox"><p class="hint" style="margin:0 0 4px"><strong>' + esc(framing) + '</strong></p><svg viewBox="0 0 ' + W + ' ' + (H + B + T) + '" role="img" aria-label="Outcomes by hint level, ' + esc(framing) + '">';
  [0, .25, .5, .75, 1].forEach(v => { const y = T + H - v * H; s += '<line x1="' + L + '" x2="' + (W - 4) + '" y1="' + y + '" y2="' + y + '" stroke="#22324d"/><text class="ax" x="' + (L - 6) + '" y="' + (y + 4) + '" text-anchor="end">' + pc(v) + '</text>'; });
  hints.forEach((h, i) => {
    const c = counts(ROWS.filter(r => r.framing === framing && r.hint === h)); if (!c.n) return;
    let y = T + H; const x = L + gap / 2 + i * (bw + gap);
    OUTS.forEach(o => { const hh = H * c[o[0]] / c.n; if (hh <= 0) return; y -= hh;
      s += '<rect x="' + x + '" y="' + y + '" width="' + bw + '" height="' + hh + '" rx="3" style="fill:' + o[2] + '"><title>' + o[1] + ': ' + c[o[0]] + '/' + c.n + '</title></rect>';
      if (hh > 16) s += '<text class="seg" x="' + (x + bw / 2) + '" y="' + (y + hh / 2 + 4) + '" text-anchor="middle">' + pc(c[o[0]] / c.n) + '</text>'; });
    s += '<text class="lv" x="' + (x + bw / 2) + '" y="' + (T + H + 20) + '" text-anchor="middle">Hint ' + h + '</text>';
  });
  return s + '</svg></div>';
}

function renderItems() {
  const ft = $("fTask").value, ff = $("fFram").value, fh = $("fHint").value, fo = $("fOut").value;
  const rows = ROWS.filter(r => (!ft || r.task === ft) && (!ff || r.framing === ff) && (!fh || String(r.hint) === fh) && (!fo || r.outcome === fo));
  let t = '<p class="note">' + rows.length + ' runs' + (rows.length > 300 ? " (showing the first 300; download the CSV for all)" : "") + '</p><div class="scroll"><table><thead><tr><th>Task</th><th>Difficulty</th><th>Framing</th><th class="num">Hint</th><th class="num">Trial</th><th>Outcome</th><th class="num">Latency</th><th>Response</th></tr></thead><tbody>';
  rows.slice(0, 300).forEach(r => t += '<tr><td>' + esc(r.task) + '</td><td>' + esc(r.difficulty) + '</td><td>' + esc(r.framing) + '</td><td class="num">' + r.hint + '</td><td class="num">' + r.trial + '</td><td>' + badge(r.outcome) + '</td><td class="num">' + r.latency_s + 's</td><td><details><summary>View</summary><pre>' + esc(r.response || "(empty)") + '</pre></details></td></tr>');
  $("itemTable").innerHTML = t + "</tbody></table></div>";
}
["fTask", "fFram", "fHint", "fOut"].forEach(id => $(id).addEventListener("change", renderItems));

$("csvBtn").onclick = () => {
  const cols = ["model", "task", "difficulty", "instance_id", "framing", "hint", "trial", "outcome", "parsed_answer", "latency_s"];
  const q = v => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"';
  const csv = [cols.join(",")].concat(ROWS.map(r => cols.map(c => q(r[c])).join(","))).join("\r\n");
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"})); a.download = "cryptoharm_runs.csv"; a.click();
};
$("reportBtn").onclick = async () => {
  const r = await fetch("/api/report", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
  const j = await r.json(); const m = $("reportMsg"); m.classList.remove("hidden");
  m.textContent = r.ok ? "Full report (tables, figures, audit sample) written to: " + j.path : (j.error || "Could not write the report.");
};

// ---- chat
function chatRender() {
  const log = $("chatLog");
  if (!history.length) { log.innerHTML = '<div class="chat-empty">Say something to the selected model.</div>'; return; }
  log.innerHTML = history.map(m => {
    const parts = esc(m.content).split("```"); let body = "";
    parts.forEach((p, i) => { body += i % 2 ? '<pre>' + p.replace(/^[a-zA-Z0-9]*\n/, "") + '</pre>' : '<span class="t">' + p + '</span>'; });
    return '<div class="msg ' + (m.role === "user" ? "user" : m.error ? "err" : "assistant") + '"><span class="who">' + (m.role === "user" ? "You" : m.error ? "Error" : "Model") + '</span>' + body + '</div>';
  }).join("");
  log.scrollTop = log.scrollHeight;
}
async function chatSend() {
  const text = $("chatInput").value.trim(); if (!text) return;
  $("chatInput").value = ""; history.push({role: "user", content: text}); chatRender(); $("chatSend").disabled = true;
  const c = cfg();
  const msgs = history.filter(m => !m.error).map(m => ({role: m.role, content: m.content}));
  try {
    const r = await fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({provider: c.provider, model_name: c.model_name, api_key: c.api_key, base_url: c.base_url, temperature: c.temperature, max_tokens: c.max_tokens, messages: msgs})});
    const j = await r.json();
    history.push(r.ok ? {role: "assistant", content: j.reply || "(empty reply)"} : {role: "assistant", content: j.error || "Request failed.", error: true});
  } catch (e) { history.push({role: "assistant", content: String(e), error: true}); }
  $("chatSend").disabled = false; chatRender();
}
$("chatSend").onclick = chatSend;
$("chatClear").onclick = () => { history = []; chatRender(); };
$("chatInput").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); chatSend(); } });
chatRender();
init();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "CryptoHarmUI"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _local(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in ("127.0.0.1", "localhost", "::1"):
            return False
        origin = self.headers.get("Origin")
        return not origin or origin.split("://", 1)[-1].rsplit(":", 1)[0].strip("[]") in ("127.0.0.1", "localhost", "::1")

    def do_GET(self):
        if not self._local():
            return self._send(403, {"error": "forbidden"})
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/config":
            return self._send(200, {"tasks": TASK_INFO, "difficulties": DIFF_INFO, "hints": HINT_INFO, "framings": FRAM_INFO})
        if path == "/api/status":
            with _LOCK:
                rows_exist = bool(_STATE["out"] and (Path(_STATE["out"]) / "raw.jsonl").exists())
                return self._send(200, {"running": _STATE["running"], "progress": _STATE["progress"], "error": _STATE["error"],
                                        "cancelled": _STATE["cancelled"], "has_results": rows_exist})
        if path == "/api/results":
            return self._send(200, results_payload())
        if path == "/api/results.json":
            return self._send(200, results_payload(), extra={"Content-Disposition": "attachment; filename=cryptoharm_results.json"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._local():
            return self._send(403, {"error": "forbidden"})
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._send(415, {"error": "JSON only"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 300_000:
                return self._send(413, {"error": "request too large"})
            c = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "bad JSON"})
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/run":
                start_run(c)
                return self._send(200, {"started": True})
            if path == "/api/stop":
                with _LOCK:
                    if _STATE["cancel"]:
                        _STATE["cancel"].set()
                return self._send(200, {"ok": True})
            if path == "/api/chat":
                return self._send(200, {"reply": chat(c)})
            if path == "/api/report":
                with _LOCK:
                    out = _STATE["out"]
                if not out or not (Path(out) / "raw.jsonl").exists():
                    return self._send(400, {"error": "Run a benchmark first."})
                analyze.analyze(Path(out) / "raw.jsonl", Path(out) / "analysis")
                return self._send(200, {"path": str((Path(out) / "analysis").resolve())})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except models.ApiError as e:
            return self._send(502, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        self._send(404, {"error": "not found"})


def serve(host="127.0.0.1", port=8000, open_browser=False):
    srv = None
    for p in range(port, port + 11):
        try:
            srv = ThreadingHTTPServer((host, p), Handler)
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit(f"Ports {port}-{port + 10} are all busy. Close the old window or pass --port.")
    url = f"http://{host}:{srv.server_address[1]}/"
    print(f"CryptoHarm UI running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.server_close()
    return srv
