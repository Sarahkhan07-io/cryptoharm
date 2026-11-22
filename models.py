"""Stdlib-only model adapters. Model spec strings:
  gemini:<model>                 key in GEMINI_API_KEY
  openai:<model>                 key in OPENAI_API_KEY
  anthropic:<model>              key in ANTHROPIC_API_KEY
  compat:<model>@<base_url>      any OpenAI-compatible server (Groq, Ollama, vLLM, OpenRouter...), optional COMPAT_API_KEY
  mock:<oracle|refuse|erode|wrong>   offline pipeline testing (reads the secret; never use for results)
`prompt` may be a plain string or a chat history: [{"role": "user"|"assistant", "content": "..."}].
An explicit api_key argument (from the UI) overrides the environment variable and is never written to disk.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class ApiError(Exception):
    def __init__(self, code, body):
        super().__init__(f"HTTP {code}: {body[:300]}")
        self.code, self.body = code, body


def _post(url, headers, body, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise ApiError(e.code, e.read().decode(errors="replace"))


def _msgs(prompt):
    return prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]


def _need_key(explicit, env_name):
    v = explicit or os.environ.get(env_name)
    if not v:
        raise RuntimeError(f"No API key: paste it in the UI or set {env_name} "
                           f"(PowerShell: $env:{env_name}=\"KEY\"  |  cmd: set {env_name}=KEY)")
    return v


ENV_KEYS = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class Gemini:
    needs_context = False

    def __init__(self, model, api_key=None):
        self.model, self.api_key = model, api_key

    def generate(self, prompt, temperature, max_tokens, context=None):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in _msgs(prompt)]
        body = {"contents": contents, "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        d = _post(url, {"x-goog-api-key": _need_key(self.api_key, "GEMINI_API_KEY")}, body)
        meta = {}
        pf = d.get("promptFeedback", {})
        if pf.get("blockReason"):
            meta.update(blocked=True, block_reason=pf["blockReason"])
        cands = d.get("candidates") or []
        text = ""
        if cands:
            c = cands[0]
            meta["finish_reason"] = c.get("finishReason")
            if c.get("finishReason") in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"):
                meta["blocked"] = True
            text = "".join(p.get("text", "") for p in (c.get("content") or {}).get("parts", []))
        elif not meta.get("blocked"):
            meta["empty"] = True
        return {"text": text, "meta": meta}


class OpenAICompat:
    needs_context = False

    def __init__(self, model, base="https://api.openai.com/v1", key_env="OPENAI_API_KEY", optional_key=False,
                 api_key=None):
        self.model, self.base, self.key_env = model, base.rstrip("/"), key_env
        self.optional_key, self.api_key = optional_key, api_key

    def generate(self, prompt, temperature, max_tokens, context=None):
        key = (self.api_key or os.environ.get(self.key_env)) if self.optional_key else _need_key(self.api_key, self.key_env)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        body = {"model": self.model, "messages": _msgs(prompt), "temperature": temperature, "max_tokens": max_tokens}
        url = f"{self.base}/chat/completions"
        try:
            d = _post(url, headers, body)
        except ApiError as e:  # newer OpenAI models: max_completion_tokens, fixed temperature
            if e.code == 400 and ("max_completion_tokens" in e.body or "temperature" in e.body):
                body.pop("max_tokens", None)
                body["max_completion_tokens"] = max_tokens
                if "temperature" in e.body:
                    body.pop("temperature", None)
                d = _post(url, headers, body)
            else:
                raise
        ch = d["choices"][0]
        msg = ch.get("message", {})
        meta = {"finish_reason": ch.get("finish_reason")}
        if msg.get("refusal"):
            meta.update(blocked=True, finish_reason="refusal")
        return {"text": msg.get("content") or msg.get("refusal") or "", "meta": meta}


class Anthropic:
    needs_context = False

    def __init__(self, model, api_key=None):
        self.model, self.api_key = model, api_key

    def generate(self, prompt, temperature, max_tokens, context=None):
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "messages": _msgs(prompt)}
        d = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": _need_key(self.api_key, "ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"}, body)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        return {"text": text, "meta": {"finish_reason": d.get("stop_reason")}}


class Mock:
    """Offline stand-ins for testing the pipeline end to end. They read the secret via `context`."""
    needs_context = True

    def __init__(self, behavior):
        self.behavior = behavior

    def _solve(self, ctx, wrong=False):
        a = ctx["secret_answer"] if not wrong else ("0" * 8)
        if ctx["mode"] == "code":
            return f"```python\nprint({a!r})\n```"
        return f"Working it out.\nANSWER: {a}"

    def generate(self, prompt, temperature, max_tokens, context=None):
        ctx = context
        if ctx is None:  # chat usage
            return {"text": f"[mock:{self.behavior}] This is an offline test model; it has no real replies.", "meta": {}}
        refuse = {"text": "I'm sorry, but I can't help with that.", "meta": {}}
        if self.behavior == "oracle":
            return {"text": self._solve(ctx), "meta": {}}
        if self.behavior == "refuse":
            return refuse
        if self.behavior == "wrong":
            return {"text": self._solve(ctx, wrong=True), "meta": {}}
        if self.behavior == "erode":  # refuses adversarial requests unless given a hint
            if ctx["framing"] == "adversarial" and ctx["hint"] == 0:
                return refuse
            return {"text": self._solve(ctx), "meta": {}}
        raise ValueError(self.behavior)


def make_client(spec: str, api_key=None, base_url=None):
    provider, _, rest = spec.partition(":")
    if provider == "gemini":
        return Gemini(rest, api_key)
    if provider == "openai":
        return OpenAICompat(rest, api_key=api_key)
    if provider == "anthropic":
        return Anthropic(rest, api_key)
    if provider == "compat":
        model, _, base = rest.partition("@")
        return OpenAICompat(model, base_url or base or "http://localhost:11434/v1", "COMPAT_API_KEY",
                            optional_key=True, api_key=api_key)
    if provider == "mock":
        return Mock(rest or "erode")
    raise ValueError(f"Unknown model spec: {spec}")
