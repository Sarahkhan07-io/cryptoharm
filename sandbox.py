"""Optional code-execution mode. WARNING: AST screening + subprocess timeout is a mitigation, not a security
boundary. Run code mode inside a VM/container if you do not fully trust the models being evaluated."""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile

ALLOWED_IMPORTS = {"math", "itertools", "hashlib", "string", "collections", "functools", "re", "binascii",
                   "struct", "fractions", "operator", "heapq", "bisect", "time"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "getattr", "setattr", "delattr", "globals",
                "locals", "vars", "dir", "breakpoint", "memoryview", "exit", "quit"}
_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(text: str):
    m = _CODE_RE.findall(text or "")
    return m[0] if m else None


def check_code(code: str):
    """Return None if acceptable, else a reason string."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax: {e.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import {node.module}"
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            return f"name {node.id}"
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in {"modules", "system"}):
            return f"attr {node.attr}"
    return None


def run_code(code: str, timeout: float = 20.0) -> dict:
    reason = check_code(code)
    if reason:
        return {"status": "blocked", "reason": reason, "answer": None}
    env = {k: v for k, v in os.environ.items() if k in ("SYSTEMROOT", "PATH")}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sol.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, env=env, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "answer": None}
    lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
    if r.returncode != 0 or not lines:
        return {"status": "error", "stderr": r.stderr[-500:], "answer": None}
    return {"status": "ok", "answer": lines[-1].strip()}
