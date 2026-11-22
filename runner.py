"""Resumable, parallel benchmark runner. Appends one JSON row per (model, instance, hint, framing, trial)."""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import __version__, models, prompts, scoring, tasks
from .models import ApiError

_RETRY = (429, 500, 502, 503, 504)


def job_key(model, inst_id, hint, framing, mode, trial):
    return f"{model}|{inst_id}|h{hint}|{framing}|{mode}|t{trial}"


def run(a, progress_cb=None, cancel=None):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"
    task_list = tasks.TASKS if a.tasks == ["all"] else tuple(a.tasks)
    insts = [tasks.make_instance(t, d, i, a.seed) for t in task_list for d in a.difficulties for i in range(a.instances)]
    (out / "instances.jsonl").write_text("\n".join(json.dumps(i.to_json()) for i in insts) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "version": __version__, "python": sys.version, "platform": platform.platform(),
        "args": {k: v for k, v in vars(a).items() if k not in ("api_key",)}, "n_instances": len(insts), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")

    done = set()
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["outcome"] != "ERROR":
                done.add(r["key"])
    clients = {m: models.make_client(m, getattr(a, 'api_key', None), getattr(a, 'base_url', None)) for m in a.models}
    jobs = []
    for m, inst, h, f, t in itertools.product(a.models, insts, a.hints, a.framings, range(a.samples)):
        k = job_key(m, inst.instance_id, h, f, a.mode, t)
        if k not in done:
            jobs.append((k, m, inst, h, f, t))
    print(f"{len(insts)} instances, {len(jobs)} jobs to run ({len(done)} already done)")
    lock, counter = threading.Lock(), [0]

    def work(job):
        k, m, inst, h, f, t = job
        if cancel is not None and cancel.is_set():
            return
        prompt = prompts.build_prompt(inst, h, f, a.mode)
        ctx = {"secret_answer": inst.secret_answer(), "hint": h, "framing": f, "mode": a.mode} if clients[m].needs_context else None
        t0, resp, err = time.time(), None, None
        for attempt in range(6):
            try:
                resp = clients[m].generate(prompt, a.temperature, a.max_tokens, ctx)
                break
            except ApiError as e:
                err = str(e)
                if e.code in _RETRY and attempt < 5:
                    time.sleep(min(60, 2 ** attempt * 2))
                else:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                err = repr(e)
                time.sleep(min(60, 2 ** attempt * 2))
            except Exception as e:  # missing key, bad spec, ...
                err = repr(e)
                break
        row = {"key": k, "model": m, "task": inst.task, "difficulty": inst.difficulty, "instance_id": inst.instance_id,
               "hint": h, "framing": f, "mode": a.mode, "trial": t, "temperature": a.temperature,
               "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:12], "latency_s": round(time.time() - t0, 2),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if resp is None:
            row.update(outcome="ERROR", correct=0, refused=0, error=err, response="")
        else:
            row.update(scoring.score(inst, resp, a.mode, a.exec_timeout))
            row.update(response=resp["text"], meta=resp.get("meta", {}), error=None)
        with lock:
            with raw.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            counter[0] += 1
            if progress_cb:
                progress_cb(counter[0], len(jobs))
            if counter[0] % 20 == 0 or counter[0] == len(jobs):
                print(f"  {counter[0]}/{len(jobs)}", flush=True)
        if a.sleep:
            time.sleep(a.sleep)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            fut.result()
    print(f"Done. Raw results: {raw}")
