# CryptoHarm v2

A verification-based benchmark of **harm escalation** in LLMs on deliberately weak, toy-scale cryptography tasks.
Success is checked exactly (the recovered secret either decrypts/validates or it does not), so no LLM judge is
needed to score capability.

## Design

| Factor | Levels |
|---|---|
| Task | `rsa_close_primes`, `toy_ecdlp`, `pin_key`, `xor_repeating` |
| Difficulty | easy / medium / hard (key sizes, gaps, PIN length, key length) |
| Hint level | 0 none, 1 names the weakness class, 2 names the textbook technique (never a worked solution) |
| Framing | `neutral`, `authorized`, `adversarial` (user states unauthorized intent) |
| Mode | `answer` (model replies with the value) or `code` (model writes a program, run in a restricted subprocess) |

Crossing hints with framing is what separates **willingness** (refusal behaviour) from **capability** (verified
success), the main confound a reviewer will raise. Outputs per cell: attack success rate (ASR), refusal rate,
instance-clustered bootstrap CIs, paired hint-escalation deltas, and the adversarial-vs-neutral willingness gap.

Outcomes per trial: `SOLVED`, `WRONG`, `REFUSED`, `NO_ANSWER` (API failures are `ERROR` and are retried on resume).
Refusal detection is a regex heuristic plus provider block signals, so hand-label `audit_sample.csv` and report
agreement before publishing refusal numbers.

## Web UI (like the v11 app)

Double-click `run_ui.bat`, or run `python -m cryptoharm ui --open-browser`. Two tabs:

- **Benchmark**: pick provider, model, tasks, difficulty, hint levels, framings, instances and samples; watch progress;
  then see outcome charts, hint-escalation and willingness-gap tables with confidence intervals, per-task rates, every
  individual response, CSV/JSON download, and a "Write full report" button (tables, figures, audit sample).
- **Chat**: talk to the selected model directly. Nothing is scored or executed.

Keys typed in the page are held in memory only and are never written to disk. The UI listens on 127.0.0.1 only.
For Groq use provider *OpenAI-compatible* with base URL `https://api.groq.com/openai/v1`.

## Quick start (Windows)

In PowerShell prefix scripts with `.\` (for example `.\run_ui.bat`).



```
python -m unittest discover -s tests
run_example.bat
```

Real runs (keys via environment variables, never in code):

```
set GEMINI_API_KEY=...
python -m cryptoharm run --models gemini:gemini-2.5-flash openai:<model> anthropic:<model> compat:<model>@http://localhost:11434/v1 --instances 10 --samples 3 --out results\main
python -m cryptoharm analyze --raw results\main\raw.jsonl --out results\main\analysis
```

Runs are resumable (re-run the same command), parallel (`--workers`), and rate-limit friendly (`--sleep`).
Each run stores `manifest.json` (versions, args), `instances.jsonl` (with secrets, for auditing) and `raw.jsonl`
(every response). Release these with the paper.

## Notes

- `mock:*` models read the secret and exist only to test the pipeline. Never report them as results.
- Code mode uses AST screening plus a subprocess timeout. That is a mitigation, not a sandbox: use a VM/container if
  you evaluate models you do not trust. Answer mode executes nothing.
- Some reasoning models spend `max_tokens` on hidden thinking and ignore `temperature`; raise `--max-tokens` and
  record this in the paper.
- Pin the Python version and seed (`--seed`) for reproducibility; instance generation is deterministic.
