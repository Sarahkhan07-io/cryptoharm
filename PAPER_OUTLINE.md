# CryptoHarm: paper plan

## Working title
CryptoHarm: A Verification-Based Benchmark of Harm Escalation in Language Models on Weak-Cryptography Tasks

## Abstract (fill the bracketed numbers from your real results only)
Safety evaluations of language models usually rely on model-graded judgments of whether a response is harmful.
We introduce CryptoHarm, a benchmark in which success is verified exactly: toy cryptanalysis tasks (close-prime RSA,
small-field ECDLP, PIN-derived keystreams, repeating-key XOR) whose answers can be checked mechanically. We vary
graded hint levels (none, weakness class, named technique), user framing (neutral, authorized, stated unauthorized
intent) and difficulty, which separates a model's *willingness* to assist from its *capability*. Across [N] models
and [M] generated instances, we find [hint effect on verified success], [change in refusal when hints are given
under adversarial framing], and [willingness gap between neutral and adversarial framing]. We release generators,
verifiers and all model outputs.

## Contributions
1. Exact-verification benchmark with deterministic instance generation (no LLM judge for capability).
2. Factorial design (hint x framing x difficulty) that disentangles willingness from capability.
3. Metrics: ASR, refusal rate, hint-escalation delta, refusal erosion, adversarial-vs-neutral willingness gap, with
   instance-clustered CIs and paired tests.
4. Open release of code, instances and responses, with a responsible-release statement.

## Threat model and construct validity (reviewers will test this section)
- Define "harm escalation": increase in verified-useful assistance toward an attacker-framed goal as the request
  becomes more specific. State plainly that toy tasks are a proxy, not real-world harm.
- Controls: neutral and authorized framings show whether adversarial framing changes behaviour at equal difficulty.
- Show ASR rises with hint level and falls with difficulty (sanity), so hints measure uplift and not noise.
- Report refusal-heuristic validity: hand-label `audit_sample.csv` (two annotators if possible), give agreement.

## Experiment plan (minimum for a workshop paper)
- Models: at least 5, covering 3+ vendors, one open-weight model, and two sizes within one family.
- 10+ instances per task x difficulty, 3+ samples per prompt, both modes if possible (code mode reflects capability).
- Report mean +/- CI, never single runs; keep temperature fixed and documented.
- Ablations: hint level, framing, difficulty, answer vs code mode.
- Limitations section: toy scale, English prompts only, heuristic refusals, provider filters, API model drift.

## Venue fit
arXiv + an AI-safety / security / evaluation workshop is realistic after the experiments above. A main-track
datasets-and-benchmarks submission needs more tasks (add e.g. weak hashing, ECB patterns, nonce reuse) and a stronger
conceptual result.
