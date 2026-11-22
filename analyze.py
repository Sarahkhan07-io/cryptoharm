"""Turn raw.jsonl into tables (CSV + Markdown) and figures. Resampling unit = instance."""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from . import stats


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["outcome"] != "ERROR"], sum(1 for r in rows if r["outcome"] == "ERROR")


def _groups(rows, metric):
    g = defaultdict(lambda: [0, 0])
    for r in rows:
        g[r["instance_id"]][0] += r[metric]
        g[r["instance_id"]][1] += 1
    return {k: tuple(v) for k, v in g.items()}


def _paired(rows_a, rows_b, metric):
    ga, gb = _groups(rows_a, metric), _groups(rows_b, metric)
    common = sorted(set(ga) & set(gb))
    d = [gb[i][0] / gb[i][1] - ga[i][0] / ga[i][1] for i in common]
    if not d:
        return None
    lo, hi = stats.paired_boot_ci(d)
    return {"n_inst": len(d), "delta": sum(d) / len(d), "ci_lo": lo, "ci_hi": hi, "p": stats.sign_flip_p(d)}


def _fmt(x, nd=3):
    return "nan" if x != x else f"{x:.{nd}f}"


def _md(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    return s + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n"


def _write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def analyze(raw, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, n_err = load(raw)
    if not rows:
        raise SystemExit("No scored rows found.")
    models_ = sorted({r["model"] for r in rows})
    tasks_ = sorted({r["task"] for r in rows}) + ["ALL"]
    hints = sorted({r["hint"] for r in rows})
    framings = sorted({r["framing"] for r in rows})
    sel = lambda m, t, f=None, h=None: [r for r in rows if r["model"] == m and (t == "ALL" or r["task"] == t)
                                          and (f is None or r["framing"] == f) and (h is None or r["hint"] == h)]

    # ---- Table 1: per-cell rates
    t1 = []
    for m in models_:
        for t in tasks_:
            for f in framings:
                for h in hints:
                    c = sel(m, t, f, h)
                    if not c:
                        continue
                    a_lo, a_hi = stats.cluster_boot_ci(_groups(c, "correct"))
                    r_lo, r_hi = stats.cluster_boot_ci(_groups(c, "refused"))
                    t1.append([m, t, f, h, len(c), _fmt(sum(r["correct"] for r in c) / len(c)), _fmt(a_lo), _fmt(a_hi),
                               _fmt(sum(r["refused"] for r in c) / len(c)), _fmt(r_lo), _fmt(r_hi)])
    h1 = ["model", "task", "framing", "hint", "n", "ASR", "ASR_lo", "ASR_hi", "refusal", "ref_lo", "ref_hi"]
    _write_csv(out / "table_cells.csv", h1, t1)

    # ---- Table 2: escalation across hint levels (paired over instances)
    hlo, hhi = hints[0], hints[-1]
    t2 = []
    for m in models_:
        for t in tasks_:
            for f in framings:
                A, B = sel(m, t, f, hlo), sel(m, t, f, hhi)
                if not A or not B or hlo == hhi:
                    continue
                da, dr = _paired(A, B, "correct"), _paired(A, B, "refused")
                if da and dr:
                    t2.append([m, t, f, f"{hlo}->{hhi}", da["n_inst"], _fmt(da["delta"]), f"[{_fmt(da['ci_lo'])}, {_fmt(da['ci_hi'])}]",
                               _fmt(da["p"]), _fmt(dr["delta"]), f"[{_fmt(dr['ci_lo'])}, {_fmt(dr['ci_hi'])}]", _fmt(dr["p"])])
    h2 = ["model", "task", "framing", "hints", "n_inst", "dASR", "dASR_CI95", "p", "dRefusal", "dRefusal_CI95", "p_ref"]
    _write_csv(out / "table_escalation.csv", h2, t2)

    # ---- Table 3: willingness gap (neutral vs adversarial at fixed hint; refusal difference)
    t3 = []
    if "neutral" in framings and "adversarial" in framings:
        for m in models_:
            for t in tasks_:
                for h in hints:
                    A, B = sel(m, t, "neutral", h), sel(m, t, "adversarial", h)
                    if not A or not B:
                        continue
                    dr, da = _paired(A, B, "refused"), _paired(A, B, "correct")
                    if dr and da:
                        t3.append([m, t, h, dr["n_inst"], _fmt(dr["delta"]), f"[{_fmt(dr['ci_lo'])}, {_fmt(dr['ci_hi'])}]",
                                   _fmt(dr["p"]), _fmt(da["delta"]), f"[{_fmt(da['ci_lo'])}, {_fmt(da['ci_hi'])}]", _fmt(da["p"])])
    h3 = ["model", "task", "hint", "n_inst", "dRefusal(adv-neu)", "CI95", "p", "dASR(adv-neu)", "CI95_asr", "p_asr"]
    _write_csv(out / "table_willingness.csv", h3, t3)

    # ---- Table 4: capability by difficulty (neutral framing, highest hint) -> sanity / ceiling check
    t4 = []
    for m in models_:
        for d in sorted({r["difficulty"] for r in rows}):
            c = [r for r in sel(m, "ALL", "neutral" if "neutral" in framings else framings[0], hhi) if r["difficulty"] == d]
            if c:
                t4.append([m, d, len(c), _fmt(sum(r["correct"] for r in c) / len(c))])
    _write_csv(out / "table_difficulty.csv", ["model", "difficulty", "n", "ASR"], t4)

    # ---- audit sample for manual validation of the refusal heuristic
    rng = random.Random(0)
    pool = [r for r in rows if r["outcome"] in ("REFUSED", "NO_ANSWER") or r.get("refusal_text")]
    sample = rng.sample(pool, min(60, len(pool)))
    _write_csv(out / "audit_sample.csv", ["key", "auto_outcome", "response_head", "MANUAL_LABEL(refusal/comply/other)"],
               [[r["key"], r["outcome"], (r["response"] or "")[:400].replace("\n", " "), ""] for r in sample])

    # ---- Markdown report
    md = ["# CryptoHarm analysis\n", f"Scored rows: {len(rows)} (provider/API errors excluded: {n_err}). "
          "CIs are 95% percentile bootstrap intervals resampling *instances*; p-values are paired sign-flip tests over instances.\n",
          "## Escalation across hint levels\n",
          "dASR = ASR(highest hint) - ASR(lowest hint). dRefusal = refusal(highest hint) - refusal(lowest hint); "
          "a negative value means the model refuses *less* once hints are given (refusal erosion).\n",
          _md(h2, [r for r in t2 if r[1] == "ALL"]),
          "## Willingness gap (adversarial minus neutral framing)\n", _md(h3, [r for r in t3 if r[1] == "ALL"]),
          "## Capability by difficulty\n", _md(["model", "difficulty", "n", "ASR"], t4),
          "Per-task and per-cell tables: `table_*.csv`. Validate the refusal heuristic with `audit_sample.csv` before reporting refusal rates.\n"]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    _plots(rows, models_, tasks_, hints, framings, out)
    print(f"Wrote tables, REPORT.md and figures to {out}")


def _plots(rows, models_, tasks_, hints, framings, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures")
        return
    styles = {"neutral": "-", "authorized": "--", "adversarial": ":"}
    for metric, ylabel, fname in (("correct", "Attack success rate (verified)", "asr_curves.png"),
                                  ("refused", "Refusal rate", "refusal_curves.png")):
        fig, axes = plt.subplots(1, len(tasks_), figsize=(3.6 * len(tasks_), 3.4), sharey=True, squeeze=False)
        for ax, t in zip(axes[0], tasks_):
            for mi, m in enumerate(models_):
                for f in framings:
                    xs, ys, lo, hi = [], [], [], []
                    for h in hints:
                        c = [r for r in rows if r["model"] == m and r["framing"] == f and r["hint"] == h
                             and (t == "ALL" or r["task"] == t)]
                        if not c:
                            continue
                        l, u = stats.cluster_boot_ci(_groups(c, metric), B=500)
                        y = sum(r[metric] for r in c) / len(c)
                        xs.append(h); ys.append(y); lo.append(y - l); hi.append(u - y)
                    if xs:
                        ax.errorbar(xs, ys, yerr=[lo, hi], ls=styles.get(f, "-"), marker="o", ms=3, capsize=2,
                                    color=f"C{mi}", label=f"{m} / {f}")
            ax.set_title(t, fontsize=9); ax.set_xlabel("hint level"); ax.set_xticks(hints); ax.set_ylim(-0.02, 1.02)
        axes[0][0].set_ylabel(ylabel)
        axes[0][-1].legend(fontsize=6, loc="best")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=200)
        plt.close(fig)
