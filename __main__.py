import argparse

from . import analyze as _an, runner, tasks


def main():
    p = argparse.ArgumentParser(prog="cryptoharm", description="CryptoHarm v2")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the benchmark (resumable)")
    r.add_argument("--models", nargs="+", required=True, help="e.g. gemini:gemini-2.5-flash mock:oracle")
    r.add_argument("--tasks", nargs="+", default=["all"], choices=["all", *tasks.TASKS])
    r.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"], choices=tasks.DIFFICULTIES)
    r.add_argument("--instances", type=int, default=10, help="instances per task x difficulty")
    r.add_argument("--hints", nargs="+", type=int, default=[0, 1, 2], choices=[0, 1, 2])
    r.add_argument("--framings", nargs="+", default=["neutral", "adversarial"], choices=["neutral", "authorized", "adversarial"])
    r.add_argument("--samples", type=int, default=3, help="independent samples per prompt")
    r.add_argument("--mode", choices=["answer", "code"], default="answer")
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--seed", type=int, default=1337)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep after each request (rate limits)")
    r.add_argument("--exec-timeout", type=float, default=20.0)
    r.add_argument("--out", default="results/run1")
    a = sub.add_parser("analyze", help="tables + figures from raw.jsonl")
    a.add_argument("--raw", required=True)
    a.add_argument("--out", required=True)
    u = sub.add_parser("ui", help="open the web UI")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8000)
    u.add_argument("--open-browser", action="store_true")
    args = p.parse_args()
    if args.cmd == "ui":
        from . import ui
        ui.serve(args.host, args.port, args.open_browser)
    elif args.cmd == "run":
        runner.run(args)
    else:
        _an.analyze(args.raw, args.out)


if __name__ == "__main__":
    main()
