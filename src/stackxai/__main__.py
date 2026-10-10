"""Command-line interface: ``python -m stackxai {run,profile,spaces,plots}``."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import load_config


def main(argv=None):
    """Parse the command line and run the chosen command; ``argv`` defaults to ``sys.argv``."""
    parser = argparse.ArgumentParser(prog="stackxai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the full experiment")
    r.add_argument("--config", required=True, nargs="+", help="one or more configuration files, run in order")
    r.add_argument("--no-resume", action="store_true", help="ignore cached intermediate results")
    r.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                   help="override configuration entries, e.g. run.n_jobs=8 tuning.mode=single")
    p = sub.add_parser("profile", help="profile a trained pipeline in a fresh process")
    p.add_argument("--run-dir", required=True)
    sub.add_parser("spaces", help="print the hyperparameter search spaces")
    g = sub.add_parser("plots", help="redraw the figures of a finished run from its stored results, without "
                                     "recomputation (e.g. after changing explain.feature_labels)")
    g.add_argument("--run-dir", required=True)
    g.add_argument("--config", help="configuration whose explain.feature_labels replace those of the run")
    args = parser.parse_args(argv)

    if args.command == "run":
        from .pipeline import run
        overrides: dict = {}
        for item in args.set:
            key, value = item.split("=", 1)
            node = overrides
            *parents, leaf = key.split(".")
            for k in parents:
                node = node.setdefault(k, {})
            node[leaf] = yaml.safe_load(value)
        for path in args.config:
            run(load_config(path, overrides), resume=not args.no_resume)
    elif args.command == "plots":
        from .pipeline import redraw
        redraw(Path(args.run_dir), args.config)
    elif args.command == "profile":
        from .profiling import profile
        run_dir = Path(args.run_dir)
        with open(run_dir / "config.resolved.yaml", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        profile(run_dir, cfg)
    else:
        from .models import space_table
        print(space_table().to_string(index=False))


if __name__ == "__main__":
    main()
