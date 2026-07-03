"""CLI: ``spot-orchestrate {setup,stage-data,baseline,spot,preempt,ddp,ddp-preempt} [--dry-run]``.

You run this; it needs your AWS creds in the environment. A git-ignored ``.env``
in the current directory is loaded into the environment on startup (values are
never printed). ``--dry-run`` makes every AWS call a no-op that just logs what it
would do — use it to review before spending anything.
"""

from __future__ import annotations

import argparse
import os
import sys


def _load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a local .env (KEY=VALUE lines). Never echoes
    values; does not override variables already set in the environment."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def main() -> None:
    _load_dotenv()

    # --dry-run is attached to each subcommand (e.g. `setup --dry-run`) via a
    # shared parent parser, so it reads naturally after the command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="log AWS calls, execute none")

    parser = argparse.ArgumentParser(prog="spot-orchestrate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "stage-data", "baseline", "spot", "preempt", "ddp", "ddp-preempt"):
        sub.add_parser(name, parents=[common])

    args = parser.parse_args()

    # Imported after dotenv so config picks up the loaded environment, and so
    # `aws` (the only creds-touching module) is configured before any call.
    from . import aws, dataset, experiments, setup
    from .config import OrchestratorConfig

    aws.set_dry_run(args.dry_run)
    cfg = OrchestratorConfig()
    aws.set_region(cfg.region)

    if args.command == "setup":
        setup.ensure_infra(cfg)
    elif args.command == "stage-data":
        dataset.stage_data(cfg)
    elif args.command == "baseline":
        experiments.run_baseline(cfg)
    elif args.command == "spot":
        experiments.run_spot(cfg)
    elif args.command == "preempt":
        experiments.run_preempt(cfg)
    elif args.command == "ddp":
        experiments.run_ddp(cfg)
    elif args.command == "ddp-preempt":
        experiments.run_preempt(cfg, ddp=True)
    else:  # pragma: no cover — argparse enforces the choices
        parser.error(f"unknown command {args.command}")

    if args.dry_run:
        print("\n[dry-run] no AWS calls were made.", file=sys.stderr)


if __name__ == "__main__":
    main()
