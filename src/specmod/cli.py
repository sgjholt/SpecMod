"""Command line interface.

Currently covers configuration inspection. ``fetch`` arrives with
:mod:`specmod.acquire`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .config import load_config
from .config.provenance import Provenance
from .config.serialize import to_toml


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="specmod", description=__doc__)
    parser.add_argument("--version", action="version", version=f"specmod {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    cfg = sub.add_parser("config", help="inspect or export configuration")
    cfg_sub = cfg.add_subparsers(dest="config_command", required=True)

    show = cfg_sub.add_parser(
        "show", help="print the resolved configuration and where each value came from"
    )
    show.add_argument("-c", "--config", help="explicit config file to apply")
    show.add_argument(
        "--no-local", action="store_true", help="ignore specmod.local.toml"
    )
    show.add_argument("--no-env", action="store_true", help="ignore SPECMOD_* vars")
    show.add_argument(
        "--provenance",
        action="store_true",
        help="emit the JSON provenance record instead of the readable form",
    )

    freeze = cfg_sub.add_parser(
        "freeze",
        help="write the resolved configuration as TOML, for committing as a study",
    )
    freeze.add_argument("-c", "--config", help="explicit config file to apply")
    freeze.add_argument(
        "--no-local", action="store_true", help="ignore specmod.local.toml"
    )
    freeze.add_argument("--no-env", action="store_true", help="ignore SPECMOD_* vars")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "config":
        resolved = load_config(
            project_file=args.config,
            use_local=not args.no_local,
            use_env=not args.no_env,
        )
        if args.config_command == "show":
            if args.provenance:
                print(
                    Provenance.capture(
                        resolved.config, sources=resolved.sources
                    ).to_json()
                )
            else:
                print(resolved.explain(), end="")
            return 0
        if args.config_command == "freeze":
            print(
                to_toml(
                    resolved.config,
                    header=(
                        f"Frozen by specmod {__version__}.\n"
                        "Commit this alongside the results it produced."
                    ),
                ),
                end="",
            )
            return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
