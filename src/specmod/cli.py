"""Command line interface.

Covers configuration inspection and dataset acquisition.

Built on ``click``: every command SpecMod grows should be a ``click`` command
so the whole surface stays consistent — one convention for options, one for
help text, and composable groups rather than nested ``argparse`` subparsers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import click

from . import __version__
from .acquire import fetch, verify
from .config import load_config
from .config.provenance import Provenance
from .config.serialize import to_toml

#: Shared by every command that resolves configuration, so the layering flags
#: cannot drift apart between them.
_resolution_options = [
    click.option(
        "-c",
        "--config",
        "config_file",
        type=click.Path(exists=True, dir_okay=False),
        help="Explicit config file to apply.",
    ),
    click.option("--no-local", is_flag=True, help="Ignore specmod.local.toml."),
    click.option("--no-env", is_flag=True, help="Ignore SPECMOD_* variables."),
]


F = TypeVar("F", bound=Callable[..., Any])


def resolution_options(fn: F) -> F:
    """Apply the shared config-resolution options to a command."""
    for option in reversed(_resolution_options):
        fn = option(fn)
    return fn


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="specmod")
def main() -> None:
    """Process and model seismic spectra."""


@main.group()
def config() -> None:
    """Inspect or export configuration."""


@config.command("show")
@resolution_options
@click.option(
    "--provenance",
    is_flag=True,
    help="Emit the JSON provenance record instead of the readable form.",
)
def config_show(
    config_file: str | None, no_local: bool, no_env: bool, provenance: bool
) -> None:
    """Print the resolved configuration and where each value came from."""
    resolved = load_config(
        project_file=config_file, use_local=not no_local, use_env=not no_env
    )
    if provenance:
        click.echo(
            Provenance.capture(resolved.config, sources=resolved.sources).to_json()
        )
    else:
        click.echo(resolved.explain(), nl=False)


@config.command("freeze")
@resolution_options
def config_freeze(config_file: str | None, no_local: bool, no_env: bool) -> None:
    """Write the resolved configuration as TOML, for committing as a study."""
    resolved = load_config(
        project_file=config_file, use_local=not no_local, use_env=not no_env
    )
    click.echo(
        to_toml(
            resolved.config,
            header=(
                f"Frozen by specmod {__version__}.\n"
                "Commit this alongside the results it produced."
            ),
        ),
        nl=False,
    )


@main.command("fetch")
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--out",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to write the event and its manifest into.",
)
@click.option(
    "--verify",
    "verify_only",
    is_flag=True,
    help="Re-hash an existing fetch against its manifest instead of fetching.",
)
def fetch_command(config_file: str, out: str, verify_only: bool) -> None:
    """Fetch an event described by an acquisition config.

    This is the one command that uses the network, which is why it is explicit
    rather than something a test could reach by accident.
    """
    if verify_only:
        problems = verify(out)
        for problem in problems:
            click.echo(problem, err=True)
        if problems:
            raise SystemExit(1)
        click.echo(f"{out}: matches its manifest")
        return

    manifest = fetch(config_file, out=out)
    channels = manifest["resolved"]["channels"]
    click.echo(
        f"{manifest['name']}: {len(channels)} channels from "
        f"{manifest['data_centre']} -> {out}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
