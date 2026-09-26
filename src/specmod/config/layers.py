"""Layered configuration resolution.

Precedence, lowest to highest:

1. Package defaults (:mod:`specmod.config.sections`)
2. A committed project config, ``specmod.toml``
3. A local override, ``specmod.local.toml`` — gitignored by default
4. Environment variables, ``SPECMOD_<SECTION>__<KEY>``
5. Explicit keyword arguments

Local overrides are deliberately uncommitted, which would make a run
irreproducible from the repository alone. That is resolved by recording the
*resolved* configuration in every output rather than by forbidding local files;
see :mod:`specmod.config.provenance`.

:func:`using` holds one configuration across the ambient reads the pipeline
makes, which is how a regression test fixes the settings its numbers were
produced under.
"""

from __future__ import annotations

import ast
import os
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields
from pathlib import Path
from typing import Any

from .sections import Config

__all__ = [
    "LAYER_NAMES",
    "ResolvedConfig",
    "load_config",
    "using",
]

PROJECT_FILE = "specmod.toml"
LOCAL_FILE = "specmod.local.toml"
ENV_PREFIX = "SPECMOD_"

LAYER_NAMES = ("default", PROJECT_FILE, LOCAL_FILE, "environment", "arguments")


class ResolvedConfig:
    """A :class:`Config` plus the layer each value came from.

    The provenance is what makes ``specmod config show`` able to answer "why
    did this run differ", which is otherwise guesswork once local overrides and
    environment variables are in play.
    """

    def __init__(self, config: Config, sources: dict[str, str]) -> None:
        self.config = config
        #: Maps ``"section.key"`` to the name of the layer that set it.
        self.sources = sources

    def source_of(self, dotted: str) -> str:
        return self.sources.get(dotted, "default")

    def explain(self) -> str:
        """Render the resolved config with the origin of every value."""
        lines: list[str] = []
        data = self.config.to_dict()
        for section in sorted(data):
            lines.append(f"[{section}]")
            for key in sorted(data[section]):
                dotted = f"{section}.{key}"
                origin = self.source_of(dotted)
                marker = "" if origin == "default" else f"   <- {origin}"
                lines.append(f"  {key} = {data[section][key]!r}{marker}")
            lines.append("")
        return "\n".join(lines)


#: The configuration :func:`using` is currently holding, if any. A
#: ``ContextVar`` rather than a module global so that a pin does not leak
#: between threads or between concurrent tasks — each gets its own view, and
#: the reset token restores exactly the previous one however the block exits.
_PINNED: ContextVar[ResolvedConfig | None] = ContextVar(
    "specmod_pinned_config", default=None
)


@contextmanager
def using(
    project_file: Path | str | None = None, **overrides: dict[str, Any]
) -> Iterator[ResolvedConfig]:
    """Hold one configuration for every ambient read inside the block.

    The pipeline reads settings by calling :func:`load_config` with no
    arguments, deep inside functions that take no configuration parameter.
    Passing a config down to them is not possible, so a caller that needs to
    fix the settings — a regression test, a study script — pins them here::

        with config.using("tests/golden/reference.toml"):
            result = spectrum_set_from_streams(signal, noise, estimator="fft")

    Everything the block calls sees that configuration, and only that: the
    local file and the environment are excluded, since a pin whose value still
    depended on the machine would not be one.

    Explicit arguments still win — ``load_config(project_file=...)`` inside the
    block reads what it was asked for. The pin replaces the *ambient* answer,
    not every answer.
    """
    resolved = load_config(
        project_file=project_file, use_local=False, use_env=False, **overrides
    )
    token = _PINNED.set(resolved)
    try:
        yield resolved
    finally:
        _PINNED.reset(token)


def pinned() -> ResolvedConfig | None:
    """The configuration :func:`using` is holding, or ``None``."""
    return _PINNED.get()


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _env_overrides() -> dict[str, dict[str, Any]]:
    """Collect ``SPECMOD_SNR__TOLERANCE=4`` style overrides.

    Values are parsed as Python literals where possible so numbers and booleans
    do not arrive as strings; anything else is left as text.
    """
    out: dict[str, dict[str, Any]] = {}
    valid = {f.name for f in fields(Config)}
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(ENV_PREFIX) or "__" not in raw_key:
            continue
        section, _, key = raw_key[len(ENV_PREFIX) :].partition("__")
        section, key = section.lower(), key.lower()
        if section not in valid:
            continue
        try:
            value: Any = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        out.setdefault(section, {})[key] = value
    return out


def _merge(
    base: dict[str, dict[str, Any]],
    incoming: dict[str, Any],
    layer: str,
    sources: dict[str, str],
) -> None:
    for section, values in incoming.items():
        if not isinstance(values, dict):
            raise ValueError(
                f"Configuration section [{section}] must be a table, got "
                f"{type(values).__name__}."
            )
        for key, value in values.items():
            base.setdefault(section, {})[key] = value
            sources[f"{section}.{key}"] = layer


def load_config(
    start: Path | str | None = None,
    /,
    *,
    project_file: Path | str | None = None,
    use_local: bool = True,
    use_env: bool = True,
    **overrides: dict[str, Any],
) -> ResolvedConfig:
    """Resolve configuration through all layers.

    Parameters
    ----------
    start
        Directory to search for config files. Defaults to the current
        directory. The search does not walk upwards — an implicit parent search
        makes it unclear which file a run actually used. Positional-only, so
        that a section named ``start`` in ``**overrides`` cannot collide with
        it.
    project_file
        Explicit path to a committed config, bypassing the search. This is how
        a study config (``studies/magna_2020_paper.toml``) is applied, and how
        tests pin an explicit configuration rather than inheriting defaults.
    use_local, use_env
        Disable the local-file and environment layers. Tests set both to False
        so a developer's machine cannot influence a result.
    **overrides
        Section-keyed dicts, e.g. ``snr={"tolerance": 4}``. Highest precedence.

    Notes
    -----
    A call with no arguments is an *ambient* read, and returns whatever
    :func:`using` is holding when it is holding something. Any argument at all
    makes the call explicit and resolves the layers as documented above, so a
    caller that asks for a particular file gets that file.
    """
    ambient = (
        start is None
        and project_file is None
        and use_local
        and use_env
        and not overrides
    )
    if ambient:
        held = _PINNED.get()
        if held is not None:
            return held

    root = Path(start) if start is not None else Path.cwd()
    merged: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}

    if project_file is not None:
        path = Path(project_file)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        _merge(merged, _read_toml(path), str(path), sources)
    else:
        candidate = root / PROJECT_FILE
        if candidate.is_file():
            _merge(merged, _read_toml(candidate), PROJECT_FILE, sources)

    if use_local:
        local = root / LOCAL_FILE
        if local.is_file():
            _merge(merged, _read_toml(local), LOCAL_FILE, sources)

    if use_env:
        _merge(merged, _env_overrides(), "environment", sources)

    if overrides:
        _merge(merged, overrides, "arguments", sources)

    return ResolvedConfig(Config.from_dict(merged), sources)
