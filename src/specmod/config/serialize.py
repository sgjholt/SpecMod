"""Minimal TOML writer for configuration.

Python has a TOML reader in the standard library (``tomllib``) but no writer,
and the configuration schema is narrow enough — sections of scalars, flat
sequences, and one string-keyed float mapping — that a dependency is not worth
it.

TOML has no null. Keys whose value is ``None`` are written as commented-out
placeholders so a frozen file still documents that the option exists.
"""

from __future__ import annotations

from typing import Any

from .sections import Config

__all__ = ["to_toml"]


def _fmt(value: Any) -> str:
    if isinstance(value, bool):  # before int — bool is a subclass of int
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} = {_fmt(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"Cannot serialise {type(value).__name__} to TOML: {value!r}")


def to_toml(config: Config, *, header: str | None = None) -> str:
    """Render a configuration as TOML.

    The output round-trips through :meth:`Config.from_dict`, so a frozen file
    reproduces the configuration it was frozen from.
    """
    lines: list[str] = []
    if header:
        lines.extend(f"# {line}" for line in header.splitlines())
        lines.append("")

    data = config.to_dict()
    for section in sorted(data):
        lines.append(f"[{section}]")
        for key in sorted(data[section]):
            value = data[section][key]
            if value is None:
                lines.append(f"# {key} = ")  # TOML has no null
            elif isinstance(value, dict) and not value:
                lines.append(f"{key} = {{}}")
            else:
                lines.append(f"{key} = {_fmt(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
