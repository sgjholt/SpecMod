"""Provenance stamping: what makes a locally-overridden run reproducible.

Local config files are uncommitted by design, so the repository alone cannot
say how a given result was produced. The output can. Every artifact SpecMod
writes carries the fully resolved configuration, a short hash of it, and the
SpecMod version.

The version matters as much as the config. Defaults move between releases, so
without it "reproducible" fails silently across an upgrade — which is exactly
what made identifying the code behind the published Magna run so difficult (see
``docs/REFACTOR_PLAN.md`` §5.2.5).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .. import __version__
from .sections import Config

__all__ = ["Provenance", "config_hash"]


def _canonical(config: Config) -> str:
    """Serialise deterministically so the hash is stable across runs."""
    return json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))


def config_hash(config: Config, *, length: int = 12) -> str:
    """Short, stable digest of a configuration.

    Comparing two runs starts here: same hash means same settings, so any
    difference is in the data or the code, not the configuration.
    """
    return hashlib.sha256(_canonical(config).encode()).hexdigest()[:length]


@dataclass(frozen=True, slots=True)
class Provenance:
    """The record attached to every output."""

    specmod_version: str
    config: dict[str, Any]
    config_hash: str
    created_at: str
    sources: dict[str, str]

    @classmethod
    def capture(
        cls,
        config: Config,
        *,
        sources: dict[str, str] | None = None,
    ) -> Provenance:
        return cls(
            specmod_version=__version__,
            config=config.to_dict(),
            config_hash=config_hash(config),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            sources=dict(sources or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "specmod_version": self.specmod_version,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            "config": self.config,
            "config_sources": self.sources,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
