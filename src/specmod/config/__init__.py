"""Configuration: semantic sections, layered overrides, recorded provenance.

See ``docs/REFACTOR_PLAN.md`` §4.7.

Defaults reproduce the behaviour shipped before the refactor. A study pins its
own values in a committed TOML file; personal experimentation goes in
``specmod.local.toml``, which is gitignored, and is promoted deliberately with
``specmod config freeze``.
"""

from __future__ import annotations

from .layers import LAYER_NAMES, ResolvedConfig, load_config
from .provenance import Provenance, config_hash
from .sections import (
    AcquireConfig,
    Config,
    FittingConfig,
    GeometryConfig,
    ModelConfig,
    SmoothingConfig,
    SnrConfig,
    TransformConfig,
    VizConfig,
    WindowsConfig,
)

__all__ = [
    "LAYER_NAMES",
    "AcquireConfig",
    "Config",
    "FittingConfig",
    "GeometryConfig",
    "ModelConfig",
    "Provenance",
    "ResolvedConfig",
    "SmoothingConfig",
    "SnrConfig",
    "TransformConfig",
    "VizConfig",
    "WindowsConfig",
    "config_hash",
    "load_config",
]
