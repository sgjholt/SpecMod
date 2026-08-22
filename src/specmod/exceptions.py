"""The exception hierarchy :mod:`specmod.api` raises.

Three kinds, because a caller does three different things with them:

- :class:`InvalidInputError` — the caller's data or arguments are wrong. A
  program shows a form error; a person fixes the input.
- :class:`MissingBackendError` — the code is fine and the environment is not:
  an optional extra is not installed. Fixed by installing something, and
  avoidable up front with :func:`specmod.api.available_estimators`.
- :class:`InternalError` — an invariant inside SpecMod is broken. Nothing the
  caller can do; it is a bug report.

Each also inherits the builtin exception the corresponding internal code
raises today, so ``except ValueError`` keeps working and internals can migrate
one at a time without a flag day.

**Internals still raise the builtins.** :mod:`specmod.api` translates at its
own boundary, so the guarantee is specific: functions reached *through
`specmod.api`* raise this hierarchy. Reaching around it gets the builtins.
"""

from __future__ import annotations

__all__ = [
    "InternalError",
    "InvalidInputError",
    "MissingBackendError",
    "SpecModError",
]


class SpecModError(Exception):
    """Base class for every error SpecMod raises deliberately."""


class InvalidInputError(SpecModError, ValueError):
    """The data or arguments given to SpecMod are not usable.

    A record containing NaN, a frequency axis that does not belong to its
    record, an unknown estimator name, a band with no samples in it.
    """


class MissingBackendError(SpecModError, ImportError):
    """An optional backend is not installed.

    Raised at call time rather than import time, so a default install stays
    importable. :func:`specmod.api.available_estimators` answers the same
    question without provoking the error.
    """


class InternalError(SpecModError, RuntimeError):
    """An invariant inside SpecMod does not hold.

    Not caused by the caller and not fixable by them. If one of these reaches
    you, it is a bug in SpecMod.
    """
