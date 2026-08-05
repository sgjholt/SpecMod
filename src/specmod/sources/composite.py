"""Composing source, attenuation and motion into something a fitter can call."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.units import Motion
from .attenuation import AttenuationModel, get_attenuation_model
from .motion import motion_scaling
from .source import SourceModel, get_source_model

__all__ = ["SpectralModel", "build_model", "from_config"]


@dataclass(frozen=True)
class SpectralModel:
    """A source, an attenuation law and a motion, evaluated together.

    .. math::

        \\log_{10} A(f) = \\log_{10} S(f) + \\log_{10} D(f) + \\log_{10} G(f)

    Immutable and self-describing: it knows its own parameter names, so a
    fitter does not have to be told them separately and cannot be told them
    wrongly. That is what the legacy could not do — ``FitSpectra.set_model``
    took a bare function, and the parameter names came from introspecting its
    signature, so the model and its parameters could disagree.
    """

    source: SourceModel
    attenuation: AttenuationModel
    motion: Motion

    @property
    def parameters(self) -> tuple[str, ...]:
        """Free parameters, in the order :meth:`evaluate` takes them."""
        return ("llpsp", "fc", *self.attenuation.parameters)

    def evaluate(
        self, freq: NDArray[np.float64], *values: float
    ) -> NDArray[np.float64]:
        """``log10 A(f)`` for the given parameter values, in order."""
        expected = len(self.parameters)
        if len(values) != expected:
            raise TypeError(
                f"{self.describe()} takes {expected} parameters "
                f"{self.parameters}, got {len(values)}"
            )
        log10_omega, f_c, *decay = values
        return np.asarray(
            self.source.log10_shape(freq, log10_omega, f_c)
            + self.attenuation.log10_decay(freq, *decay)
            + motion_scaling(freq, self.motion)
        )

    def describe(self) -> str:
        """One line naming every choice this model makes."""
        return (
            f"{self.source.name}+{self.attenuation.name} in {Motion(self.motion).value}"
        )

    def as_callable(self) -> Callable[..., NDArray[np.float64]]:
        """A plain function of ``(f, llpsp, fc, ...)``, for ``lmfit.Model``.

        ``lmfit`` discovers parameter names with :func:`inspect.signature`
        (checked, rather than assumed — it does not read ``co_varnames``), so
        attaching a ``__signature__`` is enough and no code has to be
        generated. A wrapper taking ``*args`` would give the fit a single
        positional parameter and nothing to vary.
        """

        names = self.parameters

        def evaluate(
            f: NDArray[np.float64], *args: float, **kwargs: float
        ) -> NDArray[np.float64]:
            # lmfit reads the names off `__signature__` but calls with
            # keywords, so both forms have to work — and the keyword form has
            # to be put back into the declared order, not the caller's.
            values = args if args else tuple(kwargs[name] for name in names)
            return self.evaluate(f, *values)

        evaluate.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter("f", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            + [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in self.parameters
            ]
        )
        evaluate.__name__ = self.describe()
        return evaluate


def from_config() -> SpectralModel:
    """The model the current configuration asks for.

    One call, so that ``[model] source = "boatwright"`` in a study file is what
    decides the shape being fitted — which it was not before.
    """
    from ..config import load_config  # noqa: PLC0415 - avoids a config->sources cycle

    model = load_config().config.model
    return build_model(
        source=model.source,
        motion=model.motion,
        frequency_dependent_attenuation=model.frequency_dependent_attenuation,
    )


def build_model(
    *,
    source: str = "brune",
    motion: Motion | str = Motion.VELOCITY,
    frequency_dependent_attenuation: bool = False,
) -> SpectralModel:
    """Assemble a :class:`SpectralModel` from configuration names.

    This is the join that did not exist. ``ModelConfig.source`` was a
    ``Literal["brune", "boatwright"]`` that nothing read: ``FitSpectra`` took
    the model *function* as an argument, so the caller supplied Brune or
    Boatwright by hand and the configured value was decorative. Anything
    reading configuration now comes through here.
    """
    return SpectralModel(
        source=get_source_model(source),
        attenuation=get_attenuation_model(
            "frequency_dependent_q" if frequency_dependent_attenuation else "constant_q"
        ),
        motion=Motion(motion),
    )
