"""Waveforms in, :class:`~specmod.core.SpectrumSet` out.

The replacement for ``spectral.Spectra.from_streams``, and the seam that
``spectral`` becomes a shell over. Given a signal stream and the noise cut to
match it, this produces the immutable containers directly — trace to
:class:`~specmod.core.Spectrum` to :class:`~specmod.core.SpectrumPair` — with
none of ``spectral``'s mutable classes in between.

**Why it lives here rather than in ``core``.** ``core`` and ``transforms``
know nothing about ObsPy: they operate on arrays, a duration and a sampling
rate. That is worth keeping, because it is what makes them testable without
constructing a Stream and usable on data that never came from one. This module
is the single place that knows what a ``Trace`` is, so the dependency stops at
one file rather than spreading through the containers.

**The route is shorter than the legacy one, and identical in value.**
``spectral.Spectrum`` converts the estimator's output to a PSD, copies the
arrays out, converts back to magnitude, then ``SNP`` wraps them in a
``core.Spectrum`` again — a round trip through two mutable objects that exists
only because the legacy call sequence had it. Here the estimator's spectrum is
converted once, to the unfolded magnitude convention the pipeline reads
``Omega`` in, and handed straight to :meth:`SpectrumPair.compare`.

That the two agree is not asserted, it is measured:
``tests/test_pipeline.py`` runs both over the same 28 windows and all five
estimators and requires the numbers to match to 1 part in 1e12.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import load_config
from .core import AmplitudeKind, Motion, Spectrum, SpectrumPair, SpectrumSet
from .transforms import ESTIMATORS

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping

    from numpy.typing import NDArray

__all__ = [
    "estimate_spectrum",
    "pair_from_traces",
    "spectrum_from_trace",
    "spectrum_set_from_streams",
]


def estimate_spectrum(
    data: NDArray[np.float64],
    delta: float,
    *,
    motion: Motion | str = Motion.VELOCITY,
    **kwargs: Any,
) -> Spectrum:
    """Transform a record with whichever estimator the configuration names.

    The bridge to :mod:`specmod.transforms`. Every estimator — FFT, Welch,
    multitaper, Prieto, quadratic, CWT — becomes available through
    :class:`specmod.config.TransformConfig`, where before there was a hardcoded
    ``mtspec(data, delta, 3)``.

    Returns a :class:`specmod.core.Spectrum`, which carries its own units, so
    the caller no longer has to track how many times the record has been
    integrated or what amplitude convention is in force. **The convention is
    the folded one**, ``FAS``; :func:`spectrum_from_trace` is what converts to
    the unfolded magnitude the rest of the pipeline reads ``Omega`` in.

    Keyword arguments override the configured estimator's parameters, which is
    how the ``**kwargs`` passthrough from the stream entry points works.

    Lived in ``spectral`` until the direct pipeline existed, which put the one
    typed bridge to ``transforms`` inside the untyped legacy shell and made
    every caller of it untyped too.
    """
    transform = load_config().config.transform
    name = kwargs.pop("estimator", transform.estimator)
    if name == "mtspec":
        raise ValueError(
            "estimator='mtspec' is the pre-refactor Fortran backend and is not "
            "wired into the pipeline. Use 'prieto' for the same lineage with "
            "no compiler, or 'multitaper' for the native implementation."
        )

    cls = ESTIMATORS[name]
    # By constructor signature, not by `dataclasses.fields`. The registry is
    # typed over the `SpectralEstimator` protocol, which promises a `name` and
    # an `estimate` and says nothing about being a dataclass — so asking for
    # its fields is a claim the type does not support, and an estimator that
    # was a plain class would break it. The signature is what "accepts this
    # setting" actually means.
    accepted = set(inspect.signature(cls).parameters)
    settings = {
        key: value
        for key, value in dataclasses.asdict(transform).items()
        if key in accepted
    }

    # Filtering the *configuration* to what this estimator accepts is right:
    # `[transform]` holds every estimator's settings at once, and a CWT
    # parameter is not an error when the FFT is selected. Filtering the
    # caller's own keywords the same way is not. It silently discarded
    # anything the estimator did not recognise, so a typo, or an argument
    # meant for a different stage, simply did nothing.
    #
    # This is not hypothetical: `spectrum_set_from_streams(rotate_noise=False)`
    # looks exactly like it works. `rotate_noise` belongs to `compare`, not to
    # an estimator, so it was dropped here and the run silently kept the
    # configured value — which cost three wrong measurements before the
    # recorded settings gave it away.
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        compare_only = sorted(set(unknown) & set(_compare_settings()))
        hint = ""
        if compare_only:
            verb = "configures" if len(compare_only) == 1 else "configure"
            this = "it" if len(compare_only) == 1 else "them"
            hint = (
                f" {', '.join(compare_only)} {verb} the signal-to-noise "
                f"comparison, not the transform — pass {this} as "
                f"compare={{{', '.join(f'{k!r}: ...' for k in compare_only)}}}."
            )
        raise TypeError(
            f"{name} does not accept {', '.join(unknown)}. It takes "
            f"{', '.join(sorted(accepted - {'self'}))}.{hint}"
        )

    settings.update(kwargs)
    result: Spectrum = cls(**settings).estimate(data, delta, motion=motion)
    return result


#: Trace stats that are not data and do not survive a round trip through a
#: plain mapping. ``processing`` is an ever-growing list of ObsPy call strings,
#: ``sac``/``calib``/``__format`` are format-specific, and all of them break
#: equality comparison between two runs of the same pipeline.
_DROPPED_STATS = frozenset({"processing", "sac", "calib", "__format"})

_SCALAR = (float, int, str, np.floating, np.integer)


def _trace_meta(trace: Any) -> dict[str, Any]:
    """Trace stats as a plain, comparable mapping.

    Non-scalars — ``UTCDateTime`` for the picks and window edges, mostly — are
    stringified rather than dropped, because the window times are the record of
    what was cut and a spectrum that cannot say where it came from is not
    reproducible. Same rule the legacy ``__sanitise_trace_meta`` used, stated
    as a set rather than as a chain of ``if``\\s.
    """
    return {
        key: (value if isinstance(value, _SCALAR) else str(value))
        for key, value in dict(trace.stats).items()
        if key not in _DROPPED_STATS
    }


def spectrum_from_trace(trace: Any, **kwargs: Any) -> Spectrum:
    """Transform one trace into a spectrum on the pipeline's convention.

    The convention is ``MAGNITUDE``, the *unfolded* transform magnitude
    ``|X(f)|``, not the folded ``2|X|`` the estimators return. This is not a
    preference: ``Omega`` is defined on the unfolded spectrum — the
    long-period displacement plateau is ``|X(f -> 0)| = |int u dt|`` and
    ``M0`` is proportional to it — so reading it off a folded spectrum puts
    every moment out by two, which is 0.2 magnitude units. See
    :meth:`specmod.core.Spectrum.to_kind` for the conversion.

    ``kwargs`` override the configured estimator's parameters, including
    ``estimator`` itself.
    """
    spectrum = estimate_spectrum(
        np.asarray(trace.data, dtype=float),
        float(trace.stats.delta),
        **kwargs,
    )
    meta = _trace_meta(trace)
    meta["id"] = trace.id
    # The lowest frequency this window supports, captured here because
    # interpolation onto a shared axis destroys it. For an FFT or multitaper it
    # is 1/T; for the CWT the cone-of-influence floor, about 1.4x stricter on
    # these windows because a wavelet needs several cycles in the record rather
    # than one. Taking it from the axis rather than from a formula makes the
    # right rule apply to each without a special case.
    meta["resolution_floor"] = float(spectrum.freq.min())
    meta["estimator"] = spectrum.meta.get("estimator")

    return Spectrum(
        freq=spectrum.freq,
        amp=spectrum.to_kind(AmplitudeKind.MAGNITUDE).amp,
        motion=spectrum.motion,
        kind=AmplitudeKind.MAGNITUDE,
        duration=spectrum.duration,
        sampling_rate=spectrum.sampling_rate,
        meta=meta,
    )


def _compare_settings(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Configured arguments for :meth:`SpectrumPair.compare`.

    One place, so the per-pair and per-stream entry points cannot disagree
    about what the configuration said.
    """
    config = load_config().config
    settings: dict[str, Any] = {
        "threshold": config.snr.tolerance,
        "f_min": config.smoothing.f_min,
        "f_max": config.smoothing.f_max,
        "n_bins": config.smoothing.n_bins,
        "scale_parseval": config.snr.scale_parseval,
        "rotate_noise": config.snr.rotate_noise,
        "noise_model": config.snr.rotation_method,
        "resolution_floor": config.snr.resolution_floor,
        "bandwidth": config.snr.bandwidth_method,
        "rotation_space": config.snr.rotation_space,
    }
    settings.update(overrides or {})
    return settings


def pair_from_traces(
    signal_trace: Any,
    noise_trace: Any,
    *,
    compare: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SpectrumPair:
    """Transform a signal and its noise, and pair them.

    ``kwargs`` go to the estimator; ``compare`` overrides individual arguments
    to :meth:`SpectrumPair.compare`, which otherwise come from configuration.
    """
    signal = spectrum_from_trace(signal_trace, **kwargs)
    noise = spectrum_from_trace(noise_trace, **kwargs)
    return SpectrumPair.compare(signal, noise, **_compare_settings(compare))


def spectrum_set_from_streams(
    signal: Iterable[Any],
    noise: Iterable[Any],
    *,
    event: str = "",
    compare: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SpectrumSet:
    """Pair two streams trace by trace.

    The two are matched by **position**, as the legacy path matched them, since
    that is how ``preprocess.get_noise_p`` builds the noise: one trace per
    signal trace, in order. Mismatched ids are an error rather than a warning —
    a signal compared against another station's noise is not a degraded
    measurement, it is a meaningless one.

    ``event`` defaults to the origin time carried on the first trace, which is
    what the legacy container used to name the group.
    """
    settings = _compare_settings(compare)
    pairs: dict[str, SpectrumPair] = {}

    for signal_trace, noise_trace in zip(signal, noise, strict=True):
        if signal_trace.id != noise_trace.id:
            raise ValueError(
                f"signal {signal_trace.id} paired with noise {noise_trace.id}; "
                f"the streams are not in the same order"
            )
        signal_spectrum = spectrum_from_trace(signal_trace, **kwargs)
        noise_spectrum = spectrum_from_trace(noise_trace, **kwargs)
        pairs[signal_trace.id] = SpectrumPair.compare(
            signal_spectrum, noise_spectrum, **settings
        )
        if not event:
            event = str(signal_trace.stats.get("otime", ""))

    return SpectrumSet(pairs=pairs, event=event)
