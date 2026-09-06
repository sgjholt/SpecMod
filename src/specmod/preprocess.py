"""Geometry, picks and window cutting, on ObsPy streams.

**Nothing here mutates its argument.** Every function that takes a stream or a
trace returns a new one and leaves the caller's untouched, so a stream can be
windowed twice — as signal and as noise — from the same starting point, and a
pipeline that raises half way through has not damaged its input. This is the
package-wide rule stated in ``REFACTOR_PLAN.md`` §3, and it is what
:mod:`specmod.core` has always guaranteed.

The naming carries the contract. ``with_*`` returns a copy carrying something
extra — a distance, an origin time, picks. ``p_window``, ``s_window`` and
``coda_window`` return the cut window as a new stream. Nothing is named
``set_*``, because in this module nothing sets.

This module and :mod:`specmod.pipeline` are the only two that know what a
``Trace`` is. ObsPy ships no type information and no stub package is
published, so ``stubs/obspy`` in this repository declares the surface used
here; see ``stubs/README.md``. ``Stats`` is deliberately open, so
``tr.stats.delta`` is checked and ``tr.stats["p_time"]`` — one of the fields
this module writes — is not.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import obspy
from scipy.integrate import cumulative_trapezoid

from . import picks as pk

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence
    from os import PathLike

    from numpy.typing import NDArray
    from obspy import Inventory, Stream, Trace, UTCDateTime

#: How station coordinates are supplied to :func:`with_distance`.
STREAM_DISTANCE_METHODS = ["mseed", "sac", "list"]

#: Accepted values of ``time_after``. Both spellings of the relative mode are
#: taken by both cutting functions: they used to disagree, the P cutter calling
#: it ``"relative_time"`` and the S cutter ``"relative_ps"``, with neither
#: accepting the other's word for the same idea.
_ABSOLUTE = "absolute_time"
_RELATIVE = ("relative_time", "relative_ps")
TIME_AFTER_METHODS = (_ABSOLUTE, *_RELATIVE)


def _check_time_after(time_after: str) -> None:
    if time_after not in TIME_AFTER_METHODS:
        raise ValueError(
            f"time_after={time_after!r} is not recognised; "
            f"choose one of {', '.join(map(repr, TIME_AFTER_METHODS))}"
        )


def _stamp_origin_time(tr: Trace, ot: UTCDateTime) -> None:
    """Write the origin time onto a trace this module already owns.

    The private half of :func:`with_origin_time`: the public function copies
    first, and every internal caller is already working on its own copy, so
    copying again per trace would be a copy per trace of a stream that was
    copied whole one frame up.
    """
    tr.stats["otime"] = ot


def with_origin_time(tr: Trace, ot: UTCDateTime) -> Trace:
    """A copy of ``tr`` carrying ``ot`` as its origin time."""
    out = tr.copy()
    _stamp_origin_time(out, ot)
    return out


def with_distance(
    st: Stream,
    olat: float,
    olon: float,
    odep: float,
    ot: UTCDateTime,
    stlats: Sequence[float] | None = None,
    stlons: Sequence[float] | None = None,
    stelvs: Sequence[float] | None = None,
    inventory: Inventory | None = None,
    dtype: str = "sac",
) -> Stream:
    """A copy of ``st`` carrying the origin and source-receiver geometry.

    Every trace in the returned stream gains ``otime``, ``olat``, ``olon``,
    ``dep``, ``slat``, ``slon``, ``selv``, ``repi`` and ``rhyp``; the mseed
    path adds ``azimuth`` and ``back_azimuth``. ``st`` is not touched.

    ``dtype`` selects where the station coordinates come from:

    ``"sac"``
        the SAC header already on the trace.
    ``"mseed"``
        an ObsPy ``inventory``, which is then required.
    ``"list"``
        the ``stlats``, ``stlons`` and ``stelvs`` sequences, indexed
        positionally against the stream.

    Anything else raises. It used to print ``invalid method choice`` and carry
    on, leaving traces with an origin but no distance and deferring the failure
    to whatever first asked for ``repi``.
    """
    dtype = dtype.lower()
    if dtype not in STREAM_DISTANCE_METHODS:
        raise ValueError(
            f"dtype={dtype!r} is not recognised; "
            f"choose one of {', '.join(map(repr, STREAM_DISTANCE_METHODS))}"
        )
    if dtype == "mseed" and inventory is None:
        raise ValueError('dtype="mseed" requires an inventory')
    if dtype == "list" and any(x is None for x in (stlats, stlons, stelvs)):
        raise ValueError('dtype="list" requires stlats, stlons and stelvs')

    out = st.copy()
    for i, tr in enumerate(out):
        tr.stats["dep"] = odep
        tr.stats["olon"] = olon
        tr.stats["olat"] = olat
        _stamp_origin_time(tr, ot)

        if dtype == "sac":
            stlat, stlon, stelv = (
                tr.stats.sac.stla,
                tr.stats.sac.stlo,
                tr.stats.sac.stel,
            )
        elif dtype == "mseed":
            # Restating the guard above. Both guards raise `ValueError` with a
            # message naming the missing argument; these asserts carry that
            # conclusion the twenty lines to where it is used, which neither
            # `and`-guard is a form a type checker can do on its own.
            assert inventory is not None
            stlat, stlon, stelv = get_station_loc_from_inventory(tr, inventory)
        else:
            assert stlats is not None
            assert stlons is not None
            assert stelvs is not None
            stlat, stlon, stelv = stlats[i], stlons[i], stelvs[i]

        tr.stats["slat"] = stlat
        tr.stats["slon"] = stlon
        tr.stats["selv"] = stelv

        # Order matters to the last bit: `gps2dist_azimuth` is not symmetric in
        # floating point, and the mseed path has always called it
        # origin-first while the others called it station-first. Preserved, so
        # this consolidation does not move any published number.
        if dtype == "mseed":
            r, a, ba = obspy.geodetics.gps2dist_azimuth(olat, olon, stlat, stlon)
            tr.stats["azimuth"] = a
            tr.stats["back_azimuth"] = ba
        else:
            r = obspy.geodetics.gps2dist_azimuth(stlat, stlon, olat, olon)[0]

        tr.stats["repi"] = r / 1000
        tr.stats["rhyp"] = np.sqrt((odep + (stelv / 1000)) ** 2 + tr.stats["repi"] ** 2)

    return out


def get_station_loc_from_inventory(
    tr: Trace, inv: Inventory
) -> tuple[float, float, float]:
    meta = inv.get_channel_metadata(tr.id)
    return meta["latitude"], meta["longitude"], meta["elevation"]


def sensor_id(tr: Trace) -> str:
    """``NET.STA.LOC`` — the sensor a pick belongs to.

    Not the channel: an arrival is one sensor's observation and is shared by
    its components. Not the bare station either: a borehole and a surface
    instrument differ only by location code, and they do not see the same
    arrival.

    An empty location code is written ``--``, matching how Snuffler marker
    files spell it.
    """
    return ".".join([tr.stats.network, tr.stats.station, tr.stats.location or "--"])


def read_picks(
    source: str | PathLike[str], *, format: str | None = None
) -> dict[str, dict[str, Any]]:
    """Read picks as ``{"NET.STA.LOC": {"P": UTCDateTime, ...}}``.

    Accepts a Snuffler marker file or anything :func:`obspy.read_events`
    parses, so a caller holding a path from
    :meth:`~specmod.datasets.EventDirectory.picks_file` does not have to know
    which format it got.

    Keyed on each pick's own identity, so a format supplying no network code
    keys on ``*.STA.*`` and matches no trace. :func:`with_picks` resolves
    against the sensors present instead.
    """
    return pk.select_event(pk.read(source, format=format)).mapping()


def with_picks(
    st: Stream,
    source: str | PathLike[str],
    emergency_ratio: float = 1.7,
    *,
    format: str | None = None,
    event_id: str | None = None,
    on_ambiguous: pk.AmbiguousPolicy = "error",
    duplicates: pk.DuplicatePolicy = "prefer_reviewed",
    report: list[pk.Resolution] | None = None,
) -> Stream:
    """A copy of ``st`` with ``p_time`` and ``s_time`` on every picked trace.

    ``source`` is a Snuffler marker file, a registered plugin's format, or
    anything :func:`obspy.read_events` parses — QuakeML, SEISAN Nordic,
    HypoDD, NonLinLoc, a bulletin. See ``docs/pick-formats.md``.

    Picks are matched per sensor rather than per channel, so a pick made on one
    component reaches that sensor's others, and a pick stating only part of an
    identity matches on the fields it does state. ``on_ambiguous`` and
    ``duplicates`` are passed to :func:`specmod.picks.resolve`; ``event_id`` to
    :func:`specmod.picks.select_event`, and is required for a multi-event
    source.

    A trace whose P pick has no matching S gets one extrapolated at
    ``p + (p - otime) * emergency_ratio``, unless that would place S before P,
    in which case ``s_time`` is left unset and a warning is issued.

    Pass ``report=[]`` to receive the :class:`~specmod.picks.Resolution`.

    A trace with no pick is still present in the result, carrying no
    ``p_time`` — filtering those out is the caller's decision, and the
    pipeline's idiom for "unusable" is a trace without one.
    """
    out = st.copy()
    picks = pk.resolve(
        pk.select_event(pk.read(source, format=format), event_id=event_id),
        [pk.SensorID.parse(sensor_id(tr)) for tr in out],
        on_ambiguous=on_ambiguous,
        duplicates=duplicates,
    )
    if report is not None:
        report.append(picks)

    attached = {
        key: {phase: pick.time for phase, pick in phases.items()}
        for key, phases in picks.attached.items()
    }
    for tr in out:
        id = sensor_id(tr)
        try:
            tr.stats["p_time"] = attached[id]["P"]
        except KeyError:
            continue
        try:
            tr.stats["s_time"] = attached[id]["S"]
        except KeyError:
            sdiff = (tr.stats["p_time"] - tr.stats["otime"]) * emergency_ratio
            # Only meaningful when the origin precedes the pick. It is not
            # always so — a misconfigured origin time gives a negative `sdiff`
            # and an S arrival *before* P, from which every window cut is
            # nonsense. Leave `s_time` unset instead: the pipeline's own idiom
            # for "unusable" is a trace without one, and callers already filter
            # on that.
            if sdiff <= 0:
                warnings.warn(
                    f"{id}: no S pick, and the origin time is not before the P "
                    f"pick, so one cannot be extrapolated (would give S "
                    f"{-sdiff:.3f} s before P). Leaving s_time unset.",
                    stacklevel=2,
                )
                continue
            tr.stats["s_time"] = tr.stats["p_time"] + sdiff

    return out


def with_theoretical_picks(
    st: Stream,
    otime: UTCDateTime,
    p: float = 5.9,
    s: float = 2.9,
    dmetric: str = "repi",
) -> Stream:
    """A copy of ``st`` with P and S arrivals from average velocities [km/s].

    For use where there are no real picks. ``dmetric`` names the distance the
    travel time is computed over — ``"repi"`` or ``"rhyp"``, in kilometres —
    which :func:`with_distance` must have set already.
    """
    out = st.copy()
    for tr in out:
        rel_p = tr.stats[dmetric] / p
        rel_s = tr.stats[dmetric] / s
        tr.stats["p_time"] = otime + rel_p
        tr.stats["s_time"] = otime + rel_s
        tr.stats["otime"] = otime
    return out


def rstfl(fnames: Iterable[str], wild: str = "*", ext: str = "sac") -> Stream:
    """
    rstfl reads create an obspy stream by reading each trace from an arbitrary
    list of paths.
    """

    st = obspy.Stream([])
    for f in fnames:
        st += obspy.read(f"{f}{wild}.{ext.lower()}", format=ext.upper())

    return st


def _stamp_window(tr: Trace, start: UTCDateTime, end: UTCDateTime) -> None:
    """Record a window on a trace this module already owns.

    The private half of :func:`with_window`; see :func:`_stamp_origin_time`
    for why both exist.

    Two pairs, because they are not the same thing. ``trim`` gives back
    whatever the record actually holds, so a window that runs off either end
    comes back short — which is the normal case for noise windows, not the
    exception. Recording only the request is how a truncated noise trace ends
    up claiming a duration of data it does not have.

    ``wstart``/``wend`` are what the trace holds. ``wstart_requested``/
    ``wend_requested`` are what was asked for. Callers that want a window
    length should use the former; callers reporting on the cut want the latter.

    Must be called *after* the trim, or the two pairs are the same.
    """
    tr.stats["wstart_requested"] = start
    tr.stats["wend_requested"] = end
    tr.stats["wstart"] = tr.stats.starttime
    tr.stats["wend"] = tr.stats.endtime


def with_window(tr: Trace, start: UTCDateTime, end: UTCDateTime) -> Trace:
    """A copy of ``tr`` recording the window it holds, and the one asked for.

    For callers cutting their own windows rather than using
    :func:`p_window`, :func:`s_window` or :func:`coda_window`, which record
    this themselves. Must be called *after* the trim, or both pairs describe
    the request rather than the result — see :func:`_stamp_window`.
    """
    out = tr.copy()
    _stamp_window(out, start, end)
    return out


def get_sta_shift(sta: str, sta_shift: Mapping[str, float] | None) -> float:
    """The per-station timing correction for ``sta``, or zero.

    ``sta_shift`` maps station name to a shift in seconds, e.g. ``{"STA": 0.5}``.
    """
    if sta_shift is None:
        return 0.0
    return sta_shift.get(sta, 0.0)


def p_window(
    st: Stream,
    bf: float = 0,
    tafp: float = 0.8,
    time_after: str = "relative_time",
    sta_shift: Mapping[str, float] | None = None,
    refine_window: bool = False,
) -> Stream:
    """The P-wave window of every trace in ``st``, as a new stream.

    ``st`` is left as it was found, so the same stream can be windowed again
    for the noise — which is what :func:`get_noise_p` expects.

    ``bf``
        seconds before the P arrival to start the window.
    ``tafp``
        window length: seconds when ``time_after="absolute_time"``, otherwise
        a ratio of the P-S differential time.
    ``sta_shift``
        per-station timing corrections in seconds, e.g. ``{"STA": 0.5}``.
    ``refine_window``
        narrow the window to the 1st-99th percentile of cumulative squared
        amplitude — see :func:`signal_intensity`.

    Requires ``p_time`` and ``s_time``, which :func:`with_picks` or
    :func:`with_theoretical_picks` set.
    """

    _check_time_after(time_after)

    out = st.copy()
    for tr in out:
        stas = get_sta_shift(tr.stats.station, sta_shift)

        relps = tr.stats["s_time"] - tr.stats["p_time"]

        p_start = tr.stats["p_time"] - bf + stas

        if time_after == _ABSOLUTE:
            p_end = p_start + tafp
        else:
            p_end = p_start + tafp * relps

        if p_end > tr.stats["endtime"]:
            p_end = tr.stats["endtime"]

        tr.trim(p_start, p_end)

        if refine_window:
            rw_start, rw_end = signal_intensity(tr)

            p_end = p_start + rw_end
            p_start = p_start + rw_start

            tr.trim(p_start, p_end)

        _stamp_window(tr, p_start, p_end)

    return out


def s_window(
    st: Stream,
    rafp: float = 0.8,
    tafs: float = 20,
    time_after: str = "absolute_time",
    sta_shift: Mapping[str, float] | None = None,
    refine_window: bool = True,
) -> Stream:
    """The S-wave window of every trace in ``st``, as a new stream.

    ``st`` is left as it was found.

    ``rafp``
        ratio of the P-S differential time at which the window starts.
    ``tafs``
        window length: seconds when ``time_after="absolute_time"``, otherwise
        a ratio of the P-S differential time.
    ``sta_shift``
        per-station timing corrections in seconds.
    ``refine_window``
        narrow the window to the 1st-99th percentile of cumulative squared
        amplitude — see :func:`signal_intensity`. On by default here, off in
        :func:`p_window`.

    Requires ``p_time`` and ``s_time``. Modified by Pungky Suroyo.
    """
    _check_time_after(time_after)

    out = st.copy()
    for tr in out:
        stas = get_sta_shift(tr.stats.station, sta_shift)
        relps = tr.stats["s_time"] - tr.stats["p_time"]
        p_end = tr.stats["p_time"] + relps * rafp + stas

        if time_after == _ABSOLUTE:
            s_end = p_end + tafs
        else:
            s_end = p_end + tafs * relps

        if s_end > tr.stats["endtime"]:
            s_end = tr.stats["endtime"]

        tr.trim(p_end, s_end)

        if refine_window:
            rw_start, rw_end = signal_intensity(tr)

            s_end = p_end + rw_end
            p_end = p_end + rw_start

            tr.trim(p_end, s_end)

        _stamp_window(tr, p_end, s_end)

    return out


def signal_intensity(
    tr: Trace, pctls: Sequence[float] = (1, 99), plot: bool = False
) -> tuple[float, float]:
    delta = tr.stats.delta
    data = tr.data

    inte = normalise(cumulative_trapezoid(data**2)) * 100

    w_start = np.abs(inte - pctls[0]).argmin() * delta
    w_end = np.abs(inte - pctls[1]).argmin() * delta

    if plot:
        plt.plot(np.arange(0, len(data)) * delta, normalise(data) * 100, color="grey")
        plt.plot((np.arange(0, len(data)) * delta)[:-1], inte, "k--")
        plt.vlines(w_start, 0, 100, color="red")
        plt.vlines(w_end, 0, 100, color="red")
        plt.xlim(0, w_end * 1.1)

    return w_start, w_end


def padded(st: Stream, pad_len: float = 1, pad_val: float = 0) -> Stream:
    """A copy of ``st`` with ``pad_len`` seconds of ``pad_val`` at each end."""

    out = st.copy()
    for tr in out:
        tr.trim(
            tr.stats.starttime - pad_len,
            tr.stats.endtime + pad_len,
            pad=True,
            fill_value=pad_val,
        )
    return out


def coda_window(
    st: Stream,
    bf: float = 2,
    raf: float = 0.8,
    tafp: float = 1.4,
    sta_shift: Mapping[str, float] | None = None,
) -> Stream:
    """The coda window of every trace in ``st``, as a new stream.

    Runs from ``tafp`` P-S differential times after the S window opens to the
    end of the record. ``st`` is left as it was found. Written by Pungky
    Suroyo.
    """

    out = st.copy()
    for tr in out:
        stas = get_sta_shift(tr.stats.station, sta_shift)

        relps = tr.stats["s_time"] - tr.stats["p_time"]

        s_start = tr.stats["p_time"] + relps * raf + stas

        # Was `tafp * relps + s_start`, i.e. `float + UTCDateTime`.
        # `UTCDateTime` defines no `__radd__`, so this raised `TypeError` for
        # every input the function was ever given.
        c_start = s_start + tafp * relps

        c_end = tr.stats["endtime"]

        tr.trim(c_start, c_end)

        _stamp_window(tr, c_start, c_end)

    return out


def normalise(x: NDArray[np.floating[Any]], space: Sequence[float] = (0, 1)) -> Any:
    return np.interp(x, [x.min(), x.max()], space)


def get_noise_p(st: Stream, sig: Stream, bshift: float = 0.2) -> Stream:
    """A noise window per trace, ending ``bshift`` s before the P arrival.

    Each window is as long as the matching trace in ``sig``, so signal and
    noise are compared over the same duration. ``st`` should be the stream the
    signal was cut *from*, not the cut signal itself.
    """
    stc = st.copy()
    # `strict=True`: the two streams are paired by position, so a signal
    # stream of a different length is not a shorter result but a wrong one.
    # It used to be `strict=False`, which left the unpaired traces in the
    # returned stream whole and unlinked — full records presented as noise.
    for tr, trs in zip(stc, sig, strict=True):
        end = tr.stats["p_time"] - bshift

        start = end - (trs.stats["wend"] - trs.stats["wstart"])

        tr.trim(start, end)

        _stamp_window(tr, start, end)
    return stc


def get_noise_s(
    st: Stream, bf: float = 1, bshift: float = 0.2, sig: Stream | None = None
) -> Stream:
    stc = st.copy()
    for i, tr in enumerate(stc):
        end = tr.stats["p_time"] - bshift

        if sig is not None:  # get the same length as the signal window
            start = end - (sig[i].stats["wend"] - sig[i].stats["wstart"])
        else:
            start = end - bf
        tr.trim(start, end)
        _stamp_window(tr, start, end)
    return stc
