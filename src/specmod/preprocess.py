"""Geometry, picks and window cutting, on ObsPy streams.

This module and :mod:`specmod.pipeline` are the only two that know what a
``Trace`` is. ObsPy ships no type information and no stub package is
published, so ``stubs/obspy`` in this repository declares the surface used
here; see ``stubs/README.md``. ``Stats`` is deliberately open, so
``tr.stats.delta`` is checked and ``tr.stats["p_time"]`` — one of the fields
this module sets — is not.
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
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from os import PathLike

    from numpy.typing import NDArray
    from obspy import Inventory, Stream, Trace, UTCDateTime

#: How station coordinates are supplied to :func:`set_stream_distance`.
#: ``"none"`` is a deprecated alias for ``"list"``, kept because it is the
#: spelling the function used to test for internally — see the note there.
STREAM_DISTANCE_METHODS = ["mseed", "sac", "list", "none"]

#: Accepted values of ``time_after``. Both spellings of the relative mode are
#: taken by both cutting functions: they used to disagree, ``cut_p`` calling it
#: ``"relative_time"`` and ``cut_s`` ``"relative_ps"``, with neither accepting
#: the other's word for the same idea.
_ABSOLUTE = "absolute_time"
_RELATIVE = ("relative_time", "relative_ps")
TIME_AFTER_METHODS = (_ABSOLUTE, *_RELATIVE)


def _check_time_after(time_after: str) -> None:
    if time_after not in TIME_AFTER_METHODS:
        raise ValueError(
            f"time_after={time_after!r} is not recognised; "
            f"choose one of {', '.join(map(repr, TIME_AFTER_METHODS))}"
        )


def set_origin_time(tr: Trace, ot: UTCDateTime) -> None:
    tr.stats["otime"] = ot


def set_stream_distance(
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
) -> None:
    """
    Set the origin and source-receiver geometry on every trace in a stream.

    ``dtype`` selects where the station coordinates come from:

    ``"sac"``
        the SAC header already on the trace.
    ``"mseed"``
        an ObsPy ``inventory``, which is then required.
    ``"list"``
        the ``stlats``, ``stlons`` and ``stelvs`` sequences, indexed
        positionally against the stream. ``"none"`` is a deprecated alias.

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
    if dtype == "none":
        warnings.warn(
            'dtype="none" is a deprecated alias for dtype="list"',
            DeprecationWarning,
            stacklevel=2,
        )
        dtype = "list"
    if dtype == "mseed" and inventory is None:
        raise ValueError('dtype="mseed" requires an inventory')
    if dtype == "list" and any(x is None for x in (stlats, stlons, stelvs)):
        raise ValueError('dtype="list" requires stlats, stlons and stelvs')

    for i, tr in enumerate(st):
        tr.stats["dep"] = odep
        tr.stats["olon"] = olon
        tr.stats["olat"] = olat
        set_origin_time(tr, ot)

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


def read_picks(source: str | PathLike[str]) -> dict[str, dict[str, Any]]:
    """Read picks as ``{"NET.STA.LOC": {"P": UTCDateTime, ...}}``.

    Accepts a Snuffler marker file or anything :func:`obspy.read_events`
    parses, so a caller holding a path from
    :meth:`~specmod.datasets.EventDirectory.picks_file` does not have to know
    which format it got.

    Keyed on each pick's own identity, so a format supplying no network code
    keys on ``*.STA.*`` and matches no trace. :func:`set_picks` resolves
    against the sensors present instead.
    """
    return pk.select_event(pk.read(source)).mapping()


def set_picks(
    st: Stream,
    source: str | PathLike[str],
    emergency_ratio: float = 1.7,
    *,
    event_id: str | None = None,
    on_ambiguous: pk.AmbiguousPolicy = "error",
    duplicates: pk.DuplicatePolicy = "prefer_reviewed",
    report: list[pk.Resolution] | None = None,
) -> None:
    """Attach ``p_time`` and ``s_time`` to every trace with a pick.

    ``source`` is a Snuffler marker file or anything :func:`obspy.read_events`
    parses — QuakeML, SC3ML, SEISAN Nordic, NonLinLoc, HypoDD, a bulletin.

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
    """
    picks = pk.resolve(
        pk.select_event(pk.read(source), event_id=event_id),
        [pk.SensorID.parse(sensor_id(tr)) for tr in st],
        on_ambiguous=on_ambiguous,
        duplicates=duplicates,
    )
    if report is not None:
        report.append(picks)

    attached = {
        key: {phase: pick.time for phase, pick in phases.items()}
        for key, phases in picks.attached.items()
    }
    for tr in st:
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


def set_picks_from_pyrocko(
    st: Stream, pyrock_file: str | PathLike[str], emergency_ratio: float = 1.7
) -> None:
    """Deprecated alias for :func:`set_picks`.

    Renamed because it no longer reads only Pyrocko: QuakeML is now the
    preferred format and the old name says the opposite of what the function
    does.
    """
    warnings.warn(
        "set_picks_from_pyrocko is deprecated; use set_picks, which reads "
        "QuakeML as well as Snuffler marker files.",
        DeprecationWarning,
        stacklevel=2,
    )
    set_picks(st, pyrock_file, emergency_ratio)


def basic_set_theoreticals(
    st: Stream,
    otime: UTCDateTime,
    p: float = 5.9,
    s: float = 2.9,
    dmetric: str = "repi",
) -> None:
    """
    basic_set_theoreticals uses average propagation velocities [km/s]
    to set the arrival times for P and S waves. This assumes epicentral and/or
    and hypocentral distances have already been calculated and are set in the
    trace stats dictionary as tr.stats['repi'] or tr.stats['rhyp'] in units of
    kilometres.
    """
    for tr in st:
        rel_p = tr.stats[dmetric] / p
        rel_s = tr.stats[dmetric] / s
        tr.stats["p_time"] = otime + rel_p
        tr.stats["s_time"] = otime + rel_s
        tr.stats["otime"] = otime


def rstfl(fnames: Iterable[str], wild: str = "*", ext: str = "sac") -> Stream:
    """
    rstfl reads create an obspy stream by reading each trace from an arbitrary
    list of paths.
    """

    st = obspy.Stream([])
    for f in fnames:
        st += obspy.read(f"{f}{wild}.{ext.lower()}", format=ext.upper())

    return st


def link_window_to_trace(tr: Trace, start: UTCDateTime, end: UTCDateTime) -> None:
    """Record a window on a trace, as asked for *and* as delivered.

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


def get_sta_shift(sta: str, sta_shift: Mapping[str, float] | None) -> float:
    """The per-station timing correction for ``sta``, or zero.

    ``sta_shift`` maps station name to a shift in seconds, e.g. ``{"STA": 0.5}``.
    """
    if sta_shift is None:
        return 0.0
    return sta_shift.get(sta, 0.0)


def cut_p(
    st: Stream,
    bf: float = 0,
    tafp: float = 0.8,
    time_after: str = "relative_time",
    sta_shift: Mapping[str, float] | None = None,
    refine_window: bool = False,
) -> None:
    """
    Function to cut a p wave window from an Obspy trace obeject

    bf (int/float) time shift in seconds before the P-wave arrival time

    raf (int/float) ratio of p-s time to fix the end of the P-window

    sta_shift (dict) dictionary of station names and station specific time shifts in seconds

    refine_window (bool) True if you want to use squared intergral percentiles to refine
    the signal window.
    """

    _check_time_after(time_after)

    for tr in st:
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

        link_window_to_trace(tr, p_start, p_end)


def cut_s(
    st: Stream,
    rafp: float = 0.8,
    tafs: float = 20,
    time_after: str = "absolute_time",
    sta_shift: Mapping[str, float] | None = None,
    refine_window: bool = True,
) -> None:
    """
    Function to cut a s wave window from an Obspy trace obeject.

    bf (int/float) time shift in seconds before the P-wave arrival time

    rafp (int/float) ratio of p-s time to fix the start the of S-window

    tafs (int/float) window length in seconds or scaling factor of relative p-s time

    time_after (str) can be set to 'absolute_time' or 'relative_ps'

        if time_after == 'absolute_time' the window length is given as a value in seconds

        if time_after == 'relative_ps' the value should be some number that scales with the p-s differential time

    sta_shift (dict) dictionary of station names and station specific time shifts in seconds

    Modified by Pungky Suroyo.
    """
    _check_time_after(time_after)

    for tr in st:
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

        link_window_to_trace(tr, p_end, s_end)


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


def pad_traces(st: Stream, pad_len: float = 1, pad_val: float = 0) -> None:
    """
    Util to pad waveforms with zeros before and after the start and endtime of trace.
    """

    for tr in st:
        tr.trim(
            tr.stats.starttime - pad_len,
            tr.stats.endtime + pad_len,
            pad=True,
            fill_value=pad_val,
        )


def cut_c(
    st: Stream,
    bf: float = 2,
    raf: float = 0.8,
    tafp: float = 1.4,
    sta_shift: Mapping[str, float] | None = None,
) -> None:
    """

    Function to cut a coda wave window from an Obspy trace object

    Written by Pungky Suroyo.

    """

    for tr in st:
        stas = get_sta_shift(tr.stats.station, sta_shift)

        relps = tr.stats["s_time"] - tr.stats["p_time"]

        s_start = tr.stats["p_time"] + relps * raf + stas

        # Was `tafp * relps + s_start`, i.e. `float + UTCDateTime`.
        # `UTCDateTime` defines no `__radd__`, so this raised `TypeError` for
        # every input the function was ever given.
        c_start = s_start + tafp * relps

        c_end = tr.stats["endtime"]

        tr.trim(c_start, c_end)

        link_window_to_trace(tr, c_start, c_end)


def normalise(x: NDArray[np.floating[Any]], space: Sequence[float] = (0, 1)) -> Any:
    return np.interp(x, [x.min(), x.max()], space)


def get_signal(st: Stream, func: Callable[..., None], **kwargs: Any) -> Stream:
    stc = st.copy()
    func(stc, **kwargs)
    return stc


def get_noise_p(st: Stream, sig: Stream, bshift: float = 0.2) -> Stream:
    stc = st.copy()
    # `strict=True`: the two streams are paired by position, so a signal
    # stream of a different length is not a shorter result but a wrong one.
    # It used to be `strict=False`, which left the unpaired traces in the
    # returned stream whole and unlinked — full records presented as noise.
    for tr, trs in zip(stc, sig, strict=True):
        end = tr.stats["p_time"] - bshift

        start = end - (trs.stats["wend"] - trs.stats["wstart"])

        tr.trim(start, end)

        link_window_to_trace(tr, start, end)
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
        link_window_to_trace(tr, start, end)
    return stc
