"""Reading picks and catalogues, and looking at waveforms.

What is left after the migration. The rotation machinery that used to live here
— ``find_rotation_angle``, ``find_rotation_angle_v2``, ``rotate``,
``rotate_noise_full``, ``get_centroid_freq`` and ``non_lin_boost_noise_func`` —
is now :mod:`specmod.core.noise`, where the two methods are registered models
rather than an integer flag. The SAC-discovery scaffolding that used to sit
between them (``DataSet`` and its channel-ranking helpers, ``cps``,
``path_to_utc``) had no callers anywhere and was hardcoded to another study's
station codes; it is replaced by the acquisition layer in REFACTOR_PLAN §5.2.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from matplotlib.dates import num2date
from obspy.core.event import Catalog, Event, WaveformStreamID
from obspy.core.event import Pick as ObsPyPick

from specmod.picks import PickSet, SnufflerReader, from_catalog, select_event

#: See the note in `specmod.fitting.event`: diagnostics go here, and nothing in
#: this package configures logging on a caller's behalf.
logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence
    from datetime import datetime
    from os import PathLike

    from obspy import Stream, UTCDateTime

# Type 42 embeds fonts as editable text rather than outlines, so a figure can
# be relabelled in a vector editor after the fact. Set at import because it is
# a global, and this module is the one that draws.
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


def _to_datetime(value: Any) -> Any:
    """``matplotlib.dates.num2date`` carries no annotations upstream.

    Wrapped once rather than suppressed at each of its four call sites, so
    there is one place to delete when matplotlib annotates it.
    """
    return num2date(value)  # type: ignore[no-untyped-call]


def read_pyrocko_picks(path: str | PathLike[str]) -> PickSet:
    """Snuffler marker file to a :class:`~specmod.picks.PickSet`.

    Thin wrapper over :class:`specmod.picks.SnufflerReader`, which is where the
    format lives.
    """
    return SnufflerReader().read(path)[0]


def read_pyrocko(path: str | PathLike[str]) -> dict[str, dict[str, UTCDateTime]]:
    """Snuffler marker file to ``{"NET.STA.LOC": {"P": UTCDateTime, ...}}``.

    Keyed **per sensor**, not per channel. An arrival is an observation of one
    sensor, and the component it was picked on is incidental — on the shipped
    PNR file every station has P on ``HHZ`` and S on ``HHN``, so keying by full
    SEED id would leave the horizontals with no S at all.

    The location code stays in the key, because that is what distinguishes two
    sensors at one site. A surface and a borehole instrument share a station
    name and see genuinely different arrivals; collapsing them would silently
    give one the other's picks.

    A flat mapping cannot express a partial identity or a second event; prefer
    :func:`read_pyrocko_picks` with :func:`specmod.picks.resolve`.
    """
    return read_pyrocko_picks(path).mapping()


def read_quakeml_picks(
    source: str | PathLike[str] | Catalog,
) -> dict[str, dict[str, UTCDateTime]]:
    """QuakeML picks to ``{"NET.STA.LOC": {"P": UTCDateTime, ...}}``.

    The same shape :func:`read_pyrocko` returns, so the two are
    interchangeable at the call site. ``source`` is a path or an
    ``obspy.core.event.Catalog``.

    Phase hints are normalised to ``P`` or ``S`` on their first letter, which
    folds ``Pg``/``Pn``/``Pb`` and their S counterparts together. That matches
    what this pipeline does with them — it wants a direct arrival, and the
    branch distinction is not something the windowing uses.

    Picks with an ``evaluation_status`` of ``rejected`` are dropped. Anything
    else — reviewed, preliminary, unset — is kept, matching how the Snuffler
    reader treats weights.

    A multi-event source raises rather than merging. A flat mapping cannot
    express a partial identity or a second event; prefer
    :func:`specmod.picks.read` with :func:`specmod.picks.resolve`.
    """
    catalog = source if isinstance(source, Catalog) else obspy.read_events(str(source))
    return select_event(from_catalog(catalog)).mapping()


def picks_to_quakeml(
    picks: dict[str, dict[str, UTCDateTime]],
    *,
    event_id: str | None = None,
) -> Catalog:
    """The inverse: a pick mapping as an ``obspy`` ``Catalog``.

    Used to convert a Snuffler marker file to the standard format once, rather
    than teaching every consumer both. The location code ``--`` is written back
    as an empty string, which is what StationXML and miniSEED use.
    """
    event = Event(resource_id=event_id) if event_id else Event()
    for key, phases in sorted(picks.items()):
        net, sta, loc = key.split(".")
        for phase, time in sorted(phases.items()):
            event.picks.append(
                ObsPyPick(
                    time=time,
                    phase_hint=phase,
                    waveform_id=WaveformStreamID(
                        network_code=net,
                        station_code=sta,
                        location_code="" if loc == "--" else loc,
                    ),
                )
            )
    return Catalog(events=[event])


def plot_traces(
    st: Stream,
    *,
    plot_theoreticals: bool = False,
    plot_windows: bool = False,
    conv: float | None = 1e-9,
    bft: float = 1,
    aftt: float = 60,
    sig: Stream | None = None,
    noise: Stream | None = None,
    save: str | None = None,
) -> None:
    # `plot_windows` reads `sig[i]` and `noise[i]`. Without both it used to
    # raise `TypeError: 'NoneType' object is not subscriptable` from inside the
    # loop, naming neither the argument that was missing nor the option that
    # required it.
    if plot_windows and (sig is None or noise is None):
        raise ValueError("plot_windows=True requires both sig and noise")

    sharey = False
    stc = stream_distance_sort(st)
    if conv is None:
        conv = 1
        stc.normalize()
        sharey = True

    if sig is not None and noise is not None:
        sig = stream_distance_sort(sig)
        noise = stream_distance_sort(noise)

    fig, drawn = plt.subplots(
        len(stc), 1, sharex=True, sharey=sharey, figsize=(14, len(stc) * 3)
    )
    # `subplots` returns a bare `Axes` for a single row and an array otherwise.
    # Kept as a separate name from what `subplots` returned: rebinding one
    # variable to both shapes is what made the function unannotatable.
    ax: Sequence[Any] = drawn.flatten() if len(stc) > 1 else [drawn]

    for i, tr in enumerate(stc):
        if plot_windows:
            assert sig is not None  # both guarded above
            assert noise is not None
            # get window start and end times
            sts = sig[i].stats["wstart"], sig[i].stats["wend"]
            nts = noise[i].stats["wstart"], noise[i].stats["wend"]

            tr.trim(nts[0] - bft, sts[1] + aftt)

            for signal_edge, noise_edge in zip(sts, nts, strict=True):
                ts: datetime = _to_datetime(signal_edge.matplotlib_date)
                tn: datetime = _to_datetime(noise_edge.matplotlib_date)
                ax[i].vlines(ts, tr.data.min() * conv, tr.data.max() * conv, color="k")
                ax[i].vlines(
                    tn, tr.data.min() * conv, tr.data.max() * conv, color="blue"
                )

        if plot_theoreticals:
            try:
                if not plot_windows:
                    # Trim to a window around the P arrival, on the assumption
                    # that the pick is trustworthy.
                    tr.trim(tr.stats["p_time"] - bft, tr.stats["p_time"] + aftt)

                p = _to_datetime(tr.stats["p_time"].matplotlib_date)
                s = _to_datetime(tr.stats["s_time"].matplotlib_date)
                ax[i].vlines(
                    p,
                    tr.data.min() * conv / 1.5,
                    tr.data.max() * conv / 1.5,
                    linestyles="dashed",
                    color="blue",
                    label="Pg",
                )
                ax[i].vlines(
                    s,
                    tr.data.min() * conv / 1.5,
                    tr.data.max() * conv / 1.5,
                    linestyles="dashed",
                    color="red",
                    label="Sg",
                )
            except KeyError:
                pass

        # `sample` rather than `i`: the comprehension used to reuse the name of
        # the enclosing loop variable. Comprehensions have their own scope, so
        # it was never a bug — but it reads like one, and would become one the
        # moment this is turned back into a statement.
        time = _to_datetime(
            np.array(
                [
                    (
                        tr.stats.starttime + (tr.stats.delta * (sample + 1))
                    ).matplotlib_date
                    for sample in range(len(tr.data))
                ]
            )
        )

        ax[i].plot(time, tr.data * conv, color="grey", label=tr.id, zorder=1)
        # A trace with no geometry set still plots; it just goes untitled.
        with contextlib.suppress(KeyError):
            ax[i].set_title(
                f"Repi: {tr.stats['repi']:.2f} km, Rhyp: {tr.stats['rhyp']:.2f} km"
            )
        ax[i].legend()
    ax[-1].set_xlabel("Time (UTC)")
    fig.suptitle(str(st[0].stats.otime))
    fig.tight_layout()
    if save is not None:
        assert type(save) is str
        fig.savefig(save)
        fig.clear()
        plt.close(fig)
        logger.debug("closed the time-domain figure after saving to %s", save)


def stream_distance_sort(st: Stream, dist_met: str = "repi") -> Stream:
    """A copy of ``st`` ordered by distance. Not in place.

    Returns the stream unsorted, with a warning, when the traces carry no
    distance — which is why the copy is taken on the way out rather than only
    on the sorted path.
    """
    try:
        st = obspy.Stream(sorted(st, key=lambda x: x.stats[dist_met]))
    except KeyError:
        warnings.warn(
            f"No {dist_met!r} on these traces, so the stream is returned "
            "unsorted. Set distances with `preprocess.set_stream_distance`.",
            stacklevel=2,
        )

    return st.copy()


def read_cat(path: str | PathLike[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+")


def cat2kstyle(row: Any) -> str:
    """Catalogue date and time as dot-separated whole-number fields.

    Sub-second precision is dropped, because :func:`keith2utc` parses every
    field with :func:`int`. It used to be dropped with a fixed ``[:-3]`` slice,
    which silently assumed exactly two decimal places on the seconds:

    * ``"13:09:31.00"`` worked,
    * ``"13:09:31.000"`` left a trailing separator and made ``keith2utc``
      raise ``invalid literal for int()``,
    * ``"13:09:31"`` — no decimals at all — lost the **seconds**, quietly
      returning 13:09 as though it were 13:09:31.

    The last is the one worth fixing: it is a wrong answer rather than an
    error, and a catalogue written without fractional seconds is not unusual.
    Splitting on the decimal point handles all three.
    """
    date = row["Date"].replace("/", ".")
    time = row["Time"].split(".")[0].replace(":", ".")
    return ".".join([date, time])


def keith2utc(row: Any) -> UTCDateTime:
    return obspy.UTCDateTime(*map(int, cat2kstyle(row).split(".")))
