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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from matplotlib.dates import num2date

# Type 42 embeds fonts as editable text rather than outlines, so a figure can
# be relabelled in a vector editor after the fact. Set at import because it is
# a global, and this module is the one that draws.
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


def read_pyrocko(path):

    pyrocko_map = {
        "^": ("P", "Pg", "u", "i"),
        "v": ("P", "Pg", "d", "i"),
        "P": ("P", "Pg", None, "e"),
        "S": ("S", "Sg", None, "e"),
    }

    with open(path) as f:
        # open file and store in memory as whole list. Files typically
        # aren't too large to read in at once without buffering.
        f = f.readlines()[1:]
    # Loop through the list and extract / unpack the metadata

    stations = {}
    for line in f:
        l = line.split()  # split between whitespace
        tid = l[4].replace("..", ".--.").split(".")  # ensure locs are the same
        net = tid[0]
        name = tid[1]
        time = "T".join(l[1:3])
        des, _pt, _fm, _po = pyrocko_map[l[8]]
        weight = int(l[3])
        ID = ".".join((net, name))
        if weight <= 3:
            try:
                stations[ID].update({des: obspy.UTCDateTime(time)})
            except KeyError:
                stations.update({ID: {des: obspy.UTCDateTime(time)}})

    return stations


def plot_traces(
    st,
    plot_theoreticals=False,
    plot_windows=False,
    conv=1e-9,
    bft=1,
    aftt=60,
    sig=None,
    noise=None,
    save=None,
):

    sharey = False
    stc = stream_distance_sort(st)
    if conv is None:
        conv = 1
        stc.normalize()
        sharey = True

    if sig is not None and noise is not None:
        sig = stream_distance_sort(sig)
        noise = stream_distance_sort(noise)

    fig, ax = plt.subplots(
        len(stc), 1, sharex=True, sharey=sharey, figsize=(14, len(stc) * 3)
    )
    if len(stc) > 1:
        ax = ax.flatten()
    else:
        ax = [ax]
    for i, tr in enumerate(stc):
        if plot_windows:
            # get window start and end times
            sts = sig[i].stats["wstart"], sig[i].stats["wend"]
            nts = noise[i].stats["wstart"], noise[i].stats["wend"]

            tr.trim(nts[0] - bft, sts[1] + aftt)

            for ts, tn in zip(sts, nts):
                ts, tn = num2date(ts.matplotlib_date), num2date(tn.matplotlib_date)
                ax[i].vlines(ts, tr.data.min() * conv, tr.data.max() * conv, color="k")
                ax[i].vlines(
                    tn, tr.data.min() * conv, tr.data.max() * conv, color="blue"
                )

        if plot_theoreticals:
            try:
                if not plot_windows:
                    # we should trim it down to some time before and after the p arrival (assuming we trust it)
                    tr.trim(tr.stats["p_time"] - bft, tr.stats["p_time"] + aftt)

                p = num2date(tr.stats["p_time"].matplotlib_date)
                s = num2date(tr.stats["s_time"].matplotlib_date)
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

        time = num2date(
            np.array(
                [
                    (tr.stats.starttime + (tr.stats.delta * (i + 1))).matplotlib_date
                    for i in range(len(tr.data))
                ]
            )
        )

        ax[i].plot(time, tr.data * conv, color="grey", label=tr.id, zorder=1)
        try:
            ax[i].set_title(
                "Repi: {:.2f} km, Rhyp: {:.2f} km".format(
                    tr.stats["repi"], tr.stats["rhyp"]
                )
            )
        except KeyError:
            pass
        ax[i].legend()
    ax[-1].set_xlabel("Time (UTC)")
    fig.suptitle(str(st[0].stats.otime))
    fig.tight_layout()
    if save is not None:
        assert type(save) is str
        fig.savefig(save)
        fig.clear()
        plt.close(fig)
        print("deleted td fig")


def stream_distance_sort(st, dist_met="repi"):
    """
    Sorted makes a copy so you must save it back to rhe stream
    then force the user to save a new copy. NOT INPLACE!!!
    """
    try:
        st = obspy.Stream(sorted(st, key=lambda x: x.stats[dist_met]))
    except KeyError:
        print("WARNING: No distance info, stream not sorted by distance.")

    return st.copy()


def read_cat(path):
    return pd.read_csv(path, sep=r"\s+")


def cat2kstyle(row):
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


def keith2utc(row):
    return obspy.UTCDateTime(*map(int, cat2kstyle(row).split(".")))
