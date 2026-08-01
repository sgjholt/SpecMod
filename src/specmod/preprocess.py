import matplotlib.pyplot as plt
import numpy as np
import obspy
from scipy.integrate import cumulative_trapezoid

from . import utils as ut

STREAM_DISTANCE_METHODS = ["mseed", "sac", "list"]


def set_origin_time(tr, ot):
    tr.stats["otime"] = ot


def set_stream_distance(
    st,
    olat,
    olon,
    odep,
    ot,
    stlats=None,
    stlons=None,
    stelvs=None,
    inventory=None,
    dtype="sac",
):
    """
    This assumes the sac files already have the station lat and long (degrees)
    and elevation in the sac header.
    """

    global STREAM_DISTANCE_METHODS

    if dtype.lower() in STREAM_DISTANCE_METHODS:
        for i, tr in enumerate(st):
            tr.stats["dep"] = odep
            tr.stats["olon"] = olon
            tr.stats["olat"] = olat
            set_origin_time(tr, ot)
            if dtype.lower() == "sac":
                tr.stats["repi"] = (
                    obspy.geodetics.gps2dist_azimuth(
                        tr.stats.sac.stla, tr.stats.sac.stlo, olat, olon
                    )[0]
                    / 1000
                )
                tr.stats["rhyp"] = np.sqrt(
                    (odep + (tr.stats.sac.stel / 1000)) ** 2 + tr.stats["repi"] ** 2
                )
                tr.stats["slon"] = tr.stats.sac.stlo
                tr.stats["slat"] = tr.stats.sac.stla
                tr.stats["selv"] = tr.stats.sac.stel
            elif dtype.lower() == "mseed":
                stlat, stlon, stelv = get_station_loc_from_inventory(tr, inventory)
                tr.stats["slon"] = stlon
                tr.stats["slat"] = stlat
                tr.stats["selv"] = stelv
                r, a, ba = obspy.geodetics.gps2dist_azimuth(olat, olon, stlat, stlon)
                tr.stats["back_azimuth"] = ba
                tr.stats["azimuth"] = a
                tr.stats["repi"] = r / 1000
                tr.stats["rhyp"] = np.sqrt(
                    (odep + (stelv / 1000)) ** 2 + tr.stats["repi"] ** 2
                )

            elif dtype.lower() == "none":
                stlat, stlon, stelv = stlats[i], stlons[i], stelvs[i]
                tr.stats["slon"] = stlon
                tr.stats["slat"] = stlat
                tr.stats["selv"] = stelv
                tr.stats["repi"] = (
                    obspy.geodetics.gps2dist_azimuth(stlat, stlon, olat, olon)[0] / 1000
                )
                tr.stats["rhyp"] = np.sqrt(
                    (odep + (stelv / 1000)) ** 2 + tr.stats["repi"] ** 2
                )

            else:
                print("invalid method choice")


def get_station_loc_from_inventory(tr, inv):
    meta = inv.get_channel_metadata(tr.id)
    return meta["latitude"], meta["longitude"], meta["elevation"]


def set_picks_from_pyrocko(st, pyrock_file, emergency_ratio=1.7):
    picks = ut.read_pyrocko(pyrock_file)
    for tr in st:
        id = ".".join([tr.stats.network, tr.stats.station])
        try:
            tr.stats["p_time"] = picks[id]["P"]
        except KeyError:
            continue
        try:
            tr.stats["s_time"] = picks[id]["S"]
        except KeyError:
            sdiff = (tr.stats["p_time"] - tr.stats["otime"]) * emergency_ratio
            tr.stats["s_time"] = tr.stats["p_time"] + sdiff


def basic_set_theoreticals(st, otime, p=5.9, s=2.9, dmetric="repi"):
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


def rstfl(fnames, wild="*", ext="sac"):
    """
    rstfl reads create an obspy stream by reading each trace from an arbitrary
    list of paths.
    """

    st = obspy.Stream([])
    for f in fnames:
        st += obspy.read(f"{f}{wild}.{ext.lower()}", format=ext.upper())

    return st


def link_window_to_trace(tr, start, end):
    tr.stats["wstart"] = start
    tr.stats["wend"] = end


def get_sta_shift(sta, sta_shift):
    """
    sta_shift must be a dictionary containing the station name to be shifted
    and the time shift in seconds e.g. {'STA':0.5}.
    """
    if sta in sta_shift.keys():
        return sta_shift[sta]
    else:
        return 0


def cut_p(
    st, bf=0, tafp=0.8, time_after="relative_time", sta_shift={}, refine_window=False
):
    """
    Function to cut a p wave window from an Obspy trace obeject

    bf (int/float) time shift in seconds before the P-wave arrival time

    raf (int/float) ratio of p-s time to fix the end of the P-window

    sta_shift (dict) dictionary of station names and station specific time shifts in seconds

    refine_window (bool) True if you want to use squared intergral percentiles to refine
    the signal window.
    """

    stas = 0

    for tr in st:
        stas = get_sta_shift(tr.stats.station, sta_shift)

        relps = tr.stats["s_time"] - tr.stats["p_time"]

        p_start = tr.stats["p_time"] - bf + stas

        if time_after == "absolute_time":
            p_end = p_start + tafp

        if time_after == "relative_time":
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
    st, rafp=0.8, tafs=20, time_after="absolute_time", sta_shift={}, refine_window=True
):
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
    stas = 0

    for tr in st:
        stas = get_sta_shift(tr.stats.station, sta_shift)
        relps = tr.stats["s_time"] - tr.stats["p_time"]
        p_end = tr.stats["p_time"] + relps * rafp + stas

        if time_after == "absolute_time":
            s_end = p_end + tafs

        if time_after == "relative_ps":
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


def signal_intensity(tr, pctls=[1, 99], plot=False):
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


def pad_traces(st, pad_len=1, pad_val=0):
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


def cut_c(st, bf=2, raf=0.8, tafp=1.4, sta_shift={}):
    """

    Function to cut a coda wave window from an Obspy trace object

    Written by Pungky Suroyo.

    """

    stas = 0

    for tr in st:
        stas = get_sta_shift(tr.stats.station, sta_shift)

        relps = tr.stats["s_time"] - tr.stats["p_time"]

        s_start = tr.stats["p_time"] + relps * raf + stas

        c_start = tafp * relps + s_start

        c_end = tr.stats["endtime"]

        link_window_to_trace(tr, c_start, c_end)

        tr.trim(c_start, c_end)


def normalise(x, space=[0, 1]):

    return np.interp(x, [x.min(), x.max()], space)


def get_signal(st, func, **kwargs):
    stc = st.copy()
    func(stc, **kwargs)
    return stc


def get_noise_p(st, sig, bshift=0.2):
    stc = st.copy()
    for tr, trs in zip(stc, sig, strict=False):
        end = tr.stats["p_time"] - bshift

        start = end - (trs.stats["wend"] - trs.stats["wstart"])

        link_window_to_trace(tr, start, end)

        tr.trim(start, end)
    return stc


def get_noise_s(st, bf=1, bshift=0.2, sig=None):
    stc = st.copy()
    for i, tr in enumerate(stc):
        end = tr.stats["p_time"] - bshift

        if sig is not None:  # get the same length as the signal window
            start = end - (sig[i].stats["wend"] - sig[i].stats["wstart"])
        else:
            start = end - bf
        link_window_to_trace(tr, start, end)
        tr.trim(start, end)
    return stc
