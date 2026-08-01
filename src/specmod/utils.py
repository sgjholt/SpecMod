import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import obspy

matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import numpy as np
import pandas as pd
from matplotlib.dates import num2date

# def get_filt_params(x):
#     return x.split("options=")[-1].split("::")[0].strip("{").strip("}")
#
# get_filt_params(sig[0].stats.processing[2])


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


def cps(st):
    return st.copy()


class DataSet:
    pdir = " "
    pref = ("HH", "BH", "EN")
    stat_pref = ("BRWY", "GAWY", "FMC", "SVWY")
    allpaths = []
    stations = []
    available = []
    filtered = []
    pref_sta = True

    def __init__(self, path, pdir="Data", pref_sta=True):
        self.pdir = os.path.join(pdir, path)
        self.pref_sta = pref_sta
        self.__startup()

    def get_obs_paths(self):

        apaths = [os.path.join(self.pdir, ".".join(t)) for t in zip(*self.available)]

        if self.pref_sta:
            return [p for p in apaths if p.split(".")[-2] in self.stat_pref]
        else:
            return apaths

    def __get_paths(self):
        tmp = glob.glob(os.path.join(self.pdir, "*.sac"))
        self.allpaths = [d.split("/")[-1] for d in tmp]
        print(f"found {len(self.allpaths)} files in {self.pdir}.")

    def __get_stations(self):
        self.stations = sorted({x.split(".")[-3] for x in self.allpaths})

    def __set_ranks(self):
        self.ranks = rank_chans(self.pref)

    def __get_available(self):
        self.available = get_avail(self.ranks, self.allpaths)

    def __startup(self):
        self.__get_paths()
        self.__get_stations()
        self.__set_ranks()
        self.__get_available()


def getchan(path):
    return path.split(".")[-2]


def rank_chans(prefs):
    num = [x + 1 for x in range(len(prefs))]
    ranks = {}
    for p, n in zip(prefs, num):
        ranks.update({p: n})
    return ranks


def compare_ranks(ranks, current, new):
    return ranks[current] < ranks[new]


def get_avail(ranks, allpaths):
    fs, chan = [], []
    for f in allpaths:
        if getchan(f)[:2] in ranks.keys():
            tmp = ".".join(f.split(".")[:-2])
            if tmp in fs:
                ind = fs.index(tmp)
                new = getchan(f)[:2]
                if compare_ranks(ranks, chan[ind], new):
                    chan[ind] = new

            else:
                fs.append(tmp)
                chan.append(getchan(f)[:2])
    return fs, chan


def read_cat(path):
    return pd.read_csv(path, sep=r"\s+")


def cat2kstyle(row):
    d = row["Date"].replace("/", ".")
    t = row["Time"].replace(":", ".")[:-3]
    return ".".join([d, t])


def keith2utc(row):
    return obspy.UTCDateTime(*map(int, cat2kstyle(row).split(".")))


def path_to_utc(p):
    return obspy.UTCDateTime(*list(map(int, p.split(".")[1:])))


def find_rotation_angle(
    arr_from_x, arr_from_y, arr_to, cond=-1, inc=0.01, backwards=False
):
    """
    Find the rotation angle required to raise one amplitude on a given array to the same
    amplitude on a target array. This angle may be used to rotate the entire array.
    The code increases (or decreases if backwards) the angle iteratively until the amplitude is
    higer than or equal to the target amplitude.
    """

    max_its = 1000
    its = 0
    th = 0
    tmp_from = arr_from_y[cond]

    # They might already meet at one end, don't have to rotate.
    if 10**tmp_from >= 10 ** arr_to[cond]:
        print("already same level")
        return 0

    while 10**tmp_from <= 10 ** arr_to[cond]:
        tmp_from = (
            arr_from_x[cond] * np.sin(th)
            + arr_from_y[cond] * np.cos(th)
            + arr_from_y[0] * th
        )
        if backwards:
            th -= inc
        else:
            th += inc
        its += 1
        if its > max_its:
            print("Didn't ever meet.")
            return 0
    if backwards:
        th += inc
    else:
        th -= inc
    return th


def find_rotation_angle_v2(arr_from_x, arr_from_y, arr_to, inc=0.01, backwards=False):
    """
    Find the rotation angle required to raise one amplitude on a given array to the same
    amplitude on a target array. This angle may be used to rotate the entire array.
    The code increases (or decreases if backwards) the angle iteratively until the amplitude is
    higer than or equal to the target amplitude.
    """

    max_its = 5000
    its = 0
    th = 0

    cent = np.sum((10**arr_from_x) * (10**arr_from_y)) / np.sum(10**arr_from_y)

    print(f"centroid f @ {cent:.3f} Hz")

    if backwards:
        inds = 10**arr_from_x < cent
    else:
        inds = 10**arr_from_x > cent

    # They might already meet at one end, don't have to rotate.
    if np.any(10 ** arr_to[inds] <= 10 ** arr_from_y[inds]):
        print("already same level")
        return 0

    tmp_from = arr_from_y.copy()

    while True:
        tmp_from = (
            arr_from_x * np.sin(th) + arr_from_y * np.cos(th) + arr_from_y[0] * th
        )
        # if its % 500 == 0 or its==0 or its==5000:
        #     plt.figure()
        #     plt.title(f"iteration # {its}")
        #     plt.plot(arr_from_x, tmp_from, 'b--', arr_from_x, arr_to, 'k')
        #     plt.show()
        #     input("Hi, I'm stopped")
        #     plt.close()
        if backwards:
            th -= inc
        else:
            th += inc
        its += 1

        if np.any(10 ** tmp_from[inds] >= 10 ** arr_to[inds]):
            break

        if its > max_its:
            print("Didn't ever meet.")
            return 0

    if backwards:
        th += inc
    else:
        th -= inc
    return th


def rotate(x, y, theta):
    """
    Rotate an array through a given angle (in radians).
    """
    if theta == 0:
        return y
    else:
        return x * np.sin(theta) + y * np.cos(theta) + y[0] * theta


def rotate_noise_full(
    xn, yn, ys, bcond=0, fcond=-1, inc=0.05, ret_angle=False, th1=None, th2=None
):
    """
    Performs a forward and backward rotation to match low and high frequencies.
    The output array is the input rotated forward and backwards and spliced
    together.
    """

    xn = np.log10(xn)
    yn = np.log10(yn)
    ys = np.log10(ys)
    if th1 is None and th2 is None:
        # th1=find_rotation_angle(xn, yn, ys, cond=bcond, backwards=True, inc=inc)
        # th2=find_rotation_angle(xn, yn, ys, cond=fcond, inc=inc)
        th1 = find_rotation_angle_v2(xn, yn, ys, backwards=True, inc=inc)
        th2 = find_rotation_angle_v2(xn, yn, ys, inc=inc)
    # print(th1, th2)
    yr1 = rotate(xn, yn, th1)
    yr2 = rotate(xn, yn, th2)

    if ret_angle:
        return np.maximum(10**yr1, 10**yr2), th1, th2
    else:
        return np.maximum(10**yr1, 10**yr2)


def get_centroid_freq(f, a):
    # Calc the center freq of spectrum
    return np.sum(f * a) / np.sum(a)


def non_lin_boost_noise_func(xn, yn, ys, inc, space):

    nb = 0
    max_its = 1000
    it = 0
    # determin low and high freqs with respect to centroid freq
    inds_b = xn <= get_centroid_freq(xn, ys)  # indices of 'low' freqs
    inds_f = ~inds_b  # indices of 'high' freqs

    sample_no = np.interp(xn, [xn.min(), xn.max()], space)
    # 'rotate' the low frequencies to signal
    while it < max_its:
        tmp_n_b = yn / sample_no**nb

        nb += inc
        it += 1
        # break condition looks for any low freq that is greater than signal
        if np.any(tmp_n_b[inds_b] >= ys[inds_b]):
            break

    sample_no_f = sample_no[::-1]
    nf = 0
    max_its = 1000
    it = 0

    while it < max_its:
        tmp_n_f = yn / sample_no_f**nf

        nf += inc
        it += 1

        if np.any(tmp_n_f[inds_f] >= ys[inds_f]):
            break

    return np.maximum(tmp_n_b, tmp_n_f) / yn
