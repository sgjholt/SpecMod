import dataclasses
import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np
import obspy
from matplotlib.ticker import NullFormatter, StrMethodFormatter
from scipy.integrate import cumulative_trapezoid

from . import config as cfg
from . import utils as ut


def _mtspec(*args, **kwargs):
    """Call ``mtspec.mtspec``, the pre-refactor backend.

    Retained only so a run can be compared against the original Fortran
    library; :func:`estimate_spectrum` is what the pipeline uses. mtspec 0.3.2
    ships as Fortran source with no wheels and does not build without a
    compiler, so it is resolved on first use rather than imported eagerly.
    """
    try:
        from mtspec import mtspec
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The mtspec backend is not installed. It is a legacy optional "
            "dependency needing a Fortran compiler; install it with "
            "`pip install specmod[mtspec]`, or use specmod.transforms instead."
        ) from exc
    return mtspec(*args, **kwargs)


def estimate_spectrum(data, delta, *, motion="velocity", **kwargs):
    """Transform a record, using whichever estimator the configuration names.

    The bridge between this module and :mod:`specmod.transforms`. Every
    estimator — FFT, Welch, multitaper, Prieto, quadratic, CWT — becomes
    available to the pipeline through :class:`specmod.config.TransformConfig`,
    where before there was a hardcoded call to ``mtspec(data, delta, 3)``.

    Returns a :class:`specmod.core.Spectrum`, which carries its own units, so
    the caller no longer has to track how many times the record has been
    integrated or what amplitude convention is in force.

    Keyword arguments override the configured estimator's parameters, which is
    how the legacy ``**kwargs`` passthrough from ``Spectra.from_streams``
    keeps working.
    """
    from .config import load_config
    from .transforms import ESTIMATORS

    transform = load_config().config.transform
    name = kwargs.pop("estimator", transform.estimator)
    if name == "mtspec":
        raise ValueError(
            "estimator='mtspec' is the pre-refactor Fortran backend and is not "
            "wired into the pipeline. Use 'prieto' for the same lineage with "
            "no compiler, or 'multitaper' for the native implementation."
        )

    cls = ESTIMATORS[name]
    fields = {f.name for f in dataclasses.fields(cls)}
    settings = {
        key: value
        for key, value in dataclasses.asdict(transform).items()
        if key in fields
    }
    settings.update({k: v for k, v in kwargs.items() if k in fields})
    return cls(**settings).estimate(data, delta, motion=motion)


# VARIABLES READ FROM CONFIG

SUPPORTED_SAVE_METHODS = ["pickle"]

BW_METHOD = cfg.SPECTRAL["BW_METHOD"]

PLOT_COLUMNS = cfg.SPECTRAL["PLOT_COLUMNS"]

BINNING_PARAMS = cfg.SPECTRAL["BIN_PARS"]

BIN = True

SCALE_PARSEVAL = cfg.SPECTRAL["SCALE_PARSEVAL"]

ROTATE_NOISE = cfg.SPECTRAL["ROTATE_NOISE"]
ROT_METHOD = cfg.SPECTRAL["ROT_METHOD"]
ROT_PARS = cfg.SPECTRAL["ROT_PARS"]

SNR_TOLERENCE = cfg.SPECTRAL["SNR_TOLERENCE"]
MIN_POINTS = cfg.SPECTRAL["MIN_POINTS"]

ASSERT_BANDWIDTHS = cfg.SPECTRAL["ASSERT_BANDWIDTHS"]
SBANDS = cfg.SPECTRAL["S_BANDS"]


# classes
class Spectrum:
    """
    Spectrum class.
    """

    freq = np.array([])
    amp = np.array([])
    meta = {}
    id = " "
    kind = " "
    event = " "
    freq_lims = np.array([0.0, 0.0])
    __tr = obspy.Trace(np.array([]))
    bamp = np.array([])
    bfreq = np.array([])

    def __init__(self, kind, tr=None, **kwargs):
        # if a trace is passed assume it needs to be converted to frequency.
        if tr is not None:
            self.__set_metadata_from_trace(tr, kind)
            self.__calc_spectra(**kwargs)
            self.psd_to_amp()
            self.__bin_spectrum(**BINNING_PARAMS)

    def psd_to_amp(self):
        """Convert power spectral density to Fourier amplitude.

        ``A = sqrt(PSD * T / 2)``, which is ``|X(f)|`` — the *unfolded* Fourier
        transform magnitude, ``|rfft(x)| * dt``.

        .. note::

           **This is not the same convention as** :class:`specmod.core.Spectrum`,
           deliberately. That class carries a *folded* one-sided spectrum,
           ``2|X|``, so that ``energy()`` integrates to ``sum(x**2)*dt`` over
           non-negative frequencies alone. Both are self-consistent and both
           recover the record's energy; they differ by exactly a factor of two.

           This module uses the unfolded convention because ``Omega`` is defined
           in it. The long-period plateau of the displacement spectrum is
           ``|X(f -> 0)| = |integral u dt|``, and ``M0`` is proportional to that
           — so folding would put ``M0`` out by two, which is 0.2 magnitude
           units on every event. That is a convention to hold fixed, not to
           improve upon.

           :func:`estimate_spectrum` therefore returns a ``core.Spectrum`` in
           the folded convention, and the factor is removed here on the way in.
           Anyone reading ``core.Spectrum.amp`` directly and calling it
           ``Omega`` needs to halve it first.

        .. warning::

           The pre-refactor code computed this same quantity, but as
           ``sqrt(PSD * len(freq) / sampling_rate)`` — using the length of the
           frequency axis as a stand-in for ``T/2``. That identity holds only
           for an unpadded one-sided transform, so the result moved whenever
           the axis length changed: zero-padding to ``4*npts`` halved the
           amplitude, and a backend returning a full-length axis (Prieto's)
           changed it by ``sqrt(2)``. That is the §2.2 bug, and keying off
           ``T`` is the fix. **Unpadded, the amplitudes are unchanged**, so a
           pre-refactor run reproduces.
        """
        self.amp = self._convert(self.freq, self.amp, "psd", "magnitude")
        if self.bamp.size > 0:
            self.bamp = self._convert(self.bfreq, self.bamp, "psd", "magnitude")

    def amp_to_psd(self):
        """Inverse of :meth:`psd_to_amp`."""
        self.amp = self._convert(self.freq, self.amp, "magnitude", "psd")
        if self.bamp.size > 0:
            self.bamp = self._convert(self.bfreq, self.bamp, "magnitude", "psd")

    def _convert(self, freq, amp, source, target):
        """Change amplitude convention via :class:`specmod.core.Spectrum`.

        Delegated rather than reimplemented. The relationship is not a single
        scalar — the fold between ``FAS`` and ``|X|`` is two in the interior but
        one at DC and Nyquist, which have no negative-frequency twin — and a
        second copy of that rule is precisely how the two halves of the package
        would drift apart.
        """
        from .core import Spectrum as _CoreSpectrum

        converted = _CoreSpectrum(
            freq=np.ascontiguousarray(freq, dtype=float),
            amp=np.ascontiguousarray(amp, dtype=float),
            motion=getattr(self, "motion", "velocity"),
            kind=source,
            duration=self._duration(),
            sampling_rate=float(self.meta["sampling_rate"]),
        ).to_kind(target)
        return np.array(converted.amp, dtype=float)

    def _duration(self):
        """Physical record length in seconds.

        Taken from the trace metadata, never from ``len(freq)``. That is the
        whole point: the frequency axis lengthens under zero-padding while the
        record does not, and keying the normalisation off the axis is what made
        the pre-refactor amplitudes padding-dependent.
        """
        return float(self.meta["npts"]) * float(self.meta["delta"])

    def quick_vis(self, **kwargs):
        _fig, ax = plt.subplots(1, 1)
        ax.set_title(f"Event Id: {self.event}")
        ax.loglog(self.freq, self.amp, label=self.id, **kwargs)
        ax.legend()
        ax.set_xlabel("freq [Hz]")
        ax.set_ylabel("spectral amp")

    def integrate(self):
        self.amp /= 2 * np.pi * self.freq
        self.bamp /= 2 * np.pi * self.bfreq

    def differentiate(self):
        self.amp *= 2 * np.pi * self.freq
        self.bamp *= 2 * np.pi * self.bfreq

    def __set_metadata_from_trace(self, tr, kind):
        self.__tr = tr.copy()  # make a copy so you dont delete original
        self.meta = self.__sanitise_trace_meta(dict(self.__tr.stats))
        self.id = self.__tr.id
        self.kind = kind
        try:
            self.event = str(self.meta["otime"])
        except KeyError:
            self.event = None

    def __calc_spectra(self, **kwargs):
        """Transform the trace with the configured estimator.

        Stores a PSD, because ``__init__`` calls :meth:`psd_to_amp` next and
        the legacy call sequence is preserved. The estimator itself works in
        Fourier amplitude and the conversion is exact, so nothing is lost by
        going round that way — it just keeps ``Spectra``, ``SNP`` and the
        fitting code working unchanged.
        """
        spectrum = estimate_spectrum(
            self.__tr.data.astype(float), float(self.meta["delta"]), **kwargs
        )
        # PSD here, MAGNITUDE after psd_to_amp. Every estimator is held to the
        # same Parseval contract, so they all arrive on the same convention and
        # one conversion covers all of them — verified per estimator in
        # tests/test_spectral_wiring.py rather than assumed.
        psd = spectrum.to_kind("psd")
        del self.__tr
        # Copies, not views. core.Spectrum marks its arrays read-only so a
        # spectrum cannot be mutated behind its own back, but the legacy classes
        # here update amp in place (SNP.__scale_noise_parseval, integrate,
        # differentiate, the noise rotation). Handing out the frozen array makes
        # every one of those raise.
        self.amp = np.array(psd.amp, dtype=float)
        self.freq = np.array(psd.freq, dtype=float)
        self.motion = str(spectrum.motion)
        self.estimator = spectrum.meta.get("estimator")

    def __sanitise_trace_meta(self, m):
        nm = {}
        for k, v in m.items():
            if k not in ["processing", "sac", "calib", "__format"]:
                if type(v) not in [float, int, str, np.float64, np.float32]:
                    # print(k, type(v))
                    nm.update({k: str(v)})
                else:
                    nm.update({k: v})
        return nm

    def __bin_spectrum(self, smin=0.001, smax=200, bins=101):
        """Average into log-spaced bins.

        The default edges are wider than any real record: 0.001 Hz is far below
        ``1/T`` and 200 Hz far above Nyquist for a 100 sps trace, so on the
        Magna data roughly a third of the requested bins sit below the lowest
        frequency present and a third above the highest. Those come out empty
        and are dropped, which is why the surviving axis has always been much
        shorter than ``bins``.

        Clamped to the record's own range so the requested bin count is the
        count you get. :class:`specmod.smoothing.LogBinner` is the rewritten
        version of this and derives its edges the same way; this stays here
        because the legacy pipeline reads ``bamp``/``bfreq`` directly.
        """
        smin = max(smin, float(self.freq.min()))
        smax = min(smax, float(self.freq.max()))
        # define the range of bins to use to average amplitudes and smooth spectrum
        space = np.logspace(np.log10(smin), np.log10(smax), bins)
        # initialise numpy arrays
        bamps = np.zeros(int(len(space) - 1))
        bfreqs = np.zeros(int(len(space) - 1))
        # iterate through bins to find mean log-amplitude and bin center (log space)
        for i, bbb in enumerate(itertools.pairwise(space)):
            bb, bf = bbb
            inside = self.amp[(self.freq >= bb) & (self.freq <= bf)]
            # Log bins over a linear frequency grid are inevitably sparse at the
            # low end, so empty bins are expected rather than exceptional. They
            # are marked NaN and dropped below; taking the mean of an empty
            # slice would reach the same answer via a RuntimeWarning per bin.
            bamps[i] = 10 ** np.log10(inside).mean() if inside.size else np.nan
            bfreqs[i] = 10 ** (np.mean([np.log10(bb), np.log10(bf)]))

        # remove nan values
        self.bfreq = bfreqs[np.logical_not(np.isnan(bamps))]
        self.bamp = bamps[np.logical_not(np.isnan(bamps))]
        self.BAMP = bamps
        self.BFREQ = bfreqs


class Signal(Spectrum):
    """
    Signal is a subclass of spectrum intended to compute the spectrum of a signal
    trace.
    """

    # Signal class has an additional model attributes with the model params
    # and a model function

    model = None
    pass_snr = True
    ubfreqs = np.array([])

    def __init__(self, tr=None, **kwargs):
        super().__init__("signal", tr=tr, **kwargs)

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def set_ubfreqs(self, ubfreqs):
        self.ubfreqs = ubfreqs

    def get_ubfreqs(self):
        return self.ubfreqs

    def set_pass_snr(self, p):
        self.pass_snr = p

    def get_pass_snr(self):
        return self.pass_snr


class Noise(Spectrum):
    """
    Noise is a subclass of spectrum intended to compute the spectrum of a noise
    trace.
    """

    def __init__(self, tr=None, **kwargs):
        super().__init__("noise", tr=tr, **kwargs)


class SNP:
    """
    Lower level container class to associate signal and noise spectrum objects.
    """

    signal = None
    noise = None
    bsnr = np.array([0.0])
    event = " "
    ubfreqs = np.array([])
    itrpn = True
    ROTATED = False

    def __init__(self, signal, noise, interpolate_noise=True):
        self.__check_ids(signal, noise)
        self.signal = signal
        self.noise = noise
        self.pair = (self.signal, self.noise)
        self.__set_metadata(interpolate_noise)
        if SCALE_PARSEVAL:
            self.__scale_noise_parseval()
        if self.intrp:
            self.__interp_noise_to_signal()
        self.__get_snr()

    def integrate(self):
        for s in self.pair:
            s.integrate()
        # must recalculate usable frequency-bandwidth
        if self.intrp:
            self.__get_snr()

    def differentiate(self):
        for s in self.pair:
            s.differentiate()
        # must recalculate usable frequency-bandwidth
        if self.intrp:
            self.__get_snr()

    def psd_to_amp(self):
        for s in self.pair:
            s.psd_to_amp()

    def amp_to_psd(self):
        for s in self.pair:
            s.amp_to_psd()

    @property
    def bsnr(self):
        return self._bsnr

    @bsnr.setter
    def bsnr(self, arr):
        # assert type(arr) is type(np.array())
        self._bsnr = arr

    def __scale_noise_parseval(self):
        self.noise.amp *= np.sqrt(len(self.signal.amp) / len(self.noise.amp))
        self.noise.bamp *= np.sqrt(len(self.signal.amp) / len(self.noise.amp))

    def __rotate_noise(self):
        if ROT_METHOD == 1:
            self.noise.bamp, th1, th2 = ut.rotate_noise_full(
                self.noise.bfreq,
                self.noise.bamp,
                self.signal.bamp,
                ret_angle=True,
                **ROT_PARS,
            )
            if th1 == 0 or th2 == 0:
                print(f"th1={th1}, th2={th2}")
                print(f"rotation failed for {self.signal.id}")

            self.noise.amp = ut.rotate_noise_full(
                self.noise.freq,
                self.noise.amp,
                self.signal.amp,
                th1=th1,
                th2=th2,
                **ROT_PARS,
            )

        if ROT_METHOD == 2:
            rot = ut.non_lin_boost_noise_func(
                self.noise.bfreq, self.noise.bamp, self.signal.bamp, **ROT_PARS
            )

            self.noise.bamp *= rot

            self.noise.amp *= np.interp(self.noise.freq, self.noise.bfreq, rot)

    def __calc_bsnr(self):
        if ROTATE_NOISE and not self.ROTATED:
            self.ROTATED = True
            self.__rotate_noise()
        # set bsnr to the object
        self.bsnr = self.signal.bamp / self.noise.bamp

    def __get_snr(self):
        self.__calc_bsnr()
        self.__find_bsnr_limits()
        self.__update_lims_to_meta()
        if ASSERT_BANDWIDTHS:
            self.__assert_bandwidths_test()

    def __assert_bandwidths_test(self):
        mns = np.zeros(len(SBANDS))
        for i, bws in enumerate(SBANDS):
            inds = np.where((self.signal.freq >= bws[0]) & (self.signal.freq < bws[1]))[
                0
            ]
            mns[i] = np.mean(self.signal.amp[inds]) / np.mean(self.noise.amp[inds])

        if np.any(mns < SNR_TOLERENCE):
            self.signal.set_pass_snr(False)

    def __update_lims_to_meta(self):
        if self.signal.ubfreqs.size > 0:
            self.signal.meta["lower-f-bound"] = self.signal.ubfreqs[0]
            self.signal.meta["upper-f-bound"] = self.signal.ubfreqs[1]
        else:
            self.signal.meta["lower-f-bound"] = None
            self.signal.meta["upper-f-bound"] = None

        self.signal.meta["pass_snr"] = self.signal.pass_snr

    def quick_vis(self, ax=None):

        if ax is None:
            _fig, ax = plt.subplots(1, 1)
        else:
            pass

        ax.set_title(f"Event Id: {self.event}")
        ax.loglog(self.noise.freq, self.noise.amp, "b--", label="noise")
        ax.loglog(self.signal.freq, self.signal.amp, "k", label=self.signal.id)
        if self.signal.model is not None:
            if self.signal.model.result is not None:
                ax.loglog(
                    self.signal.model.mod_freq,
                    10**self.signal.model.result.best_fit,
                    color="green",
                    label="best fit model",
                )
        if self.ubfreqs.size > 0:
            if self.signal.pass_snr:
                for lim in self.ubfreqs:
                    ax.vlines(
                        lim,
                        np.min([self.noise.amp.min(), self.signal.amp.min()]),
                        np.max([self.noise.amp.max(), self.signal.amp.max()]),
                        color="r",
                        linestyles="dashed",
                    )
            else:
                ax.set_title("SNR TEST FAILED")
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend()
        ax.set_xlabel("freq [Hz]")
        ax.set_ylabel("spectral amp")

        # if ret:
        #     return ax

    def __set_metadata(self, intrp):
        # global setting
        self.intrp = intrp
        # exposing these attributes to the highest level *lazyprogrammer*
        self.event = self.signal.event
        self.id = self.signal.id

    def __find_bsnr_limits(self):
        """
        Find the upper and lower frequncy limits of the bandwidth measure of
        signal-to-noise.
        """

        blw = np.where(self.bsnr >= SNR_TOLERENCE)[0]
        if blw.size <= MIN_POINTS:
            self.signal.set_pass_snr(False)
        else:
            if BW_METHOD == 1:
                self.set_ubfreqs(
                    self.find_optimal_signal_bandwidth(
                        self.signal.bfreq, self.bsnr, SNR_TOLERENCE
                    )
                )
            if BW_METHOD == 2:
                self.set_ubfreqs(self.find_optimal_signal_bandwidth_2())

    def set_ubfreqs(self, ubfreqs):
        self.ubfreqs = ubfreqs
        self.signal.set_ubfreqs(ubfreqs)

    def find_optimal_signal_bandwidth(
        self, freq, bsnr, bsnr_thresh, pctl=0.99, plot=False
    ):
        """
        Attempts to find the largest signal bandwidth above an arbitraty signal-to-Noise.
        We first map the SNR
        function to a space between -1, 1 by subtracting the SNR
        threshold then taking the sign)  taking the integral
        """
        inte = cumulative_trapezoid(np.sign(bsnr - bsnr_thresh))
        inte /= inte.max()
        inte[inte <= 0] = -1
        fh = np.abs(inte - pctl).argmin() - 1
        fl = np.abs(inte - (1 - pctl)).argmin()

        tryCount = 0
        while (fl >= fh) or fl == 0:
            inte[fl] = 1
            fl = np.abs(inte + 1 - pctl).argmin()
            tryCount += 1
            if tryCount == 3:
                print(f"WARNING: {self.id} is too noisy.")
                self.signal.set_pass_snr(False)
                break

        # if fl > 1:
        #     fl -= 2

        if not plot:
            if fh - fl < 3:
                self.signal.set_pass_snr(False)
            return np.array([freq[fl], freq[fh]])
        else:
            import matplotlib.pyplot as plt

            plt.plot(
                freq,
                np.sign(bsnr - bsnr_thresh),
                color="grey",
                label="sign(bsnr-bsnr limit)",
            )
            plt.plot(
                freq[1:], inte, color="k", lw=2, label="int[sign(bsnr-bsnr limit)]"
            )
            plt.vlines(
                freq[fl],
                inte.min(),
                inte.max(),
                linestyles="dashed",
                label=f"{100 - int(pctl * 100)}% & {int(pctl * 100)}%",
            )
            plt.vlines(freq[fh], inte.min(), inte.max(), linestyles="dashed", color="g")
            plt.title(f"ID:{self.id!s}, low f:{freq[fl]:.2f}, high f:{freq[fh]:.2f}")
            plt.legend()
            plt.ylabel("arb. units")
            plt.xlabel("freq [Hz]")

    def find_optimal_signal_bandwidth_2(self, plot=False):
        # get freq and ratio function
        f = self.signal.bfreq
        a = self.bsnr
        # get index of freqs > peak bsnr  and < peak bsnr
        indsgt = np.where(f > f[a == a.max()])
        indslt = np.where(f < f[a == a.max()])
        # get those freqs
        fh = f[indsgt]
        fl = f[indslt]

        try:
            ufl = fh[np.where(a[indsgt] - SNR_TOLERENCE <= 0)[0] - 1][0]
            lfl = fl[np.where(a[indslt] - SNR_TOLERENCE <= 0)[0] + 1][-1]
        except IndexError as msg:
            print(msg)
            print("-" * 20)
            print("Doesn't meet at one end")
            self.signal.pass_snr = False
            return np.array([])

        if not plot:
            return np.array([lfl, ufl])
        else:
            plt.loglog(f, a, label=str(self.id))
            plt.hlines(SNR_TOLERENCE, f.min(), f.max())
            plt.vlines(f[a == a.max()], a.min(), a.max())
            plt.vlines(
                fh[np.where(a[indsgt] - SNR_TOLERENCE <= 0)[0] - 1][0],
                a.min() * 2,
                a.max() / 2,
            )
            plt.vlines(
                fl[np.where(a[indslt] - SNR_TOLERENCE <= 0)[0] + 1][-1],
                a.min() * 2,
                a.max() / 2,
            )

    def __check_ids(self, signal, noise):
        if signal.id.upper() != noise.id.upper():
            raise ValueError(f"ID mismatch between signal: {signal.id} and noise: ")
        if signal.kind.lower() == noise.kind.lower():
            raise ValueError(
                f"Cannot pair similar spectrum kinds: {signal.kind} with {noise.kind}"
            )

    def __interp_noise_to_signal(self):
        self.noise.amp = np.interp(self.signal.freq, self.noise.freq, self.noise.amp)
        # self.noise.diff_freq = self.noise.freq[np.where(self.noise.freq <= self.signal.freq.min())]
        self.noise.freq = self.signal.freq.copy()
        self.noise._Spectrum__bin_spectrum(
            **BINNING_PARAMS
        )  # need to recalc bins after interp.

    def __str__(self):
        return f"SNP(id:{self.id}, event:{self.event})"

    def __repr__(self):
        return "SNP(id:" + self.id + ", event:" + self.event + ")"


class Spectra:
    global PLOT_COLUMNS
    """
    Higher order container class for a group of SNP objects from a single event.
    """

    def sorter(x):
        return x.signal.meta["repi"]

    group = {}

    event = None

    def __init__(self, group=None):
        if group is not None:
            self.__check_group(group)
            self.__set_group_dict(group)

    @classmethod
    def from_streams(cls, sig, noise, **kwargs):
        """
        Takes a signal obspy stream and noise obspy stream (assuming they are
        ordered the same way) and, 1. calculates spectra, 2. pairs signal and
        noise then 3. groups them together into a single event. The key word
        arguements are passed to the Signal/Noise <- Spectrum objects and are
        then passed to the mtspec function from the mtspec library.
        """
        sig, noise = sig.copy(), noise.copy()
        snps = []
        for s, n in zip(sig, noise):
            print(f"Doing {s.id}")
            snps.append(SNP(Signal(s, **kwargs), Noise(n, **kwargs)))
        return cls(snps)

    @staticmethod
    def write_spectra(path, spectra, method="pickle"):
        write_methods(path, spectra, method)

    @staticmethod
    def read_spectra(path, method, skip_warning=False):

        if skip_warning:
            return read_methods(path, method)
        else:
            print("=" * 40)
            print("WARNING: Unpickling objects is dangerous.")
            print(
                "Please ensure that these are a spectra object and you KNOW \
                   who has modified these files AND you trust them."
            )
            print("=" * 40)
            x = input("Open spectra file?")
            if x.lower() in ["y", "yes"]:
                return read_methods(path, method)
            else:
                print(f"Did not open {path}.")

    def psd_to_amp(self):
        for g in self.group.values():
            g.psd_to_amp()

    def amp_to_psd(self):
        for g in self.group.values():
            g.amp_to_psd()

    def inte(self):
        for g in self.group.values():
            g.integrate()

    def diff(self):
        for g in self.group.values():
            g.differentiate()

    def get_spectra(self, id):
        if id.upper() in self.group.keys():
            x = self.group[id.upper()]
            return x
        else:
            print(f"id {id.upper()} not found")
            print(list(self.group.keys()))

    def __check_group(self, group):
        l = [s.event for s in group]
        if not l[1:] == l[:-1]:
            raise ValueError(f"Events are mismatched: {l}")

    def __set_group_dict(self, group):
        # Use a dict so we have a simple way to reference a particular
        self.group = {g.id: g for g in group}
        self.event = group[0].event

    def get_available_channels(self):
        """
        Return a list of channels.
        """
        return list(self.group.keys())

    def quick_vis(self, save=None, ret=True):
        l = self.__num_rows()
        fig, axes = plt.subplots(l, PLOT_COLUMNS, figsize=(17, int(l * 5)))
        axes = axes.flatten()
        for g, ax in zip(self.group.values(), axes):
            g.quick_vis(ax)
        fig.tight_layout()
        if save is not None:
            assert type(save) is str
            fig.savefig(save)
            fig.clear()
            plt.close(fig)
            print("deleted spec fig")
        if ret:
            return fig, axes

    def __str__(self):
        return f"Spectra(event:{self.event}, size:{self.__len__()})"

    def __repr__(self):
        return "Spectra(event:" + self.event + ", size:" + str(self.__len__()) + ")"

    def __len__(self):
        return len(self.group)

    def __num_rows(self):
        l = self.__len__()
        cols = PLOT_COLUMNS
        if l % cols > 0:
            return int((cols * (int(l / cols) + 1)) / cols)
        else:
            return int(l / cols)


# functions


def write_methods(path, thing, method):
    """
    write_methods function has all of necesary commands to write objects in
    number of formats.
    """
    global SUPPORTED_SAVE_METHODS

    if method.lower() in SUPPORTED_SAVE_METHODS:
        if method.lower() == "pickle":
            if not path.endswith(".spec"):
                path = ".".join([path, "spec"])
            with open(path, "wb") as f:
                pickle.dump(thing, f)
    else:
        raise TypeError(f"{method.lower()} method is not currently supported")


def read_methods(path, method):
    """
    write_methods function has all of necesary commands to write objects in
    number of formats.
    """
    global SUPPORTED_SAVE_METHODS

    if method.lower() in SUPPORTED_SAVE_METHODS:
        if method.lower() == "pickle":
            if not path.endswith(".spec"):
                path = ".".join([path, "spec"])
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return obj
    else:
        raise TypeError(f"{method.lower()} method is not currently supported")
