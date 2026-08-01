# SpecMod Refactoring Plan

Status: **proposal / for discussion**
Target: a maintained, installable, tested `specmod` with pluggable spectral estimators.

---

## 1. Executive summary

SpecMod currently cannot be installed and, on a modern Python stack, cannot run.
Three separate hard breakages exist today:

| Breakage | Where | Effect |
|---|---|---|
| `mtspec` is source-only Fortran, no wheels | `Spectral.py:8` | `pip install mtspec` fails without `gfortran` (verified: `cannot execute 'f951'`). Blocks every install. |
| `scipy.integrate.cumtrapz` removed in SciPy 1.14 | `Spectral.py:398`, `PreProcess.py:231` | `AttributeError` on any SNR bandwidth call or refined window. |
| `pd.read_csv(delim_whitespace=)` removed in pandas 3.0 | `utils.py:213` | `TypeError` in `read_cat`. |

There is no `pyproject.toml`, no `__init__.py`, no tests, no CI, no version. The
tutorial works around the missing packaging with `os.chdir("../")` before every
import — that is the clearest single symptom of the state of the project.

Beyond the mechanical decay there is a structural problem that matters more for
correctness: **the package tracks none of its own physical state.** A `Spectrum`
does not know whether it holds PSD or amplitude, nor whether it is in
displacement, velocity or acceleration. That knowledge lives in a module-level
global (`Models.MOTION`, read from `config.py` at import time) which the user has
to keep manually in sync with however many times they called `.inte()` / `.diff()`.
Get that wrong and the fit silently returns a wrong seismic moment with no error.

This plan addresses, in order: (1) a safety net so the rewrite is verifiable,
(2) packaging/CI/tests, (3) a pluggable transform layer — FFT + smoothing,
multitaper, and CWT — which is the headline feature request, (4) architectural
decomposition, (5) release.

---

## 2. What is actually wrong

### 2.1 The `mtspec` dependency

`mtspec` 0.3.2 (2016) is a `ctypes`/f2py wrapper around Germán Prieto's Fortran
`mwlib`. It ships **sdist only** — every install compiles Fortran. Verified in a
clean container: the build reaches `adaptspec.f90` and dies with
`x86_64-linux-gnu-gcc: fatal error: cannot execute 'f951'`. It has no releases
since 2016 and no support for the NumPy 2.x ABI.

Replacements, in order of preference:

1. **`scipy.signal.windows.dpss` + own adaptive-weighting loop.** Zero new
   dependencies. DPSS tapers have been in SciPy since 1.1. Adaptive weighting
   (Thomson 1982) is ~40 lines. This should be the *default* multitaper path
   because it gives us full control of the normalisation — which is exactly what
   §4.4 needs.
2. **`multitaper` (PyPI, v1.2.0, pure Python)** — Prieto's own successor to the
   Fortran library, adding jackknife confidence intervals, the F-test for
   spectral lines, coherence and transfer-function estimation. Pure Python, so
   it installs anywhere. Ship it as an **optional extra**, `specmod[multitaper]`,
   for users who want the CIs.
3. Not recommended: `spectrum`, `nitime` (both lightly maintained).

The migration is not a drop-in swap. `mtspec(data, delta, 3, **kw)` returns a
*one-sided PSD*; the tutorial passes `quadratic=False, number_of_tapers=5`
straight through `Spectra.from_streams` → `Spectrum.__init__` → `mtspec`, so
backend kwargs leak through three layers of public API. Any replacement has to
be introduced behind an interface (§4.1) or the leak just moves.

### 2.2 A concrete units bug worth fixing carefully

`Spectrum.psd_to_amp` (`Spectral.py:65-82`):

```python
self.amp = np.sqrt((self.amp * len(self.freq)) / self.meta['sampling_rate'])
```

For an unpadded record this is *right*: `len(freq) ≈ N/2`, so the expression is
`sqrt(PSD · N · dt / 2) = sqrt(PSD · T / 2)`, the correct one-sided PSD →
amplitude-spectrum conversion. But it is right by coincidence of array length,
not by construction:

- **Zero-padding breaks it.** `mtspec` accepts `nfft`. Pad to `2N` and
  `len(freq)` doubles while the physical record duration `T` does not — the
  amplitude comes out inflated by `sqrt(2)`. The tutorial has a commented-out
  padding block ready to be uncommented.
- **The DC bin is stripped at `Spectral.py:127`** (`amp[1:], freq[1:]`) *before*
  the length is used, so `len(freq)` is `nfft/2` not `nfft/2+1` — a small bias,
  negligible for long records, wrong in principle.
- **The binned branch uses `len(self.freq)`,** the unbinned length. Consistent,
  but only by accident.
- **Nothing prevents calling it twice.** `Spectrum.__init__` calls it at
  construction (`Spectral.py:60`) and `Spectra.psd_to_amp()` is public. Calling
  it again square-roots amplitude a second time. There is no state flag.

The fix is not to patch the formula; it is to make the conversion a function of
**physical record duration `T = N·dt`** carried in the `Spectrum` metadata, and
to make the PSD/ASD/amplitude distinction a typed attribute so a double
conversion is an error rather than a silent factor. See §4.2.

### 2.3 Configuration is import-time global state

`Spectral.py:16-34` copies 12 values out of `config.py` into module globals at
import. `Models.py:35-36` does the same for `MODEL` and `MOTION`, and
`simple_model` reads `MOTION` via a `global` statement inside the function body
(after the docstring, so the docstring is a no-op string literal, `Models.py:57-61`).

Consequences:

- Changing `cfg.SPECTRAL[...]` after import has no effect. Users must edit
  `config.py` in the installed package.
- **You cannot fit a Brune and a Boatwright model in the same session.**
  `MODEL` is bound once at import.
- Tests cannot vary configuration without reimporting modules, and cannot run in
  parallel.
- `Spectral.py:490` `global PLOT_COLUMNS` sits inside a class body, above the
  docstring, which does nothing.

### 2.4 Class attributes standing in for instance state

Every class declares its fields at class level with mutable defaults:

```python
class Spectrum:
    freq = np.array([]); amp = np.array([]); meta = {}   # Spectral.py:44-53
class Spectra:
    group = dict()                                        # Spectral.py:496
class FitSpectrum:
    sig = sp.Signal(); meta = {}                          # Fitting.py:22-30
class FitSpectra:
    spectra = sp.Spectra(); models = {}                   # Fitting.py:175-178
```

`FitSpectrum` and `FitSpectra` construct live `Signal()` / `Spectra()` objects at
**import time**, purely so that `type(x) is type(sp.Signal())` (`Fitting.py:80`,
`Fitting.py:301`) has something to compare against. Those checks should be
`isinstance`. The empty-object-as-sentinel pattern is also why `Spectrum()` has
to tolerate being constructed with no trace.

`Spectral.py:213` assigns `bsnr = np.array([0.])` and then defines a `bsnr`
property at line 255 which overwrites the name — the assignment is dead code.
`itrpn = True` (line 216) is a dead typo of `intrp`. `BIN = True`
(`Spectral.py:22`), `Spectrum.freq_lims`, and `Spectra.sorter` are unused.

### 2.5 Latent bugs and dead code paths

| Location | Bug |
|---|---|
| `Spectral.py:458` | `plt.loglog(f, a, label=name)` — `name` is undefined. `NameError` whenever `find_optimal_signal_bandwidth_2(plot=True)`. |
| `Fitting.py:280` | `.format(weight_method)` — undefined; the parameter is `wm`. `NameError` in the warning path of `__check_wm`. |
| `Fitting.py:303` | `type(signal)` in `__check_spectra` — undefined; should be `spectra`. `NameError` in the error path. |
| `Fitting.py:246` | `self.model[name].reset()` — attribute is `models`. `AttributeError`. |
| `PreProcess.py:166-168` | `cut_p` refine: `p_start = p_start + rw_start` then `p_end = p_start + rw_end` uses the *already-shifted* start, so the window end is displaced by `rw_start`. `cut_s` (lines 220-221) computes the end **before** updating the start and is correct. The two functions disagree. |
| `Fitting.py:270` | `os.makedirs(os.path.join(*path.split("/")[:-1]))` — `TypeError` on a bare filename, and POSIX-only. |
| `Spectral.py:481` | `self.noise._Spectrum__bin_spectrum(...)` — reaching through name mangling into another object's private method. |
| `Spectral.py:536` | `read_spectra` calls `input()`. A blocking interactive prompt inside library code; unusable in scripts, batch jobs or CI. |
| `Spectral.py:141-157` | Log bins hardcoded to 0.001–200 Hz regardless of Nyquist. Empty bins produce `nan` (plus a `RuntimeWarning` per empty slice) and are then filtered out, so `bfreq` silently differs in length between traces. |

Plus ~20 `print()` calls used as diagnostics where `logging` belongs, and
`SUPPORTED_SAVE_METHODS` declared `global` in functions that only read it.

### 2.6 Mutation and reversibility

`SNP.__init__` mutates the `Signal` and `Noise` objects handed to it — scaling
noise amplitude (`__scale_noise_parseval`), rotating it, interpolating it onto
the signal frequency axis, and re-binning. The inputs are not reusable
afterwards. `Spectrum.integrate()` / `.differentiate()` mutate in place; the only
"undo" is the inverse operation, which is not numerically exact and, more
importantly, leaves no record of which domain you ended up in.

### 2.7 Packaging, repo and process

- No `pyproject.toml` / `setup.py`. Not installable. Hence `os.chdir("../")` in
  the tutorial.
- No `specmod/__init__.py` — the package works only as an implicit namespace
  package. No `__version__`, no declared public API.
- `requirements.txt` is a `pip freeze` of a 2019 macOS dev environment: 50
  fully-pinned transitive deps including `appnope` (macOS-only), `mpi4py`,
  `SQLAlchemy`, `ipykernel`, `pyzmq`. `numpy==1.16.5`, `obspy==1.1.0`
  (current: 1.5.0), `pandas==0.25.3` (current: 3.0.5).
- No tests. `Tests/` contains an empty `__init__.py` and a duplicate of the
  tutorial tree.
- Repo weight is 14.7 MiB packed, but **not** because of the waveforms — those
  are 224 KB. It is a 9.9 MB unreferenced Utah earthquake catalog in an
  abandoned `Tests/Tutorial/` scaffold, 4.4 MB of PNG output embedded in the
  tutorial notebook, and a 10.5 MB StationXML covering 501 channels where ~20
  are used. All three are fixable in place; see §5.1.
- `.gitignore` is three lines. No CI, no linting, no CHANGELOG, no tags, no
  releases, no `CITATION.cff` (this is academic software — it needs one).
- Module names are `CapitalCase` (`Spectral.py`, `PreProcess.py`,
  `ModelGuess.py`), against PEP 8.

---

## 3. Design principles for the refactor

1. **Physical state is typed, not global.** A `Spectrum` knows its motion domain
   and its amplitude kind. Illegal conversions raise.
2. **Numerics are pure functions; classes only orchestrate.** Every estimator,
   smoother and model is a function or a small stateless object over arrays.
3. **Strategy pattern for anything with more than one method.** Transforms,
   smoothers, bandwidth selectors, noise-rotation schemes.
4. **Configuration is an object passed explicitly.** Module-level reads of
   `config` are removed entirely.
5. **Operations return new objects.** No in-place mutation of user-supplied data.
6. **Plotting lives in `viz`.** Domain classes do not import matplotlib.
7. **One normalisation contract, enforced by one test, for all backends.**

### 3.1 Clean break — and the one thing that must *not* break

There are no downstream users, so this is a full makeover: the 0.x API is
removed rather than deprecated. No `legacy.py`, no shim layer, no deprecation
cycle. Concretely, that buys:

- Names chosen for clarity, not continuity — `SNP` → `SignalNoisePair`,
  `Spectra` → `SpectrumSet`, `CapitalCase` modules → `snake_case`.
- `Spectrum` can be a frozen dataclass immediately, rather than growing a
  mutable compatibility facade.
- Each phase may change signatures freely; no phase carries the previous
  phase's API forward.
- Simpler mypy migration — no untyped shim module to carve out.
- Stay on `0.x` for the whole refactor, where SemVer permits breaking changes in
  minor bumps. **1.0 is tagged when the API stops moving, not before.** Until
  then breaking changes need only a changelog entry, which release-please
  generates from `feat!:`/`BREAKING CHANGE:` commits automatically.

**But "breaking" applies to the API only, not to the answers.** A refactored
SpecMod that runs cleanly and returns a different corner frequency for the same
waveform is a failure, and it is a failure that is very easy not to notice —
`f_c` and `t*` trade off against each other, so a plausible-looking fit can be
quietly wrong. Two consequences:

- The golden snapshots (§5, Tier 3) are **more** important under a clean break,
  not less. They are the only record of what the current code computes, and once
  the `mtspec` build environment is gone they cannot be regenerated. This is the
  argument for open question 4 below.
- Where the new code *should* differ — the zero-padding normalisation (§2.2), the
  `cut_p` window ordering (§2.5), the COI bandwidth floor (§4.4.2) — the change
  must be deliberate, isolated in its own commit, and recorded in the changelog
  as a numerical change with the before/after magnitude. These are bug fixes, and
  they will move published numbers.

Anything already computed with 0.x and headed for publication should be
regenerated with the 1.0 code, or the discrepancy understood.

---

## 4. Target architecture

```
src/specmod/
  __init__.py              # public API + __version__
  config.py                # Settings dataclasses; no module-level side effects
  core/
    units.py               # Motion(DISP|VEL|ACC), AmplitudeKind(PSD|ASD|AMPLITUDE)
    spectrum.py            # Spectrum: freq, amp, motion, kind, meta, duration
    pair.py                # SignalNoisePair  (was SNP)
    collection.py          # SpectrumSet      (was Spectra)
    meta.py                # TraceMeta dataclass — the fields we actually use
  transforms/              # <-- the headline change
    base.py                # SpectralEstimator protocol
    fft.py                 # FFTEstimator, WelchEstimator
    multitaper.py          # MultitaperEstimator (scipy dpss; optional pkg backend)
    wavelet.py             # CWTEstimator (Morlet, Torrence & Compo normalisation)
    scalogram.py           # Scalogram surface + time_average() -> Spectrum
    qc.py                  # ScalogramQC checks (COI coverage, concentration, ...)
    registry.py            # name -> estimator; entry-point plugin hook
  smoothing/
    base.py                # Smoother protocol
    konno_ohmachi.py       # wraps obspy.signal.konnoohmachismoothing
    log_bins.py            # replaces Spectrum.__bin_spectrum
    simple.py              # moving average, Savitzky-Golay
  snr/
    bandwidth.py           # BandwidthSelector strategies (methods 1 and 2)
    rotation.py            # the two noise-rotation schemes, as pure functions
  models/
    source.py              # BruneSource, BoatwrightSource as objects
    attenuation.py         # ConstantQ / FrequencyDependentQ
    composite.py           # SourceModel composition + motion scaling
  fitting/
    fitter.py              # lmfit wrapper
    guess.py               # was ModelGuess.py
  preprocess/
    windows.py, picks.py, geometry.py
  io/
    readers.py, writers.py # HDF5/npz + JSON sidecar; pickle behind opt-in flag
  acquire/
    config.py              # dataset config schema (TOML)
    fetch.py               # config -> ObsPy FDSN -> raw waveforms + inventory
    manifest.py            # provenance record + --verify diffing
  datasets/
    registry.py            # name -> URL + SHA256 for pooch
    loaders.py             # load_magna_2020(), load_pnr_2019() -> Dataset
  viz/
    plots.py               # all matplotlib
```

There is deliberately no compatibility shim layer. The 0.x public names
(`Spectra`, `SNP`, `FitSpectra`, `mod.simple_model`, …) are removed outright —
see §3.1.

### 4.1 The transform layer

The core interface:

```python
class SpectralEstimator(Protocol):
    def estimate(self, data: np.ndarray, dt: float) -> Spectrum: ...
```

Every estimator returns a `Spectrum` carrying `freq`, `amp`, an explicit
`AmplitudeKind`, and the physical `duration` used for normalisation. Backend
kwargs are constructor arguments on the estimator, not `**kwargs` threaded
through three layers of container class.

| Estimator | Implementation | Dependency |
|---|---|---|
| `FFTEstimator` | `scipy.fft.rfft`, configurable taper (Hann / Tukey / cosine), window-energy correction, padding-aware normalisation | core |
| `WelchEstimator` | `scipy.signal.welch` — good default for noise windows | core |
| `MultitaperEstimator` | `scipy.signal.windows.dpss` + adaptive weighting (Thomson 1982); `backend="multitaper"` delegates to Prieto's package for jackknife CIs and the F-test | core / `specmod[multitaper]` |
| `CWTEstimator` | Morlet CWT via FFT-domain convolution, own implementation for normalisation control (§4.4) | core |

`scipy.fft` also gives a free performance path: `workers=-1` for threading, and
`scipy.fft.set_backend` to drop in `pyfftw` or `mkl_fft` without touching call
sites.

Usage after the change:

```python
from specmod.transforms import MultitaperEstimator, FFTEstimator
from specmod.smoothing import KonnoOhmachi

spec = FFTEstimator(taper="tukey", alpha=0.05).estimate(tr.data, tr.stats.delta)
spec = KonnoOhmachi(bandwidth=40).smooth(spec)
# or
spec = MultitaperEstimator(nw=3.5, k=5, adaptive=True).estimate(...)
```

`mtspec` becomes an optional legacy backend (`specmod[mtspec]`) for one minor
version, emitting a `DeprecationWarning`, purely so results can be reproduced
against the old code. Then it goes.

### 4.2 Units and domain tracking

```python
@dataclass(frozen=True)
class Spectrum:
    freq: np.ndarray
    amp: np.ndarray
    motion: Motion              # DISPLACEMENT | VELOCITY | ACCELERATION
    kind: AmplitudeKind         # PSD | ASD | AMPLITUDE
    duration: float             # seconds — physical record length, not len(freq)
    meta: TraceMeta

    def to(self, motion: Motion) -> "Spectrum": ...      # returns new object
    def as_kind(self, kind: AmplitudeKind) -> "Spectrum": ...
```

This single change kills three bug classes at once: the padding bug in §2.2, the
double-`psd_to_amp` hazard, and the `Models.MOTION` desync. `models/composite.py`
reads the motion from the `Spectrum` it is handed, so `scale_to_motion` needs no
global and the model is chosen per-fit rather than per-import.

### 4.3 Smoothing and binning

`Spectrum.__bin_spectrum` becomes `smoothing.log_bins.LogBinner(fmin, fmax, n)`,
with `fmin`/`fmax` defaulting to a sensible fraction of Nyquist and the record
length rather than the hardcoded 0.001–200 Hz. Empty bins are handled explicitly
(masked, not silently dropped) so binned axes stay comparable across traces —
which the SNR code at `Spectral.py:298` currently assumes but does not enforce.

Konno–Ohmachi is already implemented in ObsPy
(`obspy.signal.konnoohmachismoothing`, bandwidth `b`, default 40) and ObsPy is
already a hard dependency, so this costs nothing. `fast-konno-ohmachi` is a
drop-in accelerated alternative if profiling shows the O(N²) smoothing matrix
matters for long records.

### 4.4 CWT and preserving ground-motion units

This is the part that needs real care, and it is worth doing as its own
workstream with its own acceptance test.

The problem: CWT coefficients `W(a,b)` carry units of `[signal]·√time` under L2
normalisation, or `[signal]` under L1. Neither is the `[signal]·s` of a Fourier
amplitude spectrum, so a naive `|W|` plotted against scale-derived frequency is
*not* comparable to an FFT amplitude spectrum and cannot be fed to the source
model — the corner frequency would be recoverable but Ω₀, and therefore M₀,
would not.

Proposed approach:

1. **L2-normalised Morlet** in the frequency domain, `ψ̂(a·ω)` scaled by
   `√(2πa/dt)` (Torrence & Compo 1998, eq. 6).
2. **Scale → Fourier frequency** by the analytic Morlet relation
   `f = (ω₀ + √(2+ω₀²)) / (4πa)`, so the output frequency axis means the same
   thing as the FFT's.
3. **Time-average `|W(a,b)|²`** over the analysis window → wavelet power
   spectrum, in `units²·s`.
4. **Exclude the cone of influence** from that time average, or weight by the
   valid-sample fraction per scale. Without this, short S-windows are biased low
   at low frequency — precisely the frequency band that constrains Ω₀.
5. **Close the Parseval bridge** using the reconstruction constant `C_δ` and the
   `dj·dt` factors, such that
   `Σ_j (|W_j|²/s_j)·(dj·dt/C_δ) = σ²` recovers the signal variance. `C_δ` is
   wavelet-specific (0.776 for Morlet with ω₀=6) and should be *computed* by the
   package at import for the configured `ω₀`, not hardcoded.
6. **Convert power → amplitude** in the same convention as `FFTEstimator`,
   accounting for the log-spaced `Δf` implied by the scale grid.

Steps 5–6 are where the derivation is easy to get subtly wrong, so the design
decision is: **do not trust the derivation, pin it with a test.** A synthetic
sinusoid of known amplitude, plus band-limited noise of known variance, must
return the same peak amplitude and the same integrated power from
`FFTEstimator`, `MultitaperEstimator` and `CWTEstimator` within a stated
tolerance. That test is the specification. Write it first.

Implementation note: use our own FFT-domain CWT rather than PyWavelets
specifically *because* of this — `pywt.cwt` normalisation conventions are not
documented to the level of precision needed here, and we would end up
reverse-engineering them anyway. PyWavelets stays an optional extra for users
who want other wavelet families; `ssqueezepy` is a possible later addition for
synchrosqueezing if there is demand.

#### 4.4.1 Two outputs: the scalogram and its time average

`CWTEstimator` produces **both**, from one transform:

```python
class Scalogram:
    """Full time-frequency surface. |W(a,b)|, COI-masked."""
    time: np.ndarray            # (n_times,)
    freq: np.ndarray            # (n_scales,) Fourier-equivalent frequencies
    power: np.ndarray           # (n_scales, n_times)
    coi: np.ndarray             # (n_times,) e-folding frequency limit
    motion: Motion
    meta: TraceMeta

    def time_average(self, *, mask_coi: bool = True) -> Spectrum: ...
    def qc(self) -> ScalogramQC: ...
```

`Scalogram.time_average()` applies §4.4 steps 3–6 and returns an ordinary
`Spectrum`, so the fitting pipeline is unchanged and the CWT is a drop-in
alternative to the multitaper estimator. The scalogram itself is what you look
at when a fit comes out wrong.

**The normalisation contract holds at the `Spectrum` boundary, not the
scalogram.** `Scalogram.power` is `|W(a,b)|²` in whatever the L2-Morlet
convention gives, documented but not claimed to be an amplitude spectrum. The
`C_δ` / `dj·dt` bridge is applied once, inside `time_average()`. This keeps one
normalisation code path and one test — a second "already normalised" surface
would be a second thing to get wrong.

#### 4.4.2 QC gate

The scalogram makes several failure modes visible that the current
amplitude-only SNR test cannot see, and most of them are cheap to automate. The
proposal is a `ScalogramQC` record attached to the `Spectrum` metadata, carrying
both the numbers and a pass/fail per check:

| Check | Catches |
|---|---|
| **COI coverage per frequency** — fraction of the window free of edge effects at each scale | The window is too short to resolve the low frequencies that constrain Ω₀ |
| **Temporal energy concentration** — Gini/kurtosis of energy over time, per band | Spikes, glitches, dropouts masquerading as broadband signal |
| **First-half vs. second-half spectral ratio** | Coda contamination, a second arrival inside the window, a mis-picked window |
| **Fraction of window energy inside the picked window vs. at its edges** | Window mis-alignment — the arrival is clipped or the window started late |

The COI check is worth singling out. **Window length imposes a hard
low-frequency resolution limit that nothing in the current code enforces** — SNR
bandwidth selection (§4.5) is purely amplitude-based, so a 2 s S-window can
happily report a usable bandwidth down to 0.5 Hz where the transform has no
support. Feeding the COI limit into `BandwidthSelector` as a lower bound is a
genuine correctness improvement, and it applies to the FFT and multitaper paths
too (where the equivalent bound is `~1/T`, just less visible).

Default is compute-and-record, warn on failure, never silently drop a trace —
the QC flags land in the fit flat-file as columns so they can be filtered
downstream.

#### 4.4.3 Storage

A scalogram is `n_scales × n_times` floats — roughly 1 MB per trace for a 20 s
window at 100 Hz with 60 scales, so ~20 MB for a typical event. Therefore:
**scalograms are not persisted by default.** `io/` writes the derived `Spectrum`
plus the `ScalogramQC` record (a few dozen scalars); the full surface is written
only on `save(..., include_scalogram=True)`, into HDF5 with chunking and
compression. This is a large part of why §4.6 moves off pickle.

### 4.5 SNR, bandwidth and noise rotation

`find_optimal_signal_bandwidth` and `find_optimal_signal_bandwidth_2` become
`BandwidthSelector` strategies selected by argument, not by `BW_METHOD` global.
Both gain a **resolution floor** on the low-frequency end — the COI limit when
the spectrum came from a CWT, `~1/T` otherwise (§4.4.2). At present the selection
is purely amplitude-based and can return a usable band extending below what the
window length can resolve.

`rotate_noise_full` and `non_lin_boost_noise_func` move to `snr/rotation.py`
unchanged in behaviour but as pure functions with the iteration limits and
`print` diagnostics replaced by logging and a returned convergence flag — at
present `find_rotation_angle_v2` prints `"Didn't ever meet."` and returns `0`,
silently disabling the correction.

### 4.6 I/O

Pickle is the only persistence format, and `read_spectra` prompts on stdin
before unpickling. Replace with:

- **Primary:** HDF5 (`h5py`) or `.npz` for arrays + JSON sidecar for metadata —
  versioned, language-agnostic, diff-able metadata.
- **Secondary:** the existing flat-file CSV export, kept as-is.
- **Pickle:** dropped as a write format. See below for reads.

**Existing `.spec` files will not survive the clean break, and this is not
optional.** A pickle stores the *import path* of every class it contains —
`specmod.Spectral` / `Spectra`. Once `Spectral.py` becomes `core/collection.py`
and `Spectra` becomes `SpectrumSet`, `pickle.load` raises
`ModuleNotFoundError` before any of our code runs. Renaming the modules is
enough to break them; the dataclass conversion in §4.2 would finish the job. One
such file is already committed at `Tutorial/Spectra/2019-08-26T07:30:47.0.spec`,
and there are presumably more in your working directories.

Two ways out, and the first is much better:

1. **Convert in the old environment.** Phase 0 is already building a Docker image
   where the 0.1.0 code runs (§6.5). Add a small
   `scripts/convert_legacy_spec.py` to it that loads `.spec` files with the old
   classes present and writes the new HDF5 format. This is a one-shot migration
   with no lasting cost to the codebase.
2. A custom `Unpickler` with a `find_class` remapping table in the new code —
   works, but bakes the old class layout permanently into the new package, which
   is exactly the kind of thing a clean break is supposed to avoid.

Recommendation: option 1, and run it over any `.spec` files you care about
*during* Phase 0, while the image is fresh.

---

## 5. Testing strategy

Currently zero tests. Proposed, in dependency order:

**Tier 1 — property tests (`pytest` + `hypothesis`).** These encode the physics
and are backend-independent:
- Parseval: `Σ|X(f)|²Δf` matches time-domain energy, for every estimator.
- Amplitude recovery: a sinusoid of amplitude `A` returns `A` (within taper
  correction) for every estimator. **This is the §4.4 acceptance test.**
- Linearity, and invariance to zero-padding (the §2.2 bug, as a regression test).
- `integrate ∘ differentiate ≈ identity` to within float tolerance.
- Unit conversions round-trip; illegal conversions raise.

**Tier 2 — synthetic end-to-end.** Generate a Brune spectrum with known
`(Ω₀, f_c, t*)`, inverse-FFT to a synthetic seismogram, add noise, and run the
whole pipeline. Assert parameter recovery within tolerance. This is the single
most valuable test in the suite: it validates preprocessing, transform, SNR,
binning and fitting together, and it is the only way to know the mtspec removal
did not change the science.

**Tier 3 — golden/regression.** Run the *current* code on the tutorial event and
on Magna (§5.2.4), and snapshot `freq`, `amp`, `bsnr`, `ubfreqs` and the fit
table to `.npz`. Every subsequent change is diffed against those snapshots, so
behaviour changes are visible and deliberate rather than discovered later.

> This must happen **before** any code changes, and it needs an environment where
> `mtspec` still builds — a one-off Docker image or conda env with `gfortran`,
> `numpy<2`, `python 3.9`. Budget half a day; without it the refactor is
> unverifiable. Magna makes this stronger still, because published values exist
> for it — see the three-way comparison in §5.2.5.

**Tier 4 — unit tests** for the bugs in §2.5, each written as a failing test
first.

**Test data:** see §5.1. Coverage target: 80% overall, 95% on `transforms/` and
`core/`.

### 5.1 Where the test data lives — it stays in git

An earlier draft of this plan recommended moving the tutorial dataset out of git
and fetching it with `pooch` or git-lfs. **That was wrong, and it was based on
file count rather than file size.** Measured:

| Path | Size | Verdict |
|---|---|---|
| `Tutorial/Data/` — 40 mseed waveforms | **224 KB** | **Keep in git.** These are the test fixtures. |
| `Tutorial/MetaData/pnr_inventory.xml` | 10.5 MB | Keep, but subset — see below |
| `Tests/Tutorial/Meta/UUSSeq.catalog` | 9.9 MB | **Remove from the working tree** — unreferenced, and cannot support the Magna work |
| `Tutorial/SpecModTutorial.ipynb` | 4.6 MB | Strip outputs — 4.4 MB of it is embedded PNGs |
| `Tutorial/Spectra/*.spec` | 333 KB | Convert (§4.6) or delete — it is regenerable |

Total pack size today is 14.7 MiB. That is a **small** repository, and the
waveforms are not why. Three fixes, none of which needs external hosting:

1. **`Tests/Tutorial/Meta/UUSSeq.catalog` — remove from the working tree.**
   9.9 MB, 120,482 events, referenced nowhere in any `.py` or `.ipynb`. It is the
   UUSS regional catalogue for Utah, `1962-08-10` → `2019-02-28`.

   > An earlier draft called this "from another project". That was overconfident
   > — given the Magna work (§5.2) a UUSS catalogue is plausibly connected to this
   > research. The disposition is unchanged for a more specific reason: **it ends
   > 2019-02-28, thirteen months before the Magna mainshock**, so it cannot
   > support that work, and it is a catalogue rather than waveforms, so it is not
   > a test fixture either. It also stays reachable in git history and is
   > regenerable from UUSS. Removing `Tests/Tutorial/` clears 63% of the
   > repository.
2. **Strip the notebook.** 4.4 MB of the 4.6 MB is base64 PNG output; the actual
   source is 10 KB. `nbstripout` in pre-commit (already proposed in §6.2) fixes
   this permanently. Under `myst-nb` (§6.3) outputs are regenerated at docs-build
   time anyway, so storing them is pure waste.
3. **Subset the inventory.** 501 channels across four networks (GB, LV, SD, UR)
   with 1471 response stages, where the tutorial reads ~20 channels from LV and
   UR only. `inv.select(...)` to the used channels and the event time window
   should give roughly 400–500 KB at ~21 KB/channel. The response stages must be
   kept — `remove_response` needs them — but GB and SD are entirely unused.

After all three, the PNR fixtures are comfortably under 1 MB and stay in git,
versioned alongside the code and expected outputs that consume them — which is
what you want for regression tests. Larger datasets go via `pooch`; see §5.2.

git-lfs remains the option to avoid: it still consumes repository storage, adds
a bandwidth quota that CI burns through quickly, breaks plain `git clone` for
anyone without the extension, and GitHub Actions checkouts need explicit LFS
handling that is easy to get wrong and fails confusingly.

### 5.2 Larger datasets: `specmod.datasets`, pooch, and the Magna 2020 event

**GitHub Release assets are the right host and they are free.** On a public
repository they do not count toward repository size, are not metered like LFS
bandwidth, and allow up to 2 GB per file — far more than needed here. `pooch`
fetches them by URL, verifies SHA256, and caches per-user, so the fetch happens
once per machine.

#### 5.2.1 A general, config-driven acquisition tool

The acquisition step is **not** a bespoke Magna script. It is a general waveform
grabber where every event-specific detail is declarative configuration, and
Magna is one config file among several. Datasets then cost a config, not code,
and the config *is* the provenance record.

```
specmod.acquire                   published artifact          tests and users
───────────────                   ──────────────────          ───────────────
datasets/magna_2020.toml     →    GitHub Release asset   →    datasets.load_*()
  + specmod fetch <config>         magna_2020_v1.tar.gz         pooch, SHA256-pinned
  → ObsPy FDSN client              config + manifest             cached, offline
  → raw waveforms + inventory        embedded inside
  → manifest with provenance
```

A config declares the event (by FDSN `eventid` where possible, so the origin is
resolved from the catalogue rather than retyped), the station selection
(network/station/location/channel patterns plus a radius or distance range), the
window relative to origin or to predicted arrivals, the pick source, and — the
detail people forget — **which data centre**, since different centres serve
different holdings for the same event.

Two design constraints worth fixing now:

- **Fetch raw, do not pre-process.** Store counts plus the response, not a
  deconvolved trace. Baking `remove_response` into the artifact removes it from
  test coverage and freezes one ObsPy version's behaviour into the fixture.
- **Keep the wrapper thin.** ObsPy already has the client, the retry logic and
  the chunking. The value added here is the declarative layer, the manifest, and
  the event-relative windowing that `preprocess/windows.py` needs anyway — not a
  reimplementation of `MassDownloader`, which is built for large restricted-data
  campaigns and is more machinery than this needs.

TOML is suggested for the config, read by stdlib `tomllib` on 3.11+ and
consistent with `pyproject.toml`. YAML is equally fine if it reads better for
nested station rules; it costs one small dependency.

#### 5.2.2 Why a config still is not enough — and what closes the gap

A config makes the **request** reproducible. It does not make the **response**
reproducible, because FDSN is not content-addressed and the archive moves under
you:

- Instrument responses are corrected retroactively.
- Waveform archives get backfilled and gaps repaired.
- Catalogue solutions are revised — magnitudes and locations included.
- Stations are added to holdings after the fact.

So re-running `magna_2020.toml` in 2028 will not necessarily return the bytes it
returns today. The config and the pinned artifact are **complementary, not
alternatives**: the config records intent and makes the dataset regenerable and
adaptable; the SHA256-pinned tarball is what tests actually consume, and it is
the only thing that makes a regression test mean anything.

That is also why **tests must never call FDSN.** Beyond outages and rate limits,
a suite that fetches live can silently change its own expected answer — exactly
the failure mode regression tests exist to catch.

What makes this genuinely reproducible rather than merely repeatable is the
**manifest**, written next to the data and embedded in the artifact:

| Field | Why |
|---|---|
| The config, verbatim | The recipe |
| Resolved query — exact `eventid`, channel list after wildcard expansion | Wildcards mean the config alone does not say what you got |
| Data centre URL + FDSN service version | Different centres, different holdings |
| ObsPy and SpecMod versions | Client behaviour changes |
| Fetch timestamp | When this view of the archive was taken |
| Per-file SHA256 | Integrity |

With that, `specmod fetch --verify magna_2020.toml` can re-run the config and
diff against the manifest. A clean diff means the archive is unchanged; a dirty
one tells you **the data centre revised something**, which for a seismologist is
a finding in its own right rather than bookkeeping — particularly if a response
correction lands under a published result.

#### 5.2.3 Where it lives: `specmod.acquire` + `specmod.datasets`

Making the grabber general changes where it belongs. As a bespoke Magna script it
was maintainer tooling; as a config-driven fetcher it is a **user-facing feature**
— anyone studying their own sequence writes a config and gets a SpecMod-ready
dataset, which is a good reason for the package to own it. Two modules, because
they have genuinely different jobs and dependency profiles:

```python
# specmod.acquire — produces datasets. Needs network. Used rarely.
from specmod.acquire import fetch
fetch("datasets/magna_2020.toml", out="build/magna_2020")
#   or: $ specmod fetch datasets/magna_2020.toml -o build/magna_2020

# specmod.datasets — consumes published artifacts. Offline after first use.
from specmod.datasets import load_magna_2020, load_pnr_2019
ds = load_magna_2020()                    # Dataset(stream, inventory, event, picks)
ds = load_magna_2020(aftershocks=True)
```

`datasets` uses `pooch.os_cache("specmod")`, a `SPECMOD_DATA_DIR` override, and a
registry mapping dataset name → URL + SHA256.

A separate library is still not worth it. Both halves are thin — a config reader
over ObsPy calls, and a registry over pooch — and splitting them out buys a
second release cycle, second CI, second changelog and a compatibility matrix
between `specmod` and `specmod-data`. What *does* need independent versioning is
the **data artifacts**, and pooch's registry handles that within one package:
`magna_2020_v1`, `_v2` as distinct entries, old versions still fetchable.

The configs themselves are small text files and live **in the repository**, under
`datasets/*.toml`, versioned with the code. That is the piece that makes the
whole thing regenerable, so it should not be hidden inside a release asset —
though a copy also travels inside each artifact (§5.2.2) so a downloaded dataset
is self-describing.

`acquire`'s own tests must not hit the network either: mock the ObsPy client, or
record cassettes. Its logic is config parsing, wildcard resolution, windowing and
manifest generation — all testable against a fake client.

> **Concrete gotcha:** data artifacts want their own release tags (`data-v1`),
> and release-please must be configured to ignore them or it will read `data-v1`
> as a code release. Constrain it to `v*` and keep the data tags on a separate
> prefix. If the release feed gets noisy, the fallback is a sibling
> `sgjholt/specmod-data` repository holding only assets — a repo, not a package,
> so no extra release cycle for code.

Tests that need a fetched dataset get `@pytest.mark.dataset`, so
`pytest -m "not dataset"` is a complete offline run. CI caches
`~/.cache/specmod` keyed on the registry hash, so the download happens once per
registry change rather than once per job.

#### 5.2.4 Magna, Mw 5.7, 2020-03-18 — the validation dataset

This is the strongest available anchor for the whole refactor, for a reason that
has nothing to do with data volume: **there are published results for this event
produced with the 0.1.0 code.** That converts Tier 3 from "trust a snapshot we
generated" into "reproduce a peer-reviewed number".

It is also scientifically complementary to the existing PNR fixtures in exactly
the way a test suite wants:

| | PNR 2019 | Magna 2020 |
|---|---|---|
| Setting | Induced, UK | Tectonic, Utah |
| Magnitude | ~M 1–2 | **Mw 5.7** + aftershock sequence |
| Corner frequency | ~10–30 Hz | ~0.3–0.5 Hz |
| Stresses | High-frequency end, short windows | **Low-frequency end, long windows** |

Two orders of magnitude in `f_c` across the two datasets, and Magna lands
squarely on the part of the new code that PNR cannot exercise: the COI /
window-length resolution floor (§4.4.2) only bites when `f_c` approaches `1/T`.
The aftershock sequence additionally gives many events over a magnitude range
from one region on one station set — the natural basis for the multi-event
regression set later.

Caveat worth stating in the docs: at Mw 5.7 the point-source Brune assumption is
more strained than for the induced events, and finite-source or directivity
effects may be visible. That is a property of the science, not the refactor, but
it should not be discovered as a surprise.

#### 5.2.5 The three-way comparison

The published values were produced by code containing the bugs in §2.2 and
§2.5. So agreement and disagreement are both ambiguous unless the comparison is
staged:

1. **Paper** — the published values.
2. **0.1.0 re-run** in the legacy Docker image, on the same data and parameters,
   → must reproduce (1). If it does not, the discrepancy is in parameters or
   environment, and it has to be resolved *before* anything downstream is
   interpretable.
3. **New code** → compared against (2), with each deliberate numerical change
   (padding normalisation, `cut_p` ordering, COI floor) accounted for
   individually and recorded in the changelog with its magnitude.

Step 2 is the one that gets skipped, and it is the one that makes step 3 mean
anything — without it, "the new code disagrees with my paper" has at least three
possible causes and no way to distinguish them. **This resolves the open question
about whether the legacy Docker image is worth half a day: it is, and Magna is
why.**

> Not verified here: FDSN endpoints are blocked from the environment this plan
> was written in (`CONNECT tunnel failed, response 403` for both USGS and
> EarthScope), so station availability, channel coverage and total dataset size
> for Magna are **estimates pending a real query**. The event parameters
> themselves (2020-03-18 13:09 UTC, Mw 5.7, ≈40.75 N, 112.08 W, ~12 km depth,
> UU network) are from general knowledge and should be confirmed against the
> catalogue when `build_dataset.py` is first run.

Suggested dataset scope, to be confirmed once the data can actually be queried:
mainshock plus a handful of aftershocks spanning M 2–4; UU network with a
distance spread, plus regional networks if coverage is thin; traces trimmed to
roughly −60 s to +300 s about origin; inventory subset to exactly the channels
included; picks and event parameters in a JSON manifest alongside. Target under
50 MB packed so the first-use fetch stays quick.

## 6. Packaging, CI/CD and process

### 6.1 Packaging

- `pyproject.toml` (PEP 621), `hatchling` backend, `src/` layout.
- Python 3.11–3.13. Floors not pins: `numpy>=1.26`, `scipy>=1.11`,
  `obspy>=1.4`, `lmfit>=1.2`, `pandas>=2.0`, `matplotlib>=3.7`.
- Extras: `[multitaper]`, `[wavelet]`, `[mcmc]` (emcee), `[mtspec]` (legacy,
  temporary), `[dev]`, `[docs]`. `pooch` is a core dependency — `specmod.datasets`
  (§5.2) is part of the public API, not a dev-only convenience.
- `uv` for dev environments and a committed lockfile for CI reproducibility.
- Delete `requirements.txt`.

### 6.2 Lint, format and types

**Ruff** replaces black + flake8 + isort + pyupgrade in one tool. Proposed
starting rule set, in `pyproject.toml`:

```toml
[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "C4", "NPY", "RUF", "PL", "PT", "D"]
ignore = ["D105", "D107", "PLR0913"]          # magic-method / __init__ docstrings, arg count
[tool.ruff.lint.pydocstyle]
convention = "numpy"
```

Three of those rule families do real work on this codebase rather than just
tidying it:

- **`PLW0603`** (part of `PL`) flags every `global` statement — §2.3's
  `Spectral.py:490`, `Models.py:58`, and the read-only `global
  SUPPORTED_SAVE_METHODS` uses. It turns the de-globalisation work of Phase 2
  into a checklist the linter maintains.
- **`RUF012`** flags mutable class-attribute defaults — §2.4's entire pattern,
  every occurrence, automatically.
- **`NPY`** catches legacy NumPy calls that break on 2.x, and **`UP`** does the
  syntax modernisation (3.11+ typing, f-strings) mechanically.
- **`F821`** catches the four undefined-name bugs in §2.5 *statically*, before a
  single test runs. Running ruff is the cheapest possible first step.

`D` (docstring rules, numpydoc convention) is the scientific-Python norm and
matches what Sphinx's `napoleon` extension expects, so docstrings and API docs
stay in sync. Enable `D` last — it will produce a lot of findings.

**Formatting the existing code:** one `ruff format` commit across the whole
tree, on its own, touching nothing else. Add its SHA to `.git-blame-ignore-revs`
and set `blame.ignoreRevsFile` in CI so `git blame` still reaches real authorship
through it. Do this in Phase 1, before any logic changes, so reformatting never
appears in a diff that also changes behaviour.

**Module renames** (`Spectral.py` → `spectral.py` etc.) happen in the same phase.
On a case-insensitive filesystem — likely, given the macOS `appnope` in
`requirements.txt` — these need `git mv A.py tmp && git mv tmp a.py` to register.

**mypy**, staged rather than all-at-once:

```toml
[tool.mypy]
strict = true
[[tool.mypy.overrides]]
module = ["obspy.*", "lmfit.*", "mtspec.*"]
ignore_missing_imports = true     # none of these ship type stubs
[[tool.mypy.overrides]]
module = ["specmod.preprocess.*", "specmod.viz.*"]
ignore_errors = true              # shrink this list each phase; target: empty
```

`strict` from the start on `core/` and `transforms/` — they are new code, and
the typed `Motion`/`AmplitudeKind` enums of §4.2 only actually prevent the
unit-mixing bugs if the type checker is enforcing them. The override list is the
migration backlog, and CI can assert it never grows.

**pre-commit** runs ruff (lint + format), mypy, `nbstripout` on the tutorial
notebook, and `check-added-large-files` — the last one specifically to stop
another 70 waveform files landing in git.

### 6.3 Documentation (Sphinx)

- **Sphinx** with `pydata-sphinx-theme` (the NumPy/SciPy/ObsPy house style —
  familiar to this audience, good API-reference layout).
- `myst-parser` so prose pages can stay Markdown; `myst-nb` to execute and render
  the tutorial notebook as a docs page, which makes the tutorial a **tested**
  artefact rather than a snapshot that silently rots.
- `autodoc` + `napoleon` (numpydoc style) + `sphinx-autodoc-typehints`, so
  signatures come from the annotations rather than being hand-maintained.
- `intersphinx` to numpy, scipy, obspy, lmfit, matplotlib.
- `sphinx.ext.doctest` — the units/normalisation examples in §4.2 and §4.4 are
  exactly the kind of thing that should be executable in the docs.
- `sphinx-build -W` (warnings as errors) in CI: a broken cross-reference or an
  undocumented public symbol fails the build.
- Structure: Getting started → User guide (preprocessing, transforms, SNR,
  fitting) → **Theory** (the normalisation conventions, one page, with the
  Parseval contract stated explicitly) → Tutorial → API reference → Migration
  guide from 0.x → Changelog.

The theory page matters more than usual here. The units question that prompted
this refactor is not obvious from the code, and if the conventions are only
encoded in tests, the next person to add an estimator will not find them.

### 6.4 Automated versioning and release

Two pieces, deliberately separated:

**Version derivation — `hatch-vcs`.** The version comes from the git tag; no
version string is ever committed, so there is nothing to forget to bump and no
`__version__`/tag skew. `__init__.py` reads it via
`importlib.metadata.version("specmod")`.

**Version *decision* — `release-please` (GitHub Action).** It parses
[Conventional Commits](https://www.conventionalcommits.org) since the last
release, works out the SemVer bump, and opens a standing "release PR" carrying
the generated `CHANGELOG.md`. Merging that PR creates the tag; the tag triggers
publication. Nothing is released until a human merges.

That human gate is the reason to prefer `release-please` over
`python-semantic-release` (which tags on every qualifying push to `main`)
**specifically because of Zenodo**: every GitHub release mints a new DOI, and DOIs
cannot be retracted. Fully-automatic tagging plus Zenodo means a typo fix can
mint a citable version of the software. Use `python-semantic-release` only if you
would rather have zero-touch releases and accept that.

This does impose Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`,
`feat!:` for breaking) on commit messages, enforced by a `commitlint` pre-commit
hook. It is a small discipline and it is what makes the changelog automatic.

### 6.5 CI (GitHub Actions)

| Workflow | Trigger | Does |
|---|---|---|
| `test.yml` | PR, push | matrix 3.11/3.12/3.13 × ubuntu/macos → ruff check, ruff format --check, mypy, pytest + coverage → Codecov |
| `docs.yml` | PR, push | `sphinx-build -W`; on `main`, deploy to GitHub Pages. Builds on PRs too, so doc breakage is caught before merge |
| `build.yml` | PR, push | sdist + wheel, `twine check`, install-from-wheel smoke test in a clean env (catches missing package data) |
| `release-please.yml` | push to `main` | maintains the release PR; creates tag + GitHub Release on merge |
| `publish.yml` | GitHub Release published | PyPI via Trusted Publishing (OIDC — no long-lived token in secrets) |

Zenodo is wired to the GitHub Release webhook, so the DOI is minted from the same
event as the PyPI upload. Branch protection on `main`: require `test`, `docs`
and `build` green.

**Versioning policy.** SemVer. Stay on `0.x` for the whole refactor — breaking
changes are expected and permitted in minor bumps there, so no deprecation cycle
is needed (§3.1). Tag `v0.2.0` at the end of Phase 2 to prove the pipeline;
`v1.0.0` when the API stops moving. `CITATION.cff` (validated in CI) plus the
Zenodo DOI give the project a citable artefact, which it currently lacks
entirely.

### 6.6 Branch layout and preserving the pre-refactor state

**`master` is frozen; `main` is the trunk.** `main` was branched from `master` at
`453c77c` and is where all refactor work lands. `master` is never committed to
again — it becomes the permanent, named record of the pre-refactor code, doing
the job a `v0.1.0` tag would have done.

To make that real rather than aspirational, two GitHub settings changes are
needed and neither can be made from a git client:

1. **Switch the repository default branch to `main`** (Settings → Branches), so
   PRs target it by default and clones land on it.
2. **Protect `master`**: no pushes, no force-pushes, no deletion. A branch is a
   *movable* ref where a tag is not, so without protection "frozen" is a
   convention rather than a guarantee — one absent-minded `git push origin
   master` and the record is gone.

Optionally also tag `453c77c` as `v0.1.0` for good measure; belt and braces, and
it gives release-please an explicit floor to generate the first changelog from.
The two are complementary — the tag is immutable, the branch is discoverable.

> Note on tooling limits: tag pushes are blocked in the automation environment
> this plan was written in (HTTP 403 on `push origin v0.1.0`, while branch
> pushes to the same remote succeed), which is why `main` exists as a branch
> rather than the state being pinned by a tag. **This does not affect the release
> automation in §6.4** — release-please creates tags from inside GitHub Actions
> using the workflow token with `contents: write`, which is a different and
> unrestricted credential. Automated versioning is unaffected; only ad-hoc tag
> pushes from this environment are.

Worth being precise about what any of this does and does not preserve: it
preserves the **code**, but the code is not **runnable** — `mtspec` no longer
builds on any current toolchain (§2.1). Reproducing a 0.1.0 result needs three
things, which is what Phase 0 delivers:

1. `master` (and optionally `v0.1.0`) — the source.
2. A `Dockerfile` pinning `gfortran`, `python 3.9`, `numpy<2` — the environment
   that can still build `mtspec`. Also the only place existing `.spec` pickles
   can be read (§4.6).
3. The golden `.npz` snapshots — the outputs, for when even that image stops
   building.

The source ref alone is the weakest of the three. Archaeology gets harder every
year.

**Commit convention.** Conventional Commits (`feat:`, `fix:`, `refactor:`,
`docs:`, `test:`, `chore:`; `feat!:` or a `BREAKING CHANGE:` trailer for
breaks), enforced by a `commitlint` pre-commit hook. This is what makes the
changelog and version bumps automatic.

**PyPI name.** `specmod` is unregistered (checked: 404 on `specmod`, `spec-mod`
and `pyspecmod`). Worth claiming with the `v0.2.0` release at the end of Phase 2
rather than waiting for 1.0 — name squatting on PyPI is real and the name is a
fairly obvious one.

---

## 7. Phasing

Each phase ends green on CI and is independently mergeable.

| Phase | Work | Depends on | Rough size |
|---|---|---|---|
| **0. Safety net** | Freeze `master`, default branch → `main`, optional `v0.1.0` tag (§6.6); reproducible legacy env (`Dockerfile` + gfortran + `numpy<2`); write `datasets/magna_2020.toml` and a first cut of `specmod.acquire`, publish the artifact as a `data-v1` release asset (§5.2); capture golden outputs for PNR **and** Magna; reproduce the published Magna values with 0.1.0 (§5.2.5 step 2); convert any `.spec` files (§4.6) | — | 1.5–2 days |
| **1. Make it installable** | `pyproject.toml` + hatch-vcs, `src/` layout, `__init__.py`; ruff config, one-shot `ruff format` + `.git-blame-ignore-revs`, module renames to snake_case; mypy skeleton; pre-commit; `test`/`build` CI; `.gitignore`, `CITATION.cff`; fix the three hard breakages (§1) and the four `F821` bugs ruff finds (§2.5); delete `Tests/Tutorial/`, strip notebook outputs, subset the inventory (§5.1) | 0 | 3–4 days |
| **2. De-globalise** | `Settings` dataclasses passed explicitly; remove all module-level config reads (tracked by `PLW0603`); `Motion`/`AmplitudeKind` enums; `Spectrum` as a frozen dataclass with `duration`; mutable class attrs (`RUF012`); `isinstance` checks; `logging`. **Tag `v0.2.0`** | 1 | 3–4 days |
| **2b. Release plumbing** | Sphinx skeleton + `pydata-sphinx-theme` + autodoc/napoleon/intersphinx; `docs.yml` → GH Pages; release-please + `publish.yml` (PyPI Trusted Publishing); Zenodo webhook. Parallel with 2 | 1 | 1–2 days |
| **3. Transform layer** | `SpectralEstimator` protocol; `FFTEstimator`, `WelchEstimator`, `MultitaperEstimator`; `smoothing/` incl. Konno–Ohmachi and `LogBinner`; mtspec demoted to optional legacy backend; Tier 1 + Tier 2 tests; theory docs page | 2 | 5–7 days |
| **4. CWT** | `CWTEstimator` + `Scalogram`; COI handling; the Parseval/units calibration and its test; `time_average()`; `ScalogramQC` + the four QC checks; COI floor into `BandwidthSelector`; scalogram plotting; HDF5 scalogram storage | 3 | 6–8 days |
| **5. Decompose** | Split `Spectral.py` (655 lines) into `core/` + `snr/`; `Fitting.py` → `fitting/`; models as objects; `io/`; `viz/`; non-mutating operations; mypy override list → empty | 3 | 4–6 days |
| **6. Ship** | Full docs content, tutorial rewritten as an executed `myst-nb` page with no `os.chdir`, 0.1→1.0 "what changed" page, **1.0 release** | 4, 5 | 2–3 days |

Phases 2b, and later 4 and 5, can run in parallel with their siblings.

Rough total: **5–7 weeks** of focused work — up from the previous estimate,
mostly Phase 4 (the scalogram, QC checks and their storage roughly double that
phase's surface) plus the docs and release infrastructure now scoped in 2b.

Phases 0–3 alone (≈2.5 weeks) get the package installable, formatted,
type-checked, tested, mtspec-free, documented and publishing to PyPI. That is the
point at which the project stops decaying, and it remains the sensible place to
cut if time runs short — the CWT is the one genuinely new capability and it
depends on nothing after Phase 3.

Note the ordering choice: **release plumbing lands at 2b, well before there is
anything worth releasing.** Publishing infrastructure is much easier to debug
against a trivial package than against a finished one, and having `v0.2.0` go out
end-to-end proves the pipeline while the stakes are zero.

---

## 8. Decisions

### Settled

- **CWT output** — both. The full scalogram *and* the time-averaged `Spectrum`
  from one transform (§4.4.1). Time-averaged feeds the fitting pipeline; the
  surface is a QC gate (§4.4.2), not persisted by default (§4.4.3).
- **Backwards compatibility** — clean break. No downstream users, so the 0.x API
  is removed outright: no `legacy.py`, no deprecation cycle (§3.1). Breaking
  changes expected throughout `0.x`; `1.0` when the API settles.
- **Commit convention** — Conventional Commits, `commitlint`-enforced, driving
  release-please (§6.4).
- **Data acquisition** — a general config-driven grabber (`specmod.acquire`),
  not a per-event script; artifacts pinned by SHA256 and served from GitHub
  Release assets via pooch (§5.2).
- **Tooling** — ruff for lint and format, mypy staged to strict, Sphinx for docs,
  automated versioning and publishing for both docs and package (§6).
- **Branch layout** — `master` frozen as the pre-refactor record, `main` as the new trunk (§6.6). One of the three
  preservation layers in §6.5.

### Still open

1. **Default estimator.** Multitaper (matching current behaviour) or FFT +
   Konno–Ohmachi (faster, more conventional in engineering seismology)?
2. **Python floor.** 3.11 is proposed. Any users stuck on 3.9/3.10?
3. **History rewrite.** Deleting the 9.9 MB catalog and stripping the notebook
   (§5.1) shrinks the *working tree* but leaves both in history, so a fresh clone
   still pulls ~15 MiB. Rewriting history with `git-filter-repo` would recover it
   — but it rewrites every SHA, which moves what `master` and `main` currently
   point at and invalidates any `v0.1.0` tag. Given the end state is a ~1 MB repo
   either way, **the recommendation is to leave history alone**: a one-time 15 MiB
   clone is a much smaller cost than an unrecoverable pre-refactor record. Only
   worth revisiting if the history genuinely becomes a burden.
4. **Magna config contents.** The grabber is general (§5.2.1), so this is no
   longer a code question — but `datasets/magna_2020.toml` still needs the
   paper's station list, distance range, window definition and pick source, and
   the published values to compare against. Step 2 of §5.2.5 only works if the
   0.1.0 re-run uses the same inputs the paper did.

---

## 9. References

- Edwards, B., Allmann, B., Fäh, D., Clinton, J. (2010). Automatic computation of
  moment magnitudes for small earthquakes. *GJI* 183(1), 407–420.
- Thomson, D.J. (1982). Spectrum estimation and harmonic analysis. *Proc. IEEE*
  70(9), 1055–1096.
- Prieto, G.A. (2022). The multitaper spectrum analysis package in Python.
  *SRL* 93(3), 1922–1929.
- Torrence, C., Compo, G.P. (1998). A practical guide to wavelet analysis.
  *BAMS* 79(1), 61–78.
- Konno, K., Ohmachi, T. (1998). Ground-motion characteristics estimated from
  spectral ratio between horizontal and vertical components of microtremor.
  *BSSA* 88(1), 228–241.
