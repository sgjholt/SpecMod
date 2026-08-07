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
  `MODEL` is bound once at import. **Fixed** — `models.py` is deleted and
  models are values resolved through `specmod.sources`; see §4.6.5.
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
| `PreProcess.py:176` | `cut_s(bf=2, ...)` — `bf` is never referenced in the body. A dead parameter that silently does nothing, on the function the published Magna workflow depends on. |
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
  config/                  # one module per semantic group (§4.7)
    layers.py              # defaults -> specmod.toml -> *.local.toml -> env -> kwargs
    provenance.py          # resolved config + hash + version, stamped into outputs
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
  sources/                 # BUILT — named `sources` so it does not collide
    source.py              #   with the legacy `models.py` during migration
    attenuation.py         # ConstantQ / FrequencyDependentQ
    motion.py              # displacement -> velocity/acceleration scaling
    composite.py           # SpectralModel + build_model + from_config
  fitting/
    fitter.py              # lmfit wrapper
    guess.py               # was ModelGuess.py
  preprocess/
    windows.py, picks.py, geometry.py
  io/
    hdf5.py                # versioned array layout; no Python class identity
    tables.py              # Parquet for fit results and catalogues
    asdf.py                # optional export for interchange (§4.6.3)
    schema.py              # format version + migration
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

#### 4.4.4 Evaluate PyWavelets against the hand-written transform

**The one unexplained reproducibility residual is `cwt`, and `cwt` is the one
estimator implemented from scratch here.** That coincidence is worth acting
on, even though it is not yet evidence.

The transform was written rather than delegated because the normalisation is
the whole difficulty: CWT coefficients carry `[signal] * sqrt(time)` under L2
and `[signal]` under L1, neither of which is the `[signal] * s` of a Fourier
amplitude spectrum. Get it wrong and the corner frequency is right while
`Omega` — and so `M0` — is wrong, with nothing to catch it. `pywt.cwt`'s
conventions are not documented to that precision, so using it meant
reverse-engineering them.

That reasoning still holds for *trusting* pywt blindly. It does not hold
against *comparing* to it, and it does not hold against reverse-engineering the
convention either — we have five estimators held to a Parseval contract, so the
constant is **measurable** rather than needing documentation.

**Measured, so the next person does not have to.** Calibrating
`pywt.cwt` with `cmor1.5-1.0` against known-energy records:

| Signal | recovered / true energy |
|---|---|
| white noise (calibration set) | 1.0000 |
| white noise (held out) | **0.9964** |
| sinusoid, 8 Hz | 1.3732 |
| sinusoid, 20 Hz | 1.2285 |
| decaying burst | 1.3398 |
| red noise | **0.0795** |

Two conclusions, and the second is the load-bearing one.

The convention *is* recoverable: the fitted reconstruction factor comes out at
`C_delta = 0.170` against the 0.776 Torrence & Compo give for the Morlet, a
factor of 4.6 — precisely the undocumented difference, now a number rather
than an unknown. And with the correct functional form
`E = (dj·dt/C_delta)·ΣΣ|W|²/s` — note the division by scale — it transfers
across held-out white noise at 0.9964.

But **a constant calibrated on one signal type does not transfer to another**:
23-37% out on sinusoids and bursts, and a factor of 12 out on red noise. The
first attempt here was worse still — calibrating a single scalar on a
sinusoid's peak, omitting the `1/s` weighting, gave a perfect-looking fit on
the sinusoid and was **50x** out on white noise. A wrong CWT normalisation
looks entirely reasonable, which is the whole hazard.

So the work is not "fit a constant". It is to derive the reconstruction factor
against the actual scale grid, the way `CWTEstimator._c_delta` already does,
and validate across signal character rather than on one record. That is what
makes our own implementation robust to `dj`, and pywt would need the same
treatment before it could be trusted with `Omega`.

**Be careful about the expected outcome, though.** Two things argue against
"switch to pywt and the residual goes away":

1. `cwt`'s *signal* amplitudes and frequency axis are already exact on every
   runner. Only the post-rotation *noise* moves. Whatever is happening is
   downstream of the transform, or at least not visible in it.
2. PyWavelets is numpy-based too, so it inherits the same floating-point
   behaviour. There is no reason a different implementation of the same maths
   would be immune.

So the honest framing is: evaluate pywt because independent cross-validation of
the CWT normalisation is valuable regardless, and *check whether* it also
moves the residual — rather than adopting it expecting a fix.

Note also that `wavelet = ["PyWavelets>=1.5"]` is already declared as an
optional extra and **nothing imports it**. Either this work wires it up or the
extra should be removed; a declared dependency that does nothing is a false
signal about what the package needs.

### 4.5 SNR, bandwidth and noise rotation

`find_optimal_signal_bandwidth` and `find_optimal_signal_bandwidth_2` become
`BandwidthSelector` strategies selected by argument, not by `BW_METHOD` global.
Both gain a **resolution floor** on the low-frequency end — the COI limit when
the spectrum came from a CWT, `~1/T` otherwise (§4.4.2). At present the selection
is purely amplitude-based and can return a usable band extending below what the
window length can resolve.

`rotate_noise_full` and `non_lin_boost_noise_func` are now `core/noise.py`, as
registered `NoiseModel` implementations — `boost`, `rotate` and `none` —
selected by name rather than by the `ROT_METHOD` integer. Both are solved
rather than iterated, so neither has an iteration limit to print
`"Didn't ever meet."` from and silently disable the correction. The originals
are deleted.

#### 4.5.1 Status: stage 1 partially landed

`core/collection.py` and `core/noise.py` now exist and the legacy path uses
them for binning and rotation. What is done, and what is deliberately not:

| Piece | State |
|---|---|
| `log_bin` | The only implementation. `spectral.Spectrum.__bin_spectrum` is deleted. |
| `boost_noise` (`ROT_METHOD = 2`) | Shared, and solved rather than stepped (§4.5.2). Verified bit-identical against the legacy `SNP.rotate_noise` over 200 randomised cases before the discontinuity fixes. |
| `parseval_scale`, `interpolate_onto` | Shared. The legacy copies are deleted. |
| `find_bandwidth` | Shared, as a registry — `peak` (was `BW_METHOD = 2`, the default) and `widest`. |
| `rotate_noise_full` (`ROT_METHOD = 1`) | **Ported** as `noise.RotateNoise`, registered as `"rotate"`. `SNP` no longer raises: `ROT_METHOD = 1` runs end to end and gives materially narrower bands than `boost` on 25 of the 28 PNR windows, consistent with the legacy source's own description of it as "quite aggressive". See §4.5.3. |
| `SpectrumPair` | **`SNP` delegates to it.** Its rescale, interpolation, rotation, ratio and band search are gone — 171 lines removed. |
| `SpectrumSet` | `Spectra.as_spectrum_set()` converts without recomputing — each `SNP` keeps the pair its numbers came from. The golden reference and its generator read the new container; regenerating produced a **byte-identical** file, which is the proof the swap moved nothing. `Spectra.group` still holds `SNP` for the smoke test, which exists to cover the legacy seam. |
| `pipeline` | **Waveforms to `SpectrumSet` with no legacy object in between** — see §4.5.5. The conversion path is no longer the only way to get one. |

The safety net for all of this is `tests/golden/pipeline_reference.json` —
digests of amplitudes, noise, SNR and selected band over 28 real windows and
5 estimators, regenerated with `tools/make_golden.py`. It was checked to
actually bite: a 1-part-in-1e12 perturbation fails all 28.

**Why `find_bandwidth` is not wired in.** It returns `None` when the search
fails. The legacy returns a band anyway and sets `pass_snr` beside it, so a
caller that reads the band without checking the flag gets numbers that look
like a measurement. `None` is the better contract, but adopting it changes
what the legacy path returns for a failing station — a behaviour change that
needs the golden reference regenerated deliberately rather than a quiet swap.

#### 4.5.2 Three discontinuities made the noise and band machine-dependent — fixed

Found by running the golden reference on CI, where identical code on a runner
matching the reference machine exactly — same system, arch, Python, numpy
2.4.6, scipy 1.17.1 — produced noise levels **41% and 82% apart**. Not
floating-point sensitivity: perturbing the input by 1e-13 moved nothing at all.
The pipeline was *piecewise constant*, and runners landed on different pieces.

| Where | Was | Now |
|---|---|---|
| `boost_noise` | Stepped the exponent by `inc = 0.05` and stopped at the first step past the touching point. One iteration either side = **1.41x** on the noise. | Closed form: `n = min ln(signal/noise) / -ln(sample)`, used exactly. Continuous in its input. |
| `log_bin` | Tested `f >= left and f <= right` per edge. Both ends closed, so a sample on an edge joined **two** bins, and which one depended on the last bit of `np.logspace`. | Index computed from position: one sample, one bin, no edge comparison. |
| `find_optimal_signal_bandwidth` | `sign(bsnr - tol)`, integrate, read 1st/99th percentiles, retry when edges cross. Moved an edge **13 bins**. | Widest contiguous run above threshold, bridging single-bin dips. |
| `boost_noise` again | `if np.any(noise >= signal): return unchanged` — a guard, and a guard is a step. Jumped the median boost factor by **17.5%** as a bin crossed. | `max(0, min(needed))`. A bin above the signal needs a negative exponent, so clamping gives the same answer continuously. |

The fourth was found only after the first three landed: CI went from five
failing checks to one, and the survivor was `cwt` alone, disagreeing by 1-2%
on four stations. Small enough to look like tolerance, but it was another
branch — the "already touching" guard inside the lift.

**Measured after the fix**, end to end over all 28 windows: perturbing the
input by 1e-15 moves the noise by 1.8e-11 and **no band edge at all**. The
response is linear. The golden reference's exact noise comparison, which had
to be made opt-in because it could not survive a change of machine, now runs
unconditionally again.

Two of the three also removed a bias rather than merely a wobble:

- The rotation always rounded **up**, so it consistently overstated the lifted
  noise — a median **1.18x** and up to **1.41x** across 39 lifts. That made
  signal-to-noise pessimistic at exactly the band edges the ratio is read from.
- The band search's low edge lagged the true onset by **4.4 Hz** on a clean
  5-30 Hz test case, because the 1st percentile of a cumulative integral
  arrives late. It now lands within one bin. The low edge is what constrains
  `Omega`.

**What moved on the real data.** No station lost its band under any estimator.
Bands widened on 2 (fft), 1 (welch), 8 (multitaper), 8 (quadratic) and 14
(cwt) of 28 windows; median edge movement was 0.000 Hz except the cwt high
edge at +1.195 Hz. `bsnr` arrays shortened where the old binning had been
double-counting — for CWT the old rule reported *more bins than there were
samples*, which is only possible by counting a sample twice.

Anyone reproducing a pre-refactor result needs the reference regenerated and
should expect slightly wider bands and slightly lower noise. That is the
correction, not a regression: the old numbers were biased in a known
direction.

Two things to fix in the same change, since they are both about that function:

1. **The low edge lags.** On a clean 5–30 Hz passing region the selected band
   is 9.4–28.7 Hz. Reading percentiles off the integrated sign function costs
   roughly 4 Hz at the low end, and the low edge is what constrains `Omega`.
   `tests/test_collection.py` pins the current values so the improvement is
   measurable against them.
2. **The failure contract**, as above.

**Also worth knowing.** The legacy applies rotation to `bamp` directly but to
`amp` by interpolating the factor onto the finer axis. Those two are therefore
*not* related by the binning operation — re-binning the rotated `amp` does not
reproduce `bamp`, differing by about 6% on real data. Whichever becomes
canonical, it should be one of them and not both.

#### 4.5.3 `rotate` ported: the noise registry is complete

`ROT_METHOD = 1` was the last unported piece of the noise machinery, and the
one `SNP` raised `NotImplementedError` for. It is now `noise.RotateNoise`,
registered as `"rotate"` alongside `"boost"` and `"none"`, and
`SNP.__compare` maps the legacy integer onto the registry instead of refusing.
`SpectrumPair.compare` takes a `noise_model` rather than hardcoding the boost.

**Why porting it was cheap.** `ROT_METHOD = 1` is *commented out* on `master`
— it has never been the default and has never produced a published number. So
unlike every other piece of this migration, there was nothing to preserve, and
two changes could be made that would otherwise have needed a deliberate
reference regeneration:

* **The angle is solved, not stepped.** The legacy advanced `theta` by a fixed
  `inc` and stopped at the first trial past the touching point — the identical
  defect that made `boost` machine-dependent (§4.5.2). `RotateNoise` brackets
  the touching angle on a coarse grid and bisects to `1e-12`, so the answer is
  continuous in the input and independent of the grid. Both are asserted:
  perturbing the signal by 1 part in 1e12 moves the factor by less than 1e-8,
  and changing the bracket resolution from 128 steps to 37 changes nothing.
* **The low/high split comes from the signal.** The legacy took the *noise*
  centroid here and the *signal* centroid in the boost path. The signal is the
  defensible choice — the split should follow where the energy is, and the
  noise has no reason to share that shape — and using it in both is what makes
  the two bands comparable rather than merely different.

**What it does to the science.** On the 28 PNR windows, `rotate` moves the
selected band on **25 of 28**, always narrower: `LV.L002..HHE` goes from
1.51–38.60 Hz under `boost` to 1.51–13.87 Hz. That is a large enough difference
that the choice has to be an explicit, recorded one — which is the argument for
a registry over a flag, and for keeping `none` as the control both are read
against.

**A claim withdrawn.** The first draft of the module docstring asserted that
rotation, being non-monotone, can return a factor below one. That was reasoning,
not measurement, and the measurement disagrees: over 4000 randomised spectra
spanning three decades of frequency, slopes from `f**-3` to `f**1` and noise
from 0.1% to 99% of signal, the smallest factor observed was `1 - 2e-15`. There
is no proof here that it cannot happen; the registry-wide test asserts the
property, and if a real spectrum ever violates it the honest response is to
relax the claim rather than clamp the method.

Also found while porting: the legacy "Didn't ever meet." fallback is nearly
unreachable. The anchor term `log_amp[0] * theta` grows without bound with the
angle, so a backward rotation lifts the noise arbitrarily far — even thirty
decades below the signal it still touches, exactly, well inside a quarter turn.

#### 4.5.4 `utils.py` reduced to what still has callers

With the rotation machinery moved, `utils.py` went from 446 lines to 206.
Deleted: `find_rotation_angle`, `find_rotation_angle_v2`, `rotate`,
`rotate_noise_full`, `get_centroid_freq` and `non_lin_boost_noise_func`
(superseded by `core/noise.py`), and `DataSet` with its channel-ranking helpers
`getchan`/`rank_chans`/`compare_ranks`/`get_avail`, plus `cps` and
`path_to_utc` — none of which had a single caller anywhere in the package,
the tests, the tutorial or the studies, and all of which were hardcoded to
another study's station codes. The acquisition layer in §5.2 replaces them.

`non_lin_boost_noise_func` is worth a note on its way out: it **had no return
statement**. It computed both lifted arrays and returned `None`. Whatever the
legacy `SNP` was doing, it was not calling this.

What remains is characterised by `tests/test_utils.py`: `read_pyrocko`, which
is upstream of both golden references and so needs the same treatment
`preprocess` got, plus the catalogue readers and `stream_distance_sort`.
`plot_traces` is left untested — it draws, and a test that only asserts it does
not raise would pass just as happily on a broken figure.

Two defects pinned there, unfixed: `read_pyrocko` does `readlines()[1:]`
without checking for the Snuffler magic, so a marker file lacking the header
line silently loses its first pick; and a station with two picks of the same
phase — easy to produce by picking on more than one component — silently
resolves to whichever appears last in the file.

One fixed: `cat2kstyle` dropped sub-second precision with a fixed `[:-3]`
slice, which assumed exactly two decimal places on the seconds. Three decimals
raised `invalid literal for int()`; **no decimals lost the seconds entirely**,
quietly returning 13:09 for 13:09:31. The second is the one that mattered, and
splitting on the decimal point handles every case. This was the fragility
`tests/test_legacy_fixes.py` deliberately deferred to "the preprocessing
rewrite"; that test now asserts the fix across five seconds formats.

#### 4.5.5 `specmod.pipeline`: the direct route from waveforms

`Spectra.as_spectrum_set()` converts *after the fact* — it needs a legacy run
to exist first. `specmod.pipeline` removes that dependency: given a signal
stream and its noise it builds the immutable containers directly, trace to
`core.Spectrum` to `SpectrumPair`, with none of `spectral`'s mutable classes
involved.

**The route is shorter.** `spectral.Spectrum` converts the estimator's output
to a PSD, copies the arrays out into a mutable object, converts back to
magnitude, and then `SNP.__as_core` wraps them in a `core.Spectrum` again —
`FAS -> PSD -> MAGNITUDE -> copy -> copy` where one `to_kind` will do. Every
one of those copies exists because the legacy classes mutate in place; none is
needed once nothing does.

**It lives outside `core` on purpose.** `core` and `transforms` know nothing
about ObsPy — they take arrays, a duration and a sampling rate — which is what
makes them testable without constructing a Stream and usable on data that never
came from one. `pipeline` is the single file that knows what a `Trace` is.

`estimate_spectrum` moved here from `spectral` at the same time, and is now
typed. It is the one bridge to `transforms`, and leaving it inside the untyped
legacy shell made every caller of it untyped too. Its field selection now goes
through `inspect.signature` rather than `dataclasses.fields`: the registry is
typed over the `SpectralEstimator` protocol, which promises a `name` and an
`estimate` and says nothing about being a dataclass, so asking for its fields
was a claim the type did not support.

**Agreement is measured, not argued.** `tests/test_pipeline.py` runs both paths
over the same 28 windows and all five estimators — 140 results — and requires
the spectra, binned spectra, SNR, resolution floors and selected bands to match
to **1 part in 1e12**. They are not bit-identical, and should not be expected
to be: the same map composed differently rounds differently.

It also runs the **fitter** on both, because that is where a difference would
be amplified rather than carried. Powell on a shallow minimum turns the 1e-12
input difference into 7.5e-6 relative on `fc` — 4e-5 Hz on a 5 Hz corner, 2e-5
in stress drop. The tell that the spectra really are the same is that `chisqr`,
`redchi` and `bic` agree to 1e-11: the two runs land on marginally different
points of an identical surface.

**One gap the comparison found.** The legacy flatfile carries `lower-f-bound`,
`upper-f-bound` and `pass_snr`, written by `SNP.__update_lims_to_meta`. A
frozen pair cannot write back into its own signal — that is the point of
freezing it — so `FittableView` supplies them instead, under the legacy column
names so a flatfile from either container has the same schema. A fitted corner
frequency without the band it was read over is not interpretable, and it is the
first thing anyone comparing two runs asks for.

The new path also carries what the legacy one never did: `estimator`, `id` and
`resolution_floor` per row.

**What is left tying the fitter to `spectral`.** `FitSpectra.__check_spectra`
required `isinstance(spectra, sp.Spectra)` while only ever iterating and
indexing, so a `SpectrumSet` was refused despite satisfying everything the
fitter uses. It now checks for the mapping interface. `fittable_signal` returns
a pair's `FittableView` rather than its `signal`, since the fitter wants
`freq`, `amp`, `bfreq` and `bamp` side by side and the pair keeps the unbinned
and binned forms separate — rightly, for the comparison.

`FitSpectra(spectrum_set_from_streams(sig, noise))` now works end to end.

#### 4.5.7 `spectral.py` deleted

864 lines gone: `Spectrum`, `Signal`, `Noise`, `SNP`, `Spectra`, the pickle
I/O and the `_mtspec` shim. `specmod.pipeline` and `specmod.core` do all of it,
and `fitting` no longer imports the module at all.

**Two capabilities had to be built before it could go**, both found by trying:

* **Domain conversion on an event.** `Spectra.inte()`/`diff()` had no
  equivalent — `core.Spectrum.to_motion` converts one spectrum, and nothing
  converted a pair. `SpectrumPair.to_motion` and `SpectrumSet.to_motion` now
  do, returning new objects rather than overwriting, so an event can be held
  in two domains at once. To replay the binning and the band search they need
  the settings the pair was built with, so `compare` records its own arguments
  in `meta` under `SpectrumPair.SETTINGS_KEY`. That is also how the noise
  avoids being lifted twice: the replay sets `rotate_noise=False`, which is
  the `ROTATED` flag expressed as data rather than as a mutable attribute.
* **Persistence.** Deleting the pickle I/O left nothing able to write a result
  to disk — and `SpectrumSet` **could not be pickled at all**, because
  `core.Spectrum.meta` is a `MappingProxyType`. `Spectrum.__reduce__` now
  rebuilds through `__init__`, which also re-freezes the arrays: numpy restores
  a pickled array as writeable, so any other route would have returned a
  mutable spectrum and quietly lost the guarantee the class exists to make.

**What happened to the evidence.** Deleting the legacy path also deletes every
test that compared against it — `test_the_pair_reproduces_the_legacy_snr_and_band`,
the 140-result comparison in `test_pipeline.py`, `Spectra.as_spectrum_set()`'s
identity checks. That is a real loss, and the honest account of what carries
the claim now is:

`tests/golden/pipeline_reference.json` was generated by the legacy path and
**has never been regenerated**. `specmod.pipeline` reproduces it to **1e-15**
across all 140 window-estimator results, checked on every run. That is why the
reference is summaries of a committed artefact rather than a live comparison:
it is the form of the claim that survives the code being removed.
`tests/golden/motion_reference.json` is the same pattern for `to_motion`,
captured while `Spectra.inte()` still existed and validated against it at
2.2e-15 with identical bands on all 28.

**One test now asserts the opposite of what it did.** `test_pipeline_smoke.py`
checked `spectrum.amp.flags.writeable`, because the legacy classes mutated in
place and a frozen array reaching them raised. Nothing mutates, so the property
worth protecting is the one that used to break the pipeline.

`_mtspec` went with the module. Nothing called it, and `estimate_spectrum`
already refuses `estimator="mtspec"` with a message pointing at `prieto` — the
same lineage without a Fortran compiler. A §5.2.6 three-way comparison would
install mtspec and call it directly rather than through a shim.

#### 4.5.6 Two live breaks found while retiring the shells

Both in the domain-change path — `Spectra.inte()` and `Spectra.diff()`, which
move a whole event between displacement, velocity and acceleration. Neither
was covered by anything, which is why neither had been noticed.

**`SNP.integrate` and `differentiate` raised `AttributeError`.** They called
`self.__get_snr()`, which stopped existing when `__compare` replaced it; the
call sites were not updated. This fired for every *interpolated* pair, which
is the default and every pair the pipeline builds — so the domain-change API
was simply broken, not degraded.

Fixing it meant restoring something the pre-refactor code had and the rewrite
dropped: the `ROTATED` flag. The noise lift is applied **once**, and
`self.noise.amp` already carries it on any re-comparison, so re-running
`__compare` unconditionally would lift the noise again on every domain change
— compounding, narrowing the band each time. `master`'s `__calc_bsnr` guarded
on `self.ROTATED == False` for exactly this.

**The domain label never moved.** `spectral.Spectrum.integrate` divided `amp`
and `bamp` by `2*pi*f` and left `self.motion` saying `velocity`. Not cosmetic:
`SNP.__as_core` reads it to build a `core.Spectrum` and `_convert` reads it
for every amplitude-convention change, so a spectrum integrated to
displacement reported the wrong unit and converted as though it were still a
velocity. It now shifts with the amplitudes, and moving past either end raises
rather than clamping — silently stopping would leave `amp` divided by
`2*pi*f` with nothing to say so, which is the failure being fixed.

**A claim corrected by measurement.** The first version of the fix said the
band has to be recomputed because integration changes the signal-to-noise
ratio. It does not: both spectra are divided by the same `2*pi*f`, so the
*unbinned* ratio is exactly invariant. The *binned* one is not, and the reason
is worth stating — a bin holds the geometric mean of `log10(amp)`, and
averaging `log10(a/f)` over a bin is not `log10(a)` averaged minus
`log10(f_centre)` unless the centre is the geometric mean of the frequencies
in that bin. Measured over the 28 PNR windows the binned ratio moves by up to
**16%**, and **3 of 28** bands with it. So the re-comparison is doing real
work, just not the work first claimed for it.

Also deleted: `SNP.find_optimal_signal_bandwidth`, the percentile-of-the-sign-
integral search (`BW_METHOD = 1`). Superseded by `core.bandwidth.WidestBandwidth`
and referenced by nothing but documentation.

### 4.6 I/O — replacing pickle

> **Landed.** `specmod.io` (HDF5) and `specmod.tables` (Parquet) implement
> this, `specmod[io]` carries `h5py` and `pyarrow`, and **pickle is gone from
> the package entirely** — the dead `Tutorial/Spectra/*.spec` artefact is
> deleted and `tests/test_legacy_fixes.py` walks the AST of every module to
> assert nothing imports `pickle`, `cPickle` or `dill` again. See §4.6.6 for
> what the implementation measured.

Pickle is the only persistence format today, and `read_spectra` prompts on stdin
before unpickling. It fails on five counts at once, which is worth stating
because they point at different replacements:

| Problem | Consequence |
|---|---|
| Executes arbitrary code on load | Cannot safely accept a `.spec` from a collaborator |
| Stores class *import paths* | Renaming a module breaks every existing file (§4.6.2) |
| Python-only | Colleagues on MATLAB/Julia/R cannot read the outputs |
| Opaque | Cannot inspect, diff, or query without SpecMod itself |
| No schema version | No migration path — exactly how it got into this state |

#### 4.6.1 Two access patterns, so two formats

One format for everything is the wrong instinct here, because the data is used
two genuinely different ways:

1. *"Give me the spectrum for event X, station Y"* — random access into float
   arrays, occasionally large (a scalogram is ~1 MB per trace, §4.4.3).
2. *"Give me `f_c` and Ω for all 635 events and regress them"* — a columnar scan
   over a table. The published Magna run produced **11,226 rows** of exactly
   this, and a multi-event catalogue is bigger.

**Arrays → HDF5** (`h5py`), with a documented, versioned layout of our own.
Chunked, compressed, partial-read, cross-language, and the obvious scientific
default.

**Tables → Parquet** (`pyarrow`), replacing the CSV flat-file. This is a real
upgrade rather than a fashion: CSV loses dtypes, round-trips floats through
decimal text, and has to be read in full. Parquet is typed, compressed, and
queryable with DuckDB or polars **without loading it** — which matters at
11,226 rows and matters more at catalogue scale. CSV stays as an *export*, since
journal supplements want it.

**Provenance → both.** The §4.7 record goes into HDF5 attributes *and* a JSON
sidecar, because the sidecar is greppable and diffable without opening the
container.

#### 4.6.2 Rules the layout must follow

These are the lessons from how pickle failed, not general good practice:

- **Never store class identity.** Plain arrays plus a documented layout. Pickle
  broke because it recorded `specmod.Spectral` / `Spectra`; a schema that names
  no Python types cannot be broken by renaming one.
- **Every file carries `specmod_format_version`.** Readers check it and either
  migrate or fail loudly. The absence of this is the whole problem.
- **Self-describing units.** `motion`, `kind` and `duration` are stored
  attributes, not conventions the reader has to know. This is §4.2's typing
  expressed on disk, and it is what stops a file being silently misread as
  displacement.
- **One file per event, group per channel.** Matches how the science is done
  and how it is re-examined. It also sidesteps HDF5's single-writer limitation
  entirely if the 96,169-waveform workflow is ever parallelised.

#### 4.6.3 ASDF as an export target, not the primary

[ASDF](https://asdf-definition.readthedocs.io/) deserves consideration and is
the closest thing to a domain-native answer: HDF5-based, community standard,
embeds QuakeML and StationXML, carries SEIS-PROV provenance, and its docs
name time-dependent power spectral densities as an auxiliary-data use case.
`pyasdf` is maintained (0.8.2, August 2025).

It is still the wrong **primary**, for two reasons. ASDF is waveform-centric —
derived spectra live in the loose `auxiliary_data` bucket, so we would be
fitting our data to a schema built for something else. And SEIS-PROV models
*processing* provenance, not the *configuration* provenance of §4.7, so it does
not remove the need for our own record.

So: HDF5 with our own schema as primary, and `specmod export --format asdf`
behind a `specmod[asdf]` extra for interchange and archival. That keeps the core
dependency to `h5py` while still letting you hand a colleague a standard file.

#### 4.6.4 Considered and rejected

- **`.npz` + JSON.** Genuinely simpler and adequate for a single event's 1-D
  spectra. Rejected because scalograms are 2-D and large, and npz offers no
  chunking, no compression control, no attributes and no partial reads.
- **Zarr.** Attractive for parallel writes and object storage. **Zarr 3.x
  requires Python ≥3.12**, above this project's 3.11 floor, so it is not
  available without moving the floor. Revisit if that changes.
- **netCDF4 / xarray.** Labelled dimensions suit `(frequency,)` and
  `(scale, time)` nicely, but it is a heavy dependency and CF conventions do not
  describe spectra well. HDF5 attributes cover what is actually needed.

Implementation lands with `core/spectrum.py` in phase 3 — the writer serialises
that dataclass, so building it first would mean guessing at the schema.

**Existing `.spec` files will not survive the clean break, and this is not
optional.** A pickle stores the *import path* of every class it contains —
`specmod.Spectral` / `Spectra`. Once `Spectral.py` becomes `core/collection.py`
and `Spectra` becomes `SpectrumSet`, `pickle.load` raises
`ModuleNotFoundError` before any of our code runs. Renaming the modules is
enough to break them; the dataclass conversion in §4.2 would finish the job. One
such file is already committed at `Tutorial/Spectra/2019-08-26T07:30:47.0.spec`,
and there are presumably more in your working directories.

> **This has already happened — it is no longer a future consequence.** The
> `Spectral.py` → `spectral.py` rename alone was enough. The committed
> tutorial file is dead today:
>
> ```
> >>> Spectra.read_spectra("Tutorial/Spectra/2019-08-26T07:30:47.0.spec",
> ...                      method="pickle", skip_warning=True)
> ModuleNotFoundError: No module named 'specmod.Spectral'
> ```
>
> `Tutorial/SpecModTutorial.ipynb` calls exactly that in two cells, so the
> notebook cannot be run past them. The recommendation below is unchanged and
> still the right one; what has changed is that the conversion is now
> repairing something broken rather than pre-empting a break. Anyone reaching
> for the legacy Docker image for the §5.2.6 Magna comparison should convert
> this file in the same session — both need the same environment, and it is
> the only one still standing.

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

#### 4.6.5 Source models: widen the set when `models/` is built

Two source models exist today, `BRUNE_MODEL = (1, 2)` and
`BOATWRIGHT_MODEL = (2, 2)`, and neither is reachable from configuration —
see the note in §4.2 on `config.model.source` being wired to nothing. When
`models/source.py` is written, build it to hold more than the two, because at
least one more is wanted:

**Madariaga.** Requested explicitly. Madariaga (1976), *Dynamics of an
expanding circular fault*, BSSA 66(3), with the later review at
<https://www.geologie.ens.fr/~madariag/Papers/Madariaga_Ruiz2016.pdf>.

There is a trap in adding it, and it is the reason this note exists rather
than a one-line TODO. **Madariaga's difference from Brune is not primarily the
spectral shape.** Both are omega-squared; in the generalised Boatwright form

```
A(f) = Omega / (1 + (f/fc)^(gamma*n))^(1/gamma)
```

Madariaga sits at the same `(gamma, n) = (1, 2)` as Brune. What differs is the
constant relating corner frequency to source radius — Madariaga's dynamic
circular-crack solution gives a substantially smaller `k` than Brune's
kinematic one, and since stress drop goes as `r^-3`, choosing between them
moves inferred stress drop by roughly an order of magnitude on identical data.

So the design consequence is: `SourceModel` must carry the `fc`-to-radius
scaling as a named property of the model, not as a constant buried in whatever
computes stress drop. If it is only a spectral shape, adding Madariaga will
appear to do nothing — the fit will be identical to Brune — and the actual
difference will be silently lost. A model that changes no fitted parameter but
changes every derived one is exactly the kind of thing this refactor is
supposed to make impossible to get wrong by accident.

Worth checking against the source paper when implementing rather than taking
the above on trust: the `k` values differ between P and S and between authors,
and the plan should not be the citation of record for a number that ends up in
published stress drops.

#### 4.6.6 What the implementation measured

The four rules in §4.6.2 each became a test. Two things only showed up on
contact with real data.

**Compressing everything made the file three times larger.** The first version
gzipped every dataset, per the "chunked, compressed" argument above, and
produced **980 KB** for one 28-station event whose raw float payload is
**250 KB**. Uncompressed the same file is 370 KB. Compression in HDF5 requires
*chunked* storage, which costs a chunk index per dataset — and these arrays
have a median of 91 float64, so the indices cost more than the data they
describe. `specmod.io` now compresses only above 16 KB. That is where the ratio
flips, and it is the case the argument was really about: a scalogram is ~1 MB
per trace (§4.4.3), and those still compress.

**Two of the nine datasets per station were exact duplicates.**
`SpectrumPair.compare` interpolates the noise onto the *signal's* frequency
axis and bins both against the same edges, so `noise.freq == signal.freq` and
`binned_noise.freq == binned_signal.freq` hold **by construction**, on all 28.
Storing each twice was waste, and worse, made a file expressible in which the
two disagree — a state the containers cannot represent and no reader could act
on. Seven datasets per group now, with the two shared axes written once.

Together: 980 KB → **391 KB**.

**And an honest note on the comparison.** Pickle was 325 KB for the same event.
HDF5 is ~20% larger, and that is not a defect being hidden: 141 KB of the 391
is self-describing structure — a format version, per-spectrum units, duration
and sampling rate, trace metadata, and a browsable group per channel. That is
the cost of a file that can still be read after the classes are renamed, which
is precisely what the old one could not do. Parquet, by contrast, is a
straight win: 30 KB against CSV's 21 KB for the fit table, with dtypes
preserved — a CSV column carrying one `None` comes back as object-dtype
strings, so `pass_fitting` stops being boolean and a downstream `.sum()`
counts the wrong thing.

### 4.7 Configuration: semantic groups, layered overrides, recorded provenance

Scientific parameters are currently scattered across three places with no
coherent story: a `config.py` of three flat dicts, function defaults in
`PreProcess`, and values hardcoded past reach entirely — the multitaper
time-bandwidth product is the literal `3` at `Spectral.py:124`, not exposed at
all. The defaults have also drifted from the published run (§5.2.5), because
they were tuned for later studies. Both problems have the same fix.

**Principle: current behaviour stays the default; every study pins its own
config; the resolved config travels with the output.**

#### Semantic groups

One section per stage of the pipeline, each a typed dataclass, each owned by the
module that uses it:

| Section | Covers | Currently lives in |
|---|---|---|
| `[acquire]` | Data centre, event query, station/channel selection, radius | *(new, §5.2)* |
| `[windows]` | `p`/`s` group velocities, `bf`/`rafp`/`tafs`, `time_after`, `refine_window`, `pctls`, `bshift`, padding, `sta_shift`, `emergency_ratio` | `PreProcess` function defaults |
| `[transform]` | Estimator choice, taper, `nfft`/padding, time-bandwidth, `number_of_tapers`, `quadratic`, DC-bin handling | mtspec `**kwargs` + the hardcoded `3` |
| `[smoothing]` | Binning `smin`/`smax`/`bins`, Konno–Ohmachi bandwidth | `SPECTRAL.BIN_PARS` |
| `[snr]` | `SNR_TOLERENCE`, `MIN_POINTS`, `ASSERT_BANDWIDTHS`, `S_BANDS`, `SCALE_PARSEVAL`, `BW_METHOD` + its `pctl`, `ROTATE_NOISE`, `ROT_METHOD`, `ROT_PARS`, noise interpolation | `SPECTRAL` + buried kwargs |
| `[model]` | Source model, motion, γ/n | `MODELS` |
| `[fitting]` | Minimiser, `fit_bins`, weighting, bounds, initial guesses (`ts=0.01`, `a=1e-5`) | `FITTING` + `ModelGuess` |
| `[viz]` | `PLOT_COLUMNS` | duplicated in **both** `SPECTRAL` and `FITTING` |

Note the last row: `PLOT_COLUMNS` is defined twice today and the two copies can
disagree. Semantic grouping makes that structurally impossible.

`config.py` becomes a `config/` package, one module per section, so each group
lives next to the code it configures rather than in a central blob every module
imports.

#### Layering

Resolution order, lowest to highest:

1. **Package defaults** — typed dataclasses in code. **Set to today's shipped
   behaviour**, not the paper's (`s=2.9`, `bshift=0.2`,
   `ASSERT_BANDWIDTHS=False`, `ROTATE_NOISE=True`). Upgrading changes nothing.
2. **Committed project config** — `specmod.toml` at the repo root. This is what
   tutorials and published studies use.
3. **Local override** — `specmod.local.toml`, **gitignored by default**. Personal
   experimentation, never accidentally committed.
4. **Environment** — `SPECMOD_SNR__TOLERANCE=4` and similar, for CI.
5. **Explicit Python arguments** — always win.

Same schema and same loader as the acquisition config in §5.2.1, so one config
file can carry `[acquire]` *and* the processing sections. Fetch and process share
one provenance record rather than two that can drift apart.

#### Making "local and uncommitted" compatible with "reproducible"

These pull against each other, and the resolution is that **reproducibility comes
from what the output records, not from what the repository contains**:

- Every output — spectra, fit tables, HDF5 — carries the **fully resolved
  config**, plus a short **config hash**. Two runs that disagree get their hashes
  compared first. Same mechanism as the acquisition manifest (§5.2.2).
- The resolved config records the **SpecMod version**. Defaults may move between
  releases; without the version stamp, "reproducible" fails silently across an
  upgrade. This is precisely the failure that made the v0.1.1 archaeology in
  §5.2.5 so painful.
- `specmod config show` prints the resolved config **with the layer each value
  came from**, so "why did this run differ" is answerable in one command.
- `specmod config freeze > studies/my_study.toml` promotes a local config to a
  committable one. That is the explicit opt-in — local stays local until you
  deliberately publish it.
- Regression tests pin an **explicit config file**, never the defaults, so
  changing a default cannot silently move a golden test.

#### What this resolves

`studies/magna_2020_paper.toml` captures the published values — `s=3.4`,
`bshift=0.5`, `ASSERT_BANDWIDTHS=true`, and whatever noise-rotation setting the
0.1.1 re-run establishes. The paper's configuration becomes a committed artifact
rather than a question, and later studies that moved the defaults get their own
files alongside it. No archaeology, and every study reproducible from a named
file.

#### 4.7.1 `_config_legacy.py` deleted — the flat dicts are gone

`spectral.py` and `fitting.py` bound their settings at import from a
hand-maintained module of literals (`cfg.SPECTRAL["ROT_METHOD"]` and eleven
more), which duplicated the typed defaults and could drift from them silently.
Those reads now come from `Config`, and `_config_legacy.py` is deleted.

Every value was compared across the swap and is identical — the typed defaults
were written to reproduce the dict — so this changes provenance, not behaviour,
and the golden reference did not move. What it buys: a study's TOML layer now
reaches these settings instead of stopping at the module boundary, every one is
validated and recorded in provenance, and `PLOT_COLUMNS` has one home rather
than a copy in each of two dicts.

The globals stay module-level, because that is the escape hatch the legacy path
and its tests use, and `spectral` is a shell over `core` that should disappear
rather than grow a plumbing layer. `BW_METHOD` and `ROT_METHOD` survive as
integers derived from the config's *names* — the round trip is deliberate, so
that a caller setting `sp.ROT_METHOD = 1` still works while the name is what
the registry resolves. `tests/test_config.py` asserts the derivation in both
directions, because nothing else would notice if `BINNING_PARAMS` quietly
stopped tracking `smoothing.n_bins`.

One stale value found on the way: `SnrConfig.bandwidth_method` was typed
`Literal["integral", "peak"]` while the registry it names had `widest` and
`peak`. `"integral"` was left over from when the selector was a percentile of
a sign integral (§4.5.2) — a configuration value naming nothing the code would
accept. Corrected to match `BANDWIDTH_SELECTORS`, and `bandwidth_percentile`,
which only that method used, is removed.
---

## 5. Testing strategy

Proposed, in dependency order. Tiers 1, 3 and 4 are largely in place — 416
tests, 80% overall — and the module-by-module characterisation described in
§5.3 is what tier 4 turned into. Tier 2 is the outstanding one:

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
> `mtspec` still builds — a one-off Docker image with `gfortran` and the exact
> versions the paper pins (§5.2.6). Budget half a day; without it the refactor is
> unverifiable. Magna makes this stronger still, because published values exist
> for it — see the three-way comparison in §5.2.6.

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
| `Tutorial/Spectra/*.spec` | 333 KB | Convert (§4.6) or delete — it is regenerable, and **already unloadable** |

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

Source: Holt *et al.*, "Towards Robust and Routine Determination of Mw for Small
Earthquakes: Application to the 2020 Mw 5.7 Magna, Utah, Seismic Sequence",
submitted to *SRL*. The paper's **Spectral Modeling of Direct Sg Phases** section
specifies the workflow step by step, which makes `datasets/magna_2020.toml` and
the validation targets a transcription exercise rather than a guess. Everything
below is from the manuscript.

**Scope.** Mainshock 2020-03-18 13:09:31 UTC, Mww 5.7. Catalogue period
2020-03-18 → 2020-04-30, 2,103 UUSS-located events. 88 stations within a **400 km
radius** of the epicentre:

| Network | Count | Notes |
|---|---|---|
| UU | 73 | 42 broadband (100 Hz; 2 at 40 Hz) + 31 strong motion (100 Hz) |
| IW | 5 | Intermountain West |
| US | 4 | US National Seismic Network |
| IE | 3 | INL |
| N4, NN, RE | 1 each | CEUSN, Nevada, USBR |

Data from IRIS/EarthScope FDSN, accessed with ObsPy 1.2.0.

**The workflow, as published** — this is the config:

| Step | Parameter |
|---|---|
| Component | **Transverse** (horizontals rotated to R/T) |
| Motion | Ground velocity, response-corrected |
| Phase arrivals | Group velocities **Pg 5.9**, **Sg 3.4 km/s** (Pechmann *et al.* 2007) |
| Signal window | **20 s**, starting at **80% of the Pg–Sg time** |
| Window refinement | 1st and 99th percentiles of the cumulative squared-amplitude integral |
| Noise window | Ends **0.5 s before Pg**, same length as the signal window |
| Transform | Multitaper (Prieto *et al.* 2009) |
| Binning | Even bins in log10 space |
| SNR gate | **> 3** in **all** of 2–4, 4–6, 6–8 Hz |
| Model | Brune, velocity: `log10[2πf] + log10[Ω] − log10[1+(f/f_c)²] − πf t*/ln10` |
| Minimiser | Powell |
| Typical fit band | 0.6–35 Hz |

Equation 1 maps exactly onto the existing `Models.simple_model` with
`MODEL="BRUNE"` (γ=1, n=2) and `MOTION="velocity"` — the current defaults.

**Complementary to PNR in exactly the way a test suite wants:**

| | PNR 2019 | Magna 2020 |
|---|---|---|
| Setting | Induced, UK | Tectonic, Utah |
| Magnitude | ~M 1–2 | **Mw 5.7** down to ML 0.7 |
| Fit band | High-frequency | 0.6–35 Hz, `f_c` ~0.3 Hz at the top end |
| Stresses | Short windows | **20 s windows, low-frequency `f_c`** |

Magna lands squarely on what PNR cannot exercise: the COI / window-length
resolution floor (§4.4.2) bites when `f_c` approaches `1/T`, and a 20 s window
with `f_c` ~0.3 Hz is exactly that regime. The catalogue also spans ML 0.7 → 5.6
on one station set, which is the natural multi-event regression basis.

**Performance baseline.** The paper reports **96,169 waveforms fit in ~4 hours on
a single CPU core** (~6.7 waveforms/s) producing 11,226 Ω measurements. That is a
concrete benchmark target — the refactor should not regress it, and `scipy.fft`
with `workers=-1` (§4.1) should beat it.

**Caveat for the docs:** at Mw 5.7 the point-source Brune assumption is more
strained than for induced events, and directivity may be visible. The paper
handles this by fitting an *apparent* `f_c`. Property of the science, not the
refactor, but better stated than discovered.

#### 5.2.5 Published defaults do not match the shipped defaults

Transcribing the workflow surfaced four places where `config.py` and the
`PreProcess` defaults disagree with what the paper describes. The defaults were
tuned for studies after the paper, so this is drift rather than a defect — but it
is a trap for step 2 of §5.2.6, because running the current code with stock
settings will **not** reproduce the paper.

The fix is §4.7: keep the current values as defaults, and pin the published run
in `studies/magna_2020_paper.toml`. Each later study gets its own file alongside
it, so "which settings produced this" stops being a question anyone has to
reconstruct.

| Paper | Current default | Where |
|---|---|---|
| Sg group velocity **3.4** km/s | `s=2.9` | `PreProcess.basic_set_theoreticals` |
| Noise ends **0.5 s** before Pg | `bshift=0.2` | `PreProcess.get_noise_s` |
| SNR gate in 2–4/4–6/6–8 Hz **enforced** | `ASSERT_BANDWIDTHS = False` | `config.SPECTRAL` |
| No noise rotation described | `ROTATE_NOISE = True, ROT_METHOD = 2` | `config.SPECTRAL` |

The first two are straightforward parameter differences. The last two matter more:

- `ASSERT_BANDWIDTHS = False` means the three-band SNR gate the paper describes
  as its selection criterion **is switched off by default** in the shipped code.
- Noise rotation is not mentioned anywhere in the manuscript, yet it is on by
  default and materially changes the SNR bandwidth. Either the published run had
  it off, or it postdates the paper, or it was used and undocumented. This must
  be resolved before any comparison means anything.

Two further problems with reproducing the published run from this repository:

1. **The paper cites "SpecMod (v0.1.1)". No such tag exists** — the repository has
   no tags at all and no version string anywhere in the source, so the code that
   produced the published numbers cannot be identified by name. Dating it against
   the history turns up a genuine puzzle rather than an easy answer.

   Only ten commits ever touched `specmod/`, and the workflow's step 5 — window
   refinement by the 1st/99th percentiles of the cumulative squared-amplitude
   integral — is implemented by `signal_intensity`, which **did not exist until
   `ba3f7ec` on 2021-01-22**:

   | Commit | Date | `signal_intensity` |
   |---|---|---|
   | `e98621f` | 2020-10-08 | absent |
   | `ba3f7ec` "Cutting function updates" | 2021-01-22 | **introduced** |
   | `09f57b9` "removed cython" | 2021-05-11 | present |
   | `453c77c` (master HEAD) | 2021-08-12 | present |

   But the manuscript records the repository as "last accessed November, 2020"
   and its other weblinks as 2020-09-10 — both *before* that commit. So the
   published run either used local code that was only pushed in January 2021, or
   performed the refinement in analysis code that was later upstreamed. The
   manuscript filename (`...SRL3`) suggests a third revision, which would also
   explain a 2021 code state behind a 2020 access date.

   Practical consequence: **the candidate is `ba3f7ec` or later, not the
   pre-November-2020 commits**, because the earlier code cannot perform the
   workflow the paper describes. Only `09f57b9` (removed cython) changes
   `specmod/` after that, so the realistic candidates are `ba3f7ec` and
   `453c77c`. Pick one, record the reasoning, and move on — this is the
   strongest possible argument for the `v0.1.0` tag (§6.6) and for `hatch-vcs`
   (§6.4): it must never be this hard again.
2. **The full published pipeline is not in this repository.** The two-stage
   inversion (fit Ω/`f_c`/`t*` free, then fix event `f_c` to the
   inverse-hypocentral-distance-weighted mean of station `f_c` and refit) and the
   entire non-parametric G(R)/site inversion live in analysis code that was never
   part of SpecMod. `set_const` exists, but nothing orchestrates it.

Consequence for scoping — and it is an important one: **the achievable target is
reproducing the per-trace spectral fits (Ω, `f_c`, `t*`), not the published Mw
values.** That is the right target anyway, because Ω/`f_c`/`t*` is precisely what
SpecMod produces, and **Table S2 contains 11,226 of them**. Mw additionally
depends on the NP inversion, `V_S`/ρ from the Herrmann *et al.* (2011) Western US
model at source depth, `r₀ = 1000 m`, `F = 2` and `ΘλΦ = 0.55` — none of which
SpecMod computes.

The two-stage fit is, separately, a good candidate for the new API: it is the
workflow the science actually uses, and it currently has to be rebuilt by hand
by every user.

#### 5.2.6 The three-way comparison

The published values were produced by code containing the bugs in §2.2 and §2.5,
under settings that differ from the shipped defaults (§5.2.5). So agreement and
disagreement are both ambiguous unless the comparison is staged:

1. **Paper** — Table S2's 11,226 Ω measurements, plus Figure 2 (US.DUG, mainshock).
2. **0.1.1 re-run** in the legacy Docker image, on the same data with the
   paper's parameters → must reproduce (1). If it does not, the discrepancy is in
   parameters, version or environment, and must be resolved *before* anything
   downstream is interpretable.
3. **New code** → compared against (2), with each deliberate numerical change
   (padding normalisation, `cut_p` ordering, COI floor) accounted for
   individually and recorded in the changelog with its magnitude.

**Noise rotation is a free variable in step 2, and a cheap one.** It is not
recalled whether the published run used it, so the working assumption is the
shipped default: `ROTATE_NOISE = true`, `ROT_METHOD = 2`. The assumption carries
almost no risk, because the re-run *is* the experiment — the setting has only
three states (off, method 1, method 2), and step 2 either reproduces Figure 2 or
it does not. If the first attempt misses, try the other two; whichever matches is
the answer, and it gets written into
`studies/magna_2020_paper.toml` as a determined value rather than a guess.

Worth doing in that order deliberately: run the assumed configuration first and
only search if it fails. A search that starts before there is a discrepancy to
explain is how you end up tuning settings to fit an outcome.

Step 2 is the one that gets skipped, and it is the one that makes step 3 mean
anything — without it, "the new code disagrees with my paper" has at least four
possible causes (§5.2.5) and no way to distinguish them. **This settles the open
question about the legacy Docker image: build it, and Magna is why.**

The image is now precisely specifiable, since the paper's Data and Resources
section pins every version: **ObsPy 1.2.0, SciPy 1.4.1, NumPy 1.18, pandas 1.0.0,
matplotlib 3.2.1, SpecMod v0.1.1**, plus `gfortran` for `mtspec`. That is a far
better basis than the guessed "python 3.9, numpy<2" this plan previously
proposed.

**Regression targets, narrowest first** — start at the top, because a single
trace that disagrees is diagnosable and 11,226 that disagree are not:

| Target | Source | Scope |
|---|---|---|
| One spectrum + fit, US.DUG, mainshock | **Figure 2** | Single trace — the unit test |
| Ω, `f_c`, `t*` for a broadband subset | **Table S2** rows | ~5–10 stations × ~5 events |
| Nine MT-constrained events, Mw 3.40–5.54 | **Table S1** | Absolute anchor, if cheap |

Figure 2 is the highest-value item in the whole validation story: one published,
visually inspectable spectrum with its fitted model, for a named station and a
known event. It should be the first Tier 3 test written.

**Do not attempt to reproduce all 96,169 waveforms or all 11,226 Ω
measurements.** A handful is enough, and better:

- A regression suite is for *localising* a change, and bulk statistics do the
  opposite — "mean Ω shifted by 0.02 log units" tells you something moved but not
  what or where. Ten traces with per-trace tolerances name the failure.
- Bulk agreement can hide compensating errors. Ten diverse traces that each agree
  individually is stronger evidence than an aggregate that happens to land.
- Runtime and dataset size stay small enough that the fixtures live in git
  (§5.1) and the suite runs in CI on every push.

**Restrict the comparison set to broadband stations.** Of the 88 stations, 42 UU
broadband plus 15 from IE/IW/N4/NN/RE are broadband; the other 31 UU channels are
strong motion. Broadband is the right choice here because those are the
instruments the coda method also used, so the two published magnitude estimates
are comparable on the same traces, and because response correction is better
behaved across the 0.6–35 Hz fit band. Strong-motion traces are worth one or two
cases specifically to exercise the acceleration path in §4.2 — but as a distinct
test, not as bulk.

**Distance coverage is the selection criterion.** Station identity barely
matters; spanning the hypocentral distance range does, because that is the axis
along which the quantities under test actually vary — `t*` grows with path
length, the SNR bandwidth narrows with distance as high frequencies attenuate,
and the low-frequency end that constrains Ω is where the COI floor (§4.4.2)
starts to bite. A subset clustered at one distance would agree perfectly and
prove very little.

So: pick broadband stations spread roughly evenly in **log distance across the
paper's 4.5–400 km hypocentral range** — the paper's own NP selection rule asked
each event to span at least a fifth of that range, which is the same instinct.
Concretely, something like:

- **US.DUG** — fixed, it is the published Figure 2 station and the single-trace test.
- One near-source station (UU.NOQ, or UU.ASU4/ASU5 for post-mainshock events),
  covering the few-km end.
- Four or five UU/IW/US broadband stations at roughly 10, 30, 80, 200 and 400 km.
- Mainshock plus three or four aftershocks spanning ML ~1 to ~4, so the
  magnitude and distance axes are both covered without multiplying them out
  fully — small events will simply drop out at the far stations, which is itself
  the SNR gate behaving correctly and worth asserting.

Roughly 20–40 spectra. Enough to cover the range, small enough that every
disagreement gets looked at individually.

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

### 5.3 Characterising `preprocess` — and what it turned up

The golden spectral reference (§5, tier 3) starts from **cut windows**. That
leaves a blind spot exactly the size of `preprocess`: a change to how windows
are chosen *moves* the reference rather than failing against it, so
regenerating after such a change looks identical to regenerating after a
legitimate one. `preprocess.py` could not be touched safely until that gap was
closed.

`tests/golden/window_reference.json` closes it. For each of the 28 tutorial
traces it records station geometry, both picks, the window before refinement,
the window after it, the refinement offsets, and the derived noise window —
all as seconds relative to the origin, so a diff reads in seconds rather than
in ISO timestamps. `tools/make_golden.py` writes it alongside the spectral
reference. Tolerance is one sample: window edges are quantised to `delta`, so
the only route for a last-bit difference is an `argmin` flip inside
`signal_intensity`, which needs two adjacent samples equidistant from a
percentile.

Verified discriminating, not merely present: shifting the refinement
percentile from 1 to 2 fails the refined-window, refinement-offset and noise
tests while leaving the *unrefined* window green; shifting the noise window by
50 ms fails only the noise test. Both report the moved edges by name.

**Six defects the characterisation exposed, and their fixes.** Each was pinned
first as current behaviour, then fixed in a follow-up commit, so the diff of
the fix *is* the record of what observable behaviour changed:

| Where | What was wrong | Now |
|---|---|---|
| `set_stream_distance` | `STREAM_DISTANCE_METHODS` listed `"list"`; the explicit-coordinate branch tested for `"none"`. The path was **unreachable by any argument** — `"none"` failed the guard and did nothing, `"list"` passed it, set the origin fields, then printed `invalid method choice` and computed no distance. | The three coordinate sources are one branch over `stlat/stlon/stelv`; `"list"` works, `"none"` is a deprecated alias. Unknown `dtype`, a missing inventory and missing coordinates each raise naming the argument. The mseed path keeps its original `gps2dist_azimuth` argument order — the call is not symmetric in floating point, so swapping it would move published numbers. |
| `cut_c` | `tafp * relps + s_start` is `float + UTCDateTime`, and `UTCDateTime` has no `__radd__`: `TypeError` on **every** input. The function had never run. | Operands reversed. It runs, and is tested. |
| `cut_p` / `cut_s` | Un-chained `if`s on a free-text `time_after`, with the two functions spelling the relative mode differently (`"relative_time"` vs `"relative_ps"`). A typo — or the sibling's spelling — left the window end unbound, surfacing as `UnboundLocalError` from mid-function. | Validated up front against `TIME_AFTER_METHODS`; both functions accept both spellings; an unrecognised value raises naming itself. |
| `get_noise_p` | Copied the whole input stream, then paired with `zip(..., strict=False)`, so traces the signal stream had lost came back **whole and unlinked** — full-length records presented as noise windows, no error, no warning. | `strict=True`. Positional pairing against a different-length stream is a wrong answer, not a short one. |
| `set_picks_from_pyrocko` | Emergency S pick `p + (p − otime)·ratio` was unchecked, so it only lands after P when the origin precedes the pick. The tutorial config has an origin 18 minutes *after* its picks; a station missing an S pick there would have got S before P and a nonsense window. | Warns and leaves `s_time` unset, which is the pipeline's own idiom for "unusable" — callers already filter on it. |
| `link_window_to_trace` | Recorded the *requested* window and never reconciled it with what `trim` delivered, so `wend - wstart` overstated every truncated noise trace. | Records both: `wstart`/`wend` are what the trace holds, `wstart_requested`/`wend_requested` what was asked for. All call sites link *after* the trim. |

The last row is worth separating because it is not only a reporting bug.
**Every one of the 28 noise windows is truncated** — 1.1–1.7 s against 1.8–3.7 s
signals — because the records begin roughly two seconds before the P arrival.
`get_noise_p` asks for the signal's length and never gets it. This is the
reason noise and signal spectra do not share a frequency resolution, and hence
why `SpectrumPair` carries a `resolution_floor`; see
[`docs/processing.md` §2](processing.md#2-window-selection).

**None of this moved the science.** `pipeline_reference.json` is byte-identical
across the fix commit; only `window_reference.json` changed, and only in the
two ways predicted — recorded edges snapped to sample boundaries, and noise
starts moved from the requested time to the record start. The one place the
change could propagate is `get_noise_p`, which derives its window length from
the signal's recorded `wend - wstart`: now the delivered length rather than the
requested one, differing by at most `delta/2`. On this data it cannot propagate
at all, because every noise window is truncated by the record start regardless.

`utils.py` is the remaining uncharacterised module upstream of the reference.

## 6. Packaging, CI/CD and process

### 6.1 Packaging

- `pyproject.toml` (PEP 621), `hatchling` backend, `src/` layout.
- Python 3.11–3.13. Floors not pins: `numpy>=1.26`, `scipy>=1.11`,
  `obspy>=1.4`, `lmfit>=1.2`, `pandas>=2.0`, `matplotlib>=3.7`.
- Extras: `[multitaper]`, `[wavelet]`, `[mcmc]` (emcee), `[asdf]` (ASDF export,
  §4.6.3), `[mtspec]` (legacy, temporary), `[dev]`, `[docs]`.
- `h5py` and `pyarrow` are core dependencies — persistence is not optional. `pooch` is a core dependency — `specmod.datasets`
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

**Every pull request must target `sgjholt/SpecMod`, never `uofuseismo/SpecMod`.**
This is not a matter of care — it is a GitHub default working against you.
`uofuseismo/SpecMod` is the upstream repository (created 2020-02-27, the same day
as this history's initial commit; not itself a fork; 3 forks, 11 open issues), and
`sgjholt/SpecMod` does not appear in GitHub code search under `user:sgjholt`,
which is what happens to forks. When a PR is opened from a branch in a fork,
**GitHub pre-selects the parent repository as the base**. Accepting that default
proposes the entire refactor to the upstream organisation.

Two consequences worth acting on:

1. **Detach the fork** — decided. Settings → General → Danger Zone → **Leave
   fork network**, then confirm and type the repository name. This is
   self-service now; no support ticket. It is **permanent and cannot be undone**.

   Detaching drops all GitHub-layer metadata: issues, pull requests, wikis,
   stars, watchers, comments and child forks. Git commit history is fully
   preserved. Audited before deciding, the cost here is negligible — **0 issues**
   and **2 closed PRs** (#1 `removed cython`, 2021; #2 the plan merge). Nothing
   in flight.

   Worth it because this repository is becoming the canonical line, with its own
   releases, PyPI package and DOI. It also restores the repo to GitHub code
   search, which forks are excluded from — relevant for a package people are
   meant to find.
2. **GitHub Actions is disabled by default on forked repositories.** Confirm it
   is enabled in Settings → Actions after detaching, before expecting any of §6.5
   to run. Worth knowing before debugging a workflow that never triggers.

**Post-detach checklist**, all browser-only:

- [ ] Settings → Danger Zone → Leave fork network
- [ ] Settings → Branches → default branch `master` → **`main`**
- [ ] Settings → Actions → confirm workflows are enabled
- [ ] Settings → Branches → protect `master` (no pushes, no deletion) so the
      frozen record stays frozen
- [ ] `git tag -a v0.1.0 453c77c -m "pre-refactor snapshot" && git push origin v0.1.0`

**No Claude session URLs in commit messages, PR bodies, or any other published
artifact.** The repository is public and those links are private session state. A
`commit-msg` hook rejecting `Claude-Session:` trailers is installed locally; fold
the same check into the `commitlint` pre-commit configuration (§6.4) so it is
versioned and applies to every clone rather than one working copy.

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
changelog and version bumps automatic. The same `commit-msg` stage should reject
Claude session URLs (§6.6) — public repository, private links.

**PyPI name.** `specmod` is unregistered (checked: 404 on `specmod`, `spec-mod`
and `pyspecmod`). Worth claiming with the `v0.2.0` release at the end of Phase 2
rather than waiting for 1.0 — name squatting on PyPI is real and the name is a
fairly obvious one.

---

## 7. Phasing

Each phase ends green on CI and is independently mergeable.

| Phase | Work | Depends on | Rough size |
|---|---|---|---|
| **0. Safety net** | Freeze `master`, default branch → `main`, optional `v0.1.0` tag (§6.6); reproducible legacy env (`Dockerfile`: gfortran + ObsPy 1.2.0 / SciPy 1.4.1 / NumPy 1.18 / pandas 1.0.0 (§5.2.6)); write `datasets/magna_2020.toml` and a first cut of `specmod.acquire`, publish the artifact as a `data-v1` release asset (§5.2); capture golden outputs for PNR **and** Magna; reproduce Table S2 / Figure 2 with 0.1.1 (§5.2.6 step 2); convert any `.spec` files (§4.6) | — | 1.5–2 days |
| **1. Make it installable** | `pyproject.toml` + hatch-vcs, `src/` layout, `__init__.py`; ruff config, one-shot `ruff format` + `.git-blame-ignore-revs`, module renames to snake_case; mypy skeleton; pre-commit; `test`/`build` CI; `.gitignore`, `CITATION.cff`; fix the three hard breakages (§1) and the four `F821` bugs ruff finds (§2.5); delete `Tests/Tutorial/`, strip notebook outputs, subset the inventory (§5.1) | 0 | 3–4 days |
| **2. De-globalise** | `config/` package per §4.7 — semantic groups, layer resolution, `config show`/`freeze`, provenance stamping; remove all module-level config reads (tracked by `PLW0603`); `Motion`/`AmplitudeKind` enums; `Spectrum` as a frozen dataclass with `duration`; mutable class attrs (`RUF012`); `isinstance` checks; `logging`. **Tag `v0.2.0`** | 1 | 3–4 days |
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
- **Validation anchor** — Magna 2020, with the published workflow transcribed
  into `datasets/magna_2020.toml` and Figure 2 / Tables S1–S2 as regression
  targets (§5.2.4–§5.2.6).
- **Data acquisition** — a general config-driven grabber (`specmod.acquire`),
  not a per-event script; artifacts pinned by SHA256 and served from GitHub
  Release assets via pooch (§5.2).
- **Persistence** — pickle dropped. HDF5 for arrays with a versioned schema of
  our own, Parquet for tables, JSON sidecar for provenance, ASDF as an optional
  export (§4.6).
- **Configuration** — semantic groups, layered overrides with local files
  gitignored by default, resolved config and version stamped into every output
  (§4.7). Current behaviour stays the default.
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
4. **Which commit is "v0.1.1"?** The paper cites SpecMod v0.1.1; no such tag
   exists and the source carries no version string (§5.2.5). Reproducing the
   published run needs that commit identified — by submission date against the
   history, if nothing better is available. Only you can make that call.
5. ~~**Was noise rotation on for the published run?**~~ **Assumed on.** Not
   recalled, so `studies/magna_2020_paper.toml` starts from the shipped values —
   `ROTATE_NOISE = true`, `ROT_METHOD = 2`, `ROT_PARS = {inc = 0.05, space =
   [1e-3, 1.001]}` — and step 2 of §5.2.6 tests the assumption rather than
   depending on it.
6. **Are Tables S1/S2 to hand?** The comparison needs only the Table S2 rows for
   the chosen broadband subset (§5.2.6), not all 11,226. If the supplement is not
   readily available, Figure 2 alone still supports the single-trace test.

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
