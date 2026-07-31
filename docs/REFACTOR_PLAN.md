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
- ~70 binary waveform files plus a StationXML committed at `Tutorial/Data/`,
  duplicated again under `Tests/Tutorial/`.
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
  viz/
    plots.py               # all matplotlib
  legacy.py                # deprecation shims for the 0.x public names
```

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

Open question for you: for the CWT path, do you want a **single amplitude
spectrum** per window (time-averaged, directly substitutable into the existing
fitting pipeline), or the **full time-frequency surface** exposed as well? The
first is a drop-in; the second is more useful for non-stationary source studies
but needs new plotting and new storage. The plan assumes both, with the
time-averaged reduction as the default.

### 4.5 SNR, bandwidth and noise rotation

`find_optimal_signal_bandwidth` and `find_optimal_signal_bandwidth_2` become
`BandwidthSelector` strategies selected by argument, not by `BW_METHOD` global.
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
- **Pickle:** retained behind `allow_pickle=True` with a `SecurityWarning`, no
  interactive prompt. Old `.spec` files stay readable.

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
snapshot `freq`, `amp`, `bsnr`, `ubfreqs` and the fit table to `.npz`. Every
subsequent change is diffed against that snapshot, so behaviour changes are
visible and deliberate rather than discovered later.

> This must happen **before** any code changes, and it needs an environment where
> `mtspec` still builds — a one-off Docker image or conda env with `gfortran`,
> `numpy<2`, `python 3.9`. Budget half a day for this; without it the refactor is
> unverifiable.

**Tier 4 — unit tests** for the bugs in §2.5, each written as a failing test
first.

**Test data:** cut the committed tutorial data to one three-component station
(~6 files) for the test fixtures. Move the full tutorial dataset out of git
history entirely — host it as a GitHub Release asset and fetch it with `pooch`,
or use git-lfs. Coverage target: 80% overall, 95% on `transforms/` and `core/`.

---

## 6. Packaging, CI/CD and process

**Packaging**
- `pyproject.toml` (PEP 621), `hatchling` backend, `src/` layout.
- Python 3.11–3.13. Floors not pins: `numpy>=1.26`, `scipy>=1.11`,
  `obspy>=1.4`, `lmfit>=1.2`, `pandas>=2.0`, `matplotlib>=3.7`.
- Extras: `[multitaper]`, `[wavelet]`, `[mcmc]` (emcee), `[mtspec]` (legacy,
  temporary), `[dev]`, `[docs]`.
- `uv` for dev environments and a committed lockfile for CI reproducibility.
- Delete `requirements.txt`.

**Quality gates**
- `ruff` for lint **and** format (replaces black + flake8 + isort).
- `mypy --strict` on `core/` and `transforms/`; permissive elsewhere initially.
- `pre-commit` running ruff, mypy, and `nbstripout` on the tutorial notebook.

**CI (GitHub Actions)**
- `test.yml`: matrix Python 3.11/3.12/3.13 × ubuntu/macos → ruff, mypy, pytest,
  coverage upload.
- `build.yml`: build sdist + wheel, `twine check`, install-from-wheel smoke test.
- `release.yml`: on tag, publish to PyPI via Trusted Publishing (OIDC, no
  long-lived token).
- `docs.yml`: mkdocs-material → GitHub Pages, with the tutorial rendered via
  `mkdocs-jupyter`.
- Branch protection on `master`; require CI green.

**Versioning and release**
- SemVer, `0.x` until the API settles. Tag `v0.2.0` at the end of Phase 2.
- `CHANGELOG.md` in Keep a Changelog format.
- `CITATION.cff` + Zenodo integration for a DOI — this is research software and
  currently has no citable artefact beyond the Edwards et al. (2010) reference.
- Deprecation policy: old public names live in `legacy.py` with
  `DeprecationWarning` for one minor version, then removed.

---

## 7. Phasing

Each phase ends green on CI and is independently mergeable.

| Phase | Work | Depends on | Rough size |
|---|---|---|---|
| **0. Safety net** | Reproducible legacy env (Docker/conda + gfortran); capture golden outputs for the tutorial event; commit snapshots | — | 0.5 day |
| **1. Make it installable** | `pyproject.toml`, `src/` layout, `__init__.py` + `__version__`, ruff, pre-commit, CI skeleton, `.gitignore`, `CHANGELOG`, `CITATION.cff`; fix the three hard breakages (§1) and the §2.5 `NameError`s; data out of git | 0 | 2–3 days |
| **2. De-globalise** | `Settings` dataclasses passed explicitly; remove all module-level config reads; `Motion`/`AmplitudeKind` enums; `Spectrum` as a frozen dataclass with `duration`; `isinstance` checks; `logging` | 1 | 3–4 days |
| **3. Transform layer** | `SpectralEstimator` protocol; `FFTEstimator`, `WelchEstimator`, `MultitaperEstimator`; `smoothing/` incl. Konno–Ohmachi and `LogBinner`; mtspec demoted to optional legacy backend; Tier 1 + Tier 2 tests | 2 | 5–7 days |
| **4. CWT** | `CWTEstimator`, COI handling, the Parseval/units calibration and its test; time-frequency plotting | 3 | 4–6 days |
| **5. Decompose** | Split `Spectral.py` (655 lines) into `core/` + `snr/`; `Fitting.py` → `fitting/`; models as objects; `io/`; `viz/`; non-mutating operations; `legacy.py` shims | 3 | 5–7 days |
| **6. Ship** | Docs site, rewritten tutorial with no `os.chdir`, migration guide, PyPI + Zenodo release | 4, 5 | 2–3 days |

Phases 4 and 5 are independent of each other and can run in parallel.

Rough total: **4–6 weeks** of focused work. Phases 0–3 alone (≈2 weeks) get the
package installable, tested, mtspec-free and CI-covered — that is the point at
which the project stops decaying, and it is a reasonable place to cut if time is
short.

---

## 8. Decisions needed from you

1. **Backwards compatibility.** Keep `legacy.py` shims for the 0.x API, or make a
   clean break at 1.0? A clean break is much less work; the shims matter only if
   there are downstream users or unpublished analysis scripts depending on the
   current names.
2. **CWT output** (§4.4): time-averaged amplitude spectrum only, or also the full
   time-frequency surface?
3. **Default estimator.** Multitaper (matching current behaviour) or FFT +
   Konno–Ohmachi (faster, more conventional in engineering seismology)?
4. **Python floor.** 3.11 is proposed. Any users stuck on 3.9/3.10?
5. **Tutorial data.** OK to rewrite history to drop the ~70 committed waveform
   files, or keep history and just stop adding to it?
6. **Golden snapshots.** Is it worth half a day standing up a legacy `mtspec`
   environment to capture them? Strongly recommended, but it is the one task with
   no direct deliverable.

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
