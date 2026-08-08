"""Configuration sections, grouped semantically by pipeline stage.

Each section is a frozen dataclass owned by the stage it configures, replacing
the flat dicts in the old ``config.py`` and the parameters that were previously
reachable only as function defaults — or, in the case of the multitaper
time-bandwidth product, not reachable at all.

Defaults here reproduce the behaviour shipped before this refactor, not the
values used for the published Magna run. Those live in
``studies/magna_2020_paper.toml``. See ``docs/REFACTOR_PLAN.md`` §4.7 and §5.2.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal, Self

__all__ = [
    "AcquireConfig",
    "Config",
    "FittingConfig",
    "ModelConfig",
    "SmoothingConfig",
    "SnrConfig",
    "TransformConfig",
    "VizConfig",
    "WindowsConfig",
]


@dataclass(frozen=True, slots=True)
class AcquireConfig:
    """Waveform acquisition. Consumed by :mod:`specmod.acquire`."""

    #: FDSN data centre. Different centres serve different holdings for the
    #: same event, so this is part of the provenance record, not a detail.
    client: str = "IRIS"
    event_id: str | None = None
    #: Fallback when ``event_id`` is not given: explicit origin.
    origin_time: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    depth_km: float | None = None
    magnitude: float | None = None

    networks: tuple[str, ...] = ("*",)
    stations: tuple[str, ...] = ("*",)
    locations: tuple[str, ...] = ("*",)
    channels: tuple[str, ...] = ("HH?", "BH?", "EN?")
    max_radius_km: float = 400.0

    #: Seconds either side of origin to request.
    seconds_before: float = 60.0
    seconds_after: float = 300.0

    #: Store raw counts plus the response rather than a deconvolved trace, so
    #: response removal stays under test and no ObsPy version is baked in.
    remove_response: bool = False


@dataclass(frozen=True, slots=True)
class WindowsConfig:
    """Phase arrivals and signal/noise window construction."""

    #: Group velocities (km/s) for theoretical arrivals. The published Magna
    #: run used s=3.4; 2.9 is the shipped default and is kept as such.
    p_velocity: float = 5.9
    s_velocity: float = 2.9
    distance_metric: Literal["repi", "rhyp"] = "repi"

    #: Used when an S pick is missing: s_time = p_time + emergency_ratio * (p - o).
    emergency_ratio: float = 1.7

    #: S window: opens at ``s_start_ratio`` of the P-S time, runs ``s_length``.
    s_start_ratio: float = 0.8
    s_length: float = 20.0
    s_length_mode: Literal["absolute_time", "relative_ps"] = "absolute_time"

    #: P window.
    p_before: float = 0.0
    p_length: float = 0.8
    p_length_mode: Literal["absolute_time", "relative_time"] = "relative_time"

    #: Refine windows to percentiles of the cumulative squared-amplitude
    #: integral. This is step 5 of the published Magna workflow.
    refine: bool = True
    refine_percentiles: tuple[float, float] = (1.0, 99.0)

    #: Noise window ends this many seconds before the P arrival. The published
    #: run used 0.5; 0.2 is the shipped default.
    noise_shift: float = 0.2
    noise_length: float = 1.0

    pad_seconds: float = 0.0
    pad_value: float = 0.0
    station_shifts: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransformConfig:
    """Time-to-frequency conversion. Consumed by :mod:`specmod.transforms`."""

    estimator: Literal[
        "multitaper", "fft", "welch", "cwt", "prieto", "quadratic", "mtspec"
    ] = "multitaper"

    #: Multitaper. ``time_bandwidth`` was previously the literal 3 passed
    #: positionally to mtspec, with no way to configure it.
    time_bandwidth: float = 3.0
    n_tapers: int = 5
    #: On by default: leakage suppression is the point of multitaper, and flat
    #: weighting leaves the high-frequency floor ~287x high under a strong
    #: low-frequency peak. See specmod.transforms.multitaper.
    adaptive: bool = True
    #: Rescale the spectrum to integrate to the record variance, as mtspec and
    #: Prieto's multitaper do. Needed to reproduce pre-refactor results; off by
    #: default because it makes the Parseval check circular.
    normalize_to_variance: bool = False

    #: FFT / Welch.
    taper: Literal["hann", "tukey", "boxcar"] = "tukey"
    taper_alpha: float = 0.05
    #: ``None`` for no padding, an integer, or "fast"/"pow2". Padding is a pure
    #: interpolation here -- the normalisation is keyed off duration, not
    #: len(freq), which is what the old psd_to_amp got wrong. Use "fast" to
    #: avoid the slow FFT path on prime-length cut windows.
    n_fft: int | str | None = None
    welch_segment_length: int | None = None

    #: Continuous wavelet transform.
    wavelet: Literal["morlet"] = "morlet"
    omega0: float = 6.0
    #: Scale resolution: voices per octave.
    dj: float = 0.125
    mask_coi: bool = True

    #: Drop the DC bin. The old code did this unconditionally and *before*
    #: using len(freq) for normalisation, biasing amplitudes slightly.
    drop_dc: bool = True


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    """Spectral smoothing and log-space binning."""

    method: Literal["log_bins", "konno_ohmachi", "none"] = "log_bins"

    #: Log bin edges. ``None`` derives them from the record: fmin from 1/T,
    #: fmax from Nyquist. The old code hardcoded 0.001-200 Hz regardless.
    f_min: float | None = 0.001
    f_max: float | None = 200.0
    n_bins: int = 151

    #: Konno-Ohmachi bandwidth ``b``. Smaller smooths harder.
    konno_ohmachi_bandwidth: float = 40.0


@dataclass(frozen=True, slots=True)
class SnrConfig:
    """Signal-to-noise assessment and usable bandwidth selection."""

    tolerance: float = 3.0
    min_points: int = 10

    #: Require SNR above ``tolerance`` in every band. The published Magna run
    #: used this as its selection criterion; it ships disabled.
    assert_bandwidths: bool = False
    bands: tuple[tuple[float, float], ...] = ((2.0, 4.0), (4.0, 6.0), (6.0, 8.0))

    #: Scale noise amplitude by sqrt(len(signal)/len(noise)) when the noise
    #: window is shorter than the signal window.
    scale_parseval: bool = True
    interpolate_noise: bool = True

    #: Names come from :data:`specmod.core.bandwidth.BANDWIDTH_SELECTORS`.
    #: This said ``"integral"`` while the registry said ``"widest"``, from the
    #: period when the selector was still a percentile of a sign integral —
    #: a config value that named nothing the code would accept.
    bandwidth_method: Literal["widest", "peak"] = "peak"

    #: Impose a low-frequency floor from the window length (~1/T), or the cone
    #: of influence when the spectrum came from a CWT. Nothing enforced this
    #: before, so a short window could report bandwidth it could not resolve.
    resolution_floor: bool = True

    rotate_noise: bool = True
    rotation_method: Literal["rotate", "boost"] = "boost"
    rotation_increment: float = 0.05
    rotation_space: tuple[float, float] = (1e-3, 1.001)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Source and attenuation model."""

    source: Literal["brune", "boatwright"] = "brune"
    #: The motion the *model* is expressed in. Once Spectrum carries its own
    #: motion (§4.2) this is a default rather than a global that must be kept
    #: in sync by hand.
    motion: Literal["displacement", "velocity", "acceleration"] = "velocity"
    frequency_dependent_attenuation: bool = False


@dataclass(frozen=True, slots=True)
class FittingConfig:
    """Minimisation."""

    method: str = "powell"
    fit_bins: bool = False
    weight_method: Literal["none", "log"] = "none"

    #: Initial guesses that were hardcoded in ModelGuess.
    initial_t_star: float = 0.01
    initial_alpha: float = 1e-5

    #: Lower bounds on the fitted parameters that have one physically.
    #:
    #: A negative ``t*`` says the wave gained energy travelling; a corner
    #: frequency at or below zero is not a poor measurement but a meaningless
    #: one. Neither is prevented by the misfit surface, and lmfit will return
    #: either if the surface leans that way — with the shipped multitaper
    #: default it returned ``fc = -4.45 Hz`` on one PNR station.
    #:
    #: Zero rather than a small positive number, deliberately: a parameter that
    #: lands *on* its bound is flagged by ``pass_fitting``, so the fit is
    #: rejected rather than reported as a corner frequency of nothing.
    t_star_min: float = 1e-4
    corner_frequency_min: float = 0.0

    #: The two-stage event fit; see :mod:`specmod.staged`.
    #:
    #: One spectrum cannot separate the source corner from the path
    #: attenuation — they trade off on the falling limb — so the corner is
    #: determined by the ensemble and then held fixed while each station
    #: refits the rest. ``event_parameter`` is what the ensemble decides.
    #: ``"fc"`` because that is the term belonging to the source; ``"ts"`` is
    #: the meaningful alternative for a study with an independent handle on Q.
    event_parameter: str = "fc"
    #: How stations are weighted into the event value. The published choice is
    #: inverse hypocentral distance: the nearer station has less path, so less
    #: of its falloff can be attenuation. See ``specmod.staged.WEIGHT_MODELS``.
    event_weighting: str = "inverse_hypocentral_distance"

    #: Which channels contribute to the event value, as shell globs matched
    #: against the trace id and each of its SEED components — so ``"AQ07"``
    #: means the station, ``"HHE"`` means the component, ``"UR"`` means the
    #: network. Empty ``include`` means "everything not excluded".
    #:
    #: These exist to be edited *after* looking at a first pass. Quality
    #: control is a judgement — a clipped record, a bad response, a pick on
    #: the wrong phase — and a station that is confidently wrong moves the
    #: event value for every other station. Putting the decision in the study
    #: file is what makes it part of the record rather than something done in
    #: a notebook and forgotten.
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    #: Drop a station whose stage-1 fit ended with a parameter pinned against
    #: one of its bounds. The value reported there is the bound rather than a
    #: measurement, so averaging it in is averaging in a constant.
    require_pass: bool = True


@dataclass(frozen=True, slots=True)
class VizConfig:
    """Plotting.

    ``PLOT_COLUMNS`` was previously defined in *both* the SPECTRAL and FITTING
    dicts, and the two copies could disagree. One home makes that impossible.
    """

    plot_columns: int = 3


@dataclass(frozen=True, slots=True)
class Config:
    """The whole resolved configuration."""

    acquire: AcquireConfig = field(default_factory=AcquireConfig)
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    snr: SnrConfig = field(default_factory=SnrConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)
    viz: VizConfig = field(default_factory=VizConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict, suitable for TOML or JSON."""
        result: dict[str, Any] = _as_dict(self)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build a Config from a nested mapping, validating section names."""
        known = {f.name: f.type for f in fields(cls)}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(
                f"Unknown configuration section(s): {sorted(unknown)}. "
                f"Valid sections are {sorted(known)}."
            )
        kwargs: dict[str, Any] = {}
        for name, f in ((f.name, f) for f in fields(cls)):
            if name in data:
                kwargs[name] = _build_section(f.default_factory(), data[name], name)  # type: ignore[misc]
        return cls(**kwargs)


def _build_section(default: Any, values: dict[str, Any], section: str) -> Any:
    """Construct one section, rejecting unknown keys loudly.

    A silently ignored typo in a config file is a reproducibility bug: the run
    looks configured and is not.
    """
    valid = {f.name for f in fields(default)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(
            f"Unknown key(s) in [{section}]: {sorted(unknown)}. "
            f"Valid keys are {sorted(valid)}."
        )
    # Use the live attribute values, not _as_dict: that flattens tuples to
    # lists for serialisation, and feeding them back in would rebuild the
    # section with list-valued fields that compare unequal and are unhashable.
    defaults = {f.name: getattr(default, f.name) for f in fields(default)}
    coerced = {k: _coerce(defaults.get(k), v) for k, v in values.items()}
    return type(default)(**{**defaults, **coerced})


def _coerce(default_value: Any, incoming: Any) -> Any:
    """Reshape a TOML value to match the default's container types.

    TOML has arrays but no tuples, so every sequence arrives as a list. The
    dataclasses use tuples for immutability, and the nesting has to be matched
    all the way down — ``bands`` is a tuple *of tuples*, and coercing only the
    outer level leaves inner lists that compare unequal and break hashing.
    """
    if isinstance(default_value, tuple) and isinstance(incoming, (list, tuple)):
        inner = default_value[0] if default_value else None
        return tuple(_coerce(inner, item) for item in incoming)
    if isinstance(default_value, dict) and isinstance(incoming, dict):
        return dict(incoming)
    return incoming


def _as_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _as_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return [_as_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    return obj
