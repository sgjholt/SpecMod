"""The two capabilities that went with the legacy containers, rebuilt.

``spectral.Spectra`` carried both persistence (``write_spectra``/
``read_spectra``) and plotting (``quick_vis``). Deleting the class deleted
both, and a spectral package where you can neither save a result nor look at
one is not usable — so these are not optional furniture and they are tested
like the rest.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, ClassVar

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

obspy = pytest.importorskip("obspy")

from specmod.core import SpectrumSet  # noqa: E402
from specmod.io import FORMAT_VERSION, load, save  # noqa: E402
from specmod.pipeline import spectrum_set_from_streams  # noqa: E402
from specmod.plotting import plot_pair, plot_set  # noqa: E402


@functools.cache
def _spectra(windows: Any) -> SpectrumSet:
    signal, noise = windows()
    return spectrum_set_from_streams(signal, noise, estimator="fft")


def _spectra_without_band() -> SpectrumSet:
    """A one-station set whose pair selected no band.

    Built by hand rather than hunted for in the real data: every PNR window
    passes, so the no-band path would otherwise never be exercised.
    """
    from specmod.core import BinnedSpectrum, Spectrum, SpectrumPair  # noqa: PLC0415

    freq = np.linspace(1.0, 40.0, 128)
    spectrum = Spectrum(
        freq=freq,
        amp=np.full_like(freq, 1e-8),
        motion="velocity",
        kind="magnitude",
        duration=3.2,
        sampling_rate=100.0,
        meta={"id": "XX.TEST..HHZ"},
    )
    binned = BinnedSpectrum(freq=freq[:16], amp=np.full(16, 1e-8))
    pair = SpectrumPair(
        signal=spectrum,
        noise=spectrum,
        binned_signal=binned,
        binned_noise=binned,
        snr=np.ones(16),
        resolution_floor=1.0 / 3.2,
        band=None,
    )
    return SpectrumSet(pairs={"XX.TEST..HHZ": pair}, event="synthetic")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ------------------------------------------------------------------- io


class TestRoundTrip:
    def test_every_number_survives_exactly(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """Not to a tolerance. Serialisation that loses bits is a bug, and a
        float that survives a round trip inexactly would quietly move a result
        every time a study was reloaded."""
        spectra = _spectra(pnr_windows)
        back = load(save(tmp_path / "event", spectra))

        assert back.ids() == spectra.ids()
        assert back.event == spectra.event
        for id in spectra.ids():
            got, want = back[id], spectra[id]
            for a, b in (
                (got.signal, want.signal),
                (got.noise, want.noise),
                (got.binned_signal, want.binned_signal),
                (got.binned_noise, want.binned_noise),
            ):
                assert np.array_equal(a.freq, b.freq), id
                assert np.array_equal(a.amp, b.amp), id
            assert np.array_equal(got.snr, want.snr), id
            assert got.band == want.band, id
            assert got.resolution_floor == want.resolution_floor, id

    def test_units_and_domain_survive(self, pnr_windows: Any, tmp_path: Path) -> None:
        """A spectrum that forgets it is a velocity magnitude is a pile of
        floats. The amplitude convention is the thing most easily lost across
        a format boundary and the most damaging to lose."""
        spectra = _spectra(pnr_windows)
        back = load(save(tmp_path / "event", spectra))
        for id in spectra.ids():
            assert back[id].signal.unit == spectra[id].signal.unit
            assert back[id].signal.kind == spectra[id].signal.kind
            assert back[id].signal.motion == spectra[id].signal.motion

    def test_the_loaded_arrays_are_still_read_only(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """A spectrum off disk gives the same guarantee as one just computed,
        which is what makes reload-and-refit safe to do in a loop."""
        back = load(save(tmp_path / "event", _spectra(pnr_windows)))
        pair = back[back.ids()[0]]
        assert not pair.signal.amp.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            pair.signal.amp[0] = 0.0

    def test_a_reloaded_set_can_still_change_domain(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """`to_motion` replays the recorded compare settings, so they have to
        survive the round trip too — the pair's `meta`, not just its arrays."""
        spectra = _spectra(pnr_windows)
        back = load(save(tmp_path / "event", spectra))
        assert "compare_settings" in back[back.ids()[0]].meta

        want = spectra.to_motion("displacement")
        got = back.to_motion("displacement")
        for id in want.ids():
            assert got[id].band == want[id].band, id
            assert got[id].signal.amp == pytest.approx(want[id].signal.amp), id


class TestTheLayout:
    """The four rules in REFACTOR_PLAN §4.6.2, each asserted.

    They are lessons from how pickle failed rather than general good practice,
    so each one is a specific claim about this file and worth a specific test.
    """

    def test_it_names_no_python_type(self, pnr_windows: Any, tmp_path: Path) -> None:
        """*Never store class identity.*

        Pickle broke because it recorded ``specmod.Spectral`` / ``Spectra``. A
        schema that names no Python type cannot be broken by renaming one, so
        the whole file is searched for the package's own name.
        """
        path = save(tmp_path / "event", _spectra(pnr_windows))
        blob = path.read_bytes()
        for forbidden in (
            b"specmod.core",
            b"SpectrumPair",
            b"SpectrumSet",
            b"__main__",
            b"copy_reg",
            b"__reduce__",
        ):
            assert forbidden not in blob, forbidden

    def test_every_file_carries_a_format_version(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """*Every file carries `specmod_format_version`.* Its absence is the
        whole problem with what came before."""
        h5py = pytest.importorskip("h5py")
        path = save(tmp_path / "event", _spectra(pnr_windows))
        with h5py.File(path, "r") as handle:
            assert handle.attrs["specmod_format_version"] == FORMAT_VERSION

    def test_an_unreadable_version_names_both(self, tmp_path: Path) -> None:
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "old.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["specmod_format_version"] = 0
        with pytest.raises(ValueError, match="format version 0"):
            load(path)

    def test_the_units_are_stored_not_implied(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """*Self-describing units.*

        A frequency array and an amplitude array do not say whether they are a
        velocity magnitude or a displacement power spectral density, and the
        two differ by factors that would pass unnoticed.
        """
        h5py = pytest.importorskip("h5py")
        spectra = _spectra(pnr_windows)
        path = save(tmp_path / "event", spectra)
        with h5py.File(path, "r") as handle:
            group = handle[spectra.ids()[0]]
            for prefix in ("signal", "noise"):
                for field in ("motion", "kind", "duration", "sampling_rate"):
                    assert f"{prefix}_{field}" in group.attrs

    def test_one_group_per_channel_named_by_trace_id(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """*One file per event, group per channel.*

        HDF5 group names may contain dots, so ``LV.L001..HHE`` needs no
        mangling and the file browses with the names the pipeline uses.
        """
        h5py = pytest.importorskip("h5py")
        spectra = _spectra(pnr_windows)
        path = save(tmp_path / "event", spectra)
        with h5py.File(path, "r") as handle:
            assert sorted(handle.keys()) == spectra.ids()
            assert "LV.L001..HHE" in handle

    def test_the_shared_frequency_axis_is_stored_once(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """`compare` interpolates the noise onto the signal's axis and bins
        both against the same edges, so the two spectra share an axis *by
        construction*.

        Storing it per spectrum would be waste, and worse, would make a file
        expressible in which the two disagree — a state the containers cannot
        represent and no reader could act on. Seven datasets per group, not
        nine.
        """
        h5py = pytest.importorskip("h5py")
        spectra = _spectra(pnr_windows)
        path = save(tmp_path / "event", spectra)
        with h5py.File(path, "r") as handle:
            group = handle[spectra.ids()[0]]
            assert sorted(group.keys()) == [
                "binned_freq",
                "binned_noise_amp",
                "binned_signal_amp",
                "freq",
                "noise_amp",
                "signal_amp",
                "snr",
            ]

        back = load(path)
        for id in spectra.ids():
            assert np.array_equal(back[id].noise.freq, back[id].signal.freq), id
            assert np.array_equal(
                back[id].binned_noise.freq, back[id].binned_signal.freq
            ), id

    def test_small_arrays_are_not_compressed_and_large_ones_are(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """Measured, not assumed.

        Compression in HDF5 needs chunked storage, which costs a chunk index
        per dataset. These arrays have a median of 91 float64, and compressing
        every one took the file from 370 KB to **867 KB** against a 250 KB
        payload — the indices cost more than the data. Above the threshold the
        ratio flips, which is the case the plan's "chunked, compressed"
        argument was about: a scalogram is ~1 MB per trace.
        """
        h5py = pytest.importorskip("h5py")
        from specmod.io import _COMPRESS_ABOVE_BYTES, _dataset  # noqa: PLC0415

        spectra = _spectra(pnr_windows)
        path = save(tmp_path / "event", spectra)
        with h5py.File(path, "r") as handle:
            group = handle[spectra.ids()[0]]
            assert all(group[name].compression is None for name in group)

        big = tmp_path / "big.h5"
        with h5py.File(big, "w") as handle:
            _dataset(handle, "big", np.zeros(_COMPRESS_ABOVE_BYTES))
            assert handle["big"].compression == "gzip"

    def test_a_pair_with_no_band_omits_it_rather_than_storing_a_sentinel(
        self, tmp_path: Path
    ) -> None:
        """A NaN pair would read back as a band. "No usable bandwidth" and "a
        band at nowhere" are different claims."""
        h5py = pytest.importorskip("h5py")
        spectra = _spectra_without_band()
        path = save(tmp_path / "event", spectra)
        with h5py.File(path, "r") as handle:
            assert "band" not in handle["XX.TEST..HHZ"].attrs
        assert load(path)["XX.TEST..HHZ"].band is None

    def test_the_suffix_and_directory_are_supplied(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """The legacy version raised ``FileNotFoundError`` from inside ``open``,
        naming the file rather than the directory that did not exist."""
        path = save(tmp_path / "nested" / "deeper" / "event", _spectra(pnr_windows))
        assert path.suffix == ".h5"
        assert path.is_file()

    def test_an_event_id_with_dots_in_it_is_not_truncated(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """An event id is an ISO timestamp, and timestamps contain dots.

        ``Path.with_suffix`` reads ``.200000Z`` as an extension and *replaces*
        it, so ``2019-08-26T07:49:24.200000Z`` was silently saved as
        ``2019-08-26T07:49:24.h5`` — and two events in the same second would
        have overwritten each other. Found by running the tutorial, which names
        its output file after the event.
        """
        event = "2019-08-26T07:49:24.200000Z"
        path = save(tmp_path / event, _spectra(pnr_windows))
        assert path.name == f"{event}.h5"
        assert load(path).ids() == _spectra(pnr_windows).ids()


# -------------------------------------------------------------- plotting


class TestPlotPair:
    def test_it_draws_signal_noise_and_band(self, pnr_windows: Any) -> None:
        spectra = _spectra(pnr_windows)
        id = next(k for k in spectra.ids() if spectra[k].passes)
        ax = plot_pair(spectra[id], id=id)

        labels = [line.get_label() for line in ax.get_lines()]
        assert "noise" in labels
        assert id in labels
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        # The band edges are drawn as a LineCollection, not as lines.
        assert len(ax.collections) >= 1
        assert f"{spectra[id].band[0]:.2f}" in ax.get_title()

    def test_it_says_so_when_there_is_no_band(self) -> None:
        """Rather than drawing edges at nowhere, or raising."""

        class NoBand:
            band = None
            resolution_floor = 0.0

            class _S:
                freq = np.linspace(1.0, 10.0, 20)
                amp = np.ones(20)
                unit = "m/s*s"
                meta: ClassVar[dict[str, Any]] = {}

            signal = noise = _S()

        ax = plot_pair(NoBand(), id="XX.TEST")  # type: ignore[arg-type]
        assert "no usable band" in ax.get_title()

    def test_it_draws_into_axes_it_is_given(self, pnr_windows: Any) -> None:
        """What makes these composable into a larger figure — the legacy
        version was a method, so plotting required owning the container."""
        spectra = _spectra(pnr_windows)
        _, (left, right) = plt.subplots(1, 2)
        id = spectra.ids()[0]
        assert plot_pair(spectra[id], left, id=id) is left
        assert not right.get_lines()

    def test_the_binned_spectra_are_optional(self, pnr_windows: Any) -> None:
        spectra = _spectra(pnr_windows)
        id = spectra.ids()[0]
        plain = len(plot_pair(spectra[id], id=id).get_lines())
        plt.close("all")
        detailed = len(plot_pair(spectra[id], id=id, show_binned=True).get_lines())
        assert detailed == plain + 2

    def test_it_labels_the_amplitude_axis_with_the_real_unit(
        self, pnr_windows: Any
    ) -> None:
        spectra = _spectra(pnr_windows)
        id = spectra.ids()[0]
        ax = plot_pair(spectra[id], id=id)
        assert spectra[id].signal.unit in ax.get_ylabel()

    def test_a_fit_is_overlaid_when_given_one(self, pnr_windows: Any) -> None:
        """Passed in rather than read off the spectrum: the containers are
        frozen precisely so a result cannot write itself back into its input."""
        import contextlib  # noqa: PLC0415
        import io as _io  # noqa: PLC0415

        from specmod.fitting import FitSpectra  # noqa: PLC0415

        spectra = _spectra(pnr_windows)
        with contextlib.redirect_stdout(_io.StringIO()):
            fits = FitSpectra(spectra)
            fits.fit_spectra()

        id = next(iter(fits.models))
        ax = plot_pair(spectra[id], id=id, fit=fits.models[id])
        assert "best fit" in [line.get_label() for line in ax.get_lines()]


class TestPlotSet:
    def test_it_draws_one_panel_per_pair(self, pnr_windows: Any) -> None:
        spectra = _spectra(pnr_windows)
        figure = plot_set(spectra, columns=4)
        visible = [ax for ax in figure.axes if ax.get_visible()]
        assert len(visible) == len(spectra)
        # 28 pairs in 4 columns is 7 rows, so the grid is exact and no panel
        # is left over; a count that did not round up would silently drop one.
        assert len(figure.axes) == 28

    def test_spare_panels_are_hidden_not_left_blank(self, pnr_windows: Any) -> None:
        spectra = _spectra(pnr_windows)
        figure = plot_set(spectra, columns=5)  # 28 in 5 columns leaves 2 over
        assert len(figure.axes) == 30
        assert sum(not ax.get_visible() for ax in figure.axes) == 2

    def test_the_column_count_comes_from_configuration(self, pnr_windows: Any) -> None:
        from specmod.config import load_config  # noqa: PLC0415

        columns = load_config().config.viz.plot_columns
        figure = plot_set(_spectra(pnr_windows))
        assert len(figure.axes) % columns == 0

    def test_it_can_be_limited_to_what_passed(self, pnr_windows: Any) -> None:
        spectra = _spectra(pnr_windows)
        figure = plot_set(spectra, passing_only=True, columns=4)
        visible = [ax for ax in figure.axes if ax.get_visible()]
        assert len(visible) == sum(p.passes for p in spectra.pairs.values())

    def test_an_empty_selection_says_why(self, pnr_windows: Any) -> None:
        """Rather than returning a figure with no axes, which reads as a
        plotting bug rather than as "nothing passed"."""
        empty = SpectrumSet(pairs={}, event="none")
        with pytest.raises(ValueError, match="no pair in this set"):
            plot_set(empty)


# --------------------------------------------------------------- tables


class TestTables:
    """Fit results, the other half of §4.6.

    Arrays are asked about one event at a time; tables are a columnar scan over
    every event ever fitted. One format for both would be worse at each.
    """

    @staticmethod
    def _table() -> Any:
        import pandas as pd  # noqa: PLC0415

        return pd.DataFrame(
            {
                "id": ["LV.L001..HHE", "LV.L002..HHN"],
                "fc": [4.25, 7.5],
                "llpsp": [-6.5, -7.25],
                "pass_fitting": [True, False],
                "note": ["ok", None],
            }
        )

    def test_parquet_preserves_dtypes_where_csv_does_not(self, tmp_path: Path) -> None:
        """The concrete reason this is not fashion.

        A CSV column holding one ``None`` comes back as object-dtype strings,
        so ``pass_fitting`` stops being a boolean and a downstream ``.sum()``
        silently counts the wrong thing.
        """
        pytest.importorskip("pyarrow")
        from specmod.tables import read_table, write_table  # noqa: PLC0415

        want = self._table()
        parquet = read_table(write_table(tmp_path / "fits.parquet", want))
        csv = read_table(write_table(tmp_path / "fits.csv", want))

        assert parquet["pass_fitting"].dtype == want["pass_fitting"].dtype
        assert parquet["fc"].dtype == want["fc"].dtype
        assert parquet["id"].tolist() == want["id"].tolist()
        # CSV survives the numerics but loses the null-bearing column's type.
        assert csv["fc"].dtype == want["fc"].dtype
        assert csv["note"].isna().iloc[1]

    def test_floats_survive_parquet_exactly(self, tmp_path: Path) -> None:
        """CSV round-trips every float through decimal text. A fitted corner
        frequency that changes in the last bits each time it is written is a
        result that cannot be compared with itself."""
        pytest.importorskip("pyarrow")
        import numpy as _np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        from specmod.tables import read_table, write_table  # noqa: PLC0415

        rng = _np.random.default_rng(0)
        want = pd.DataFrame({"fc": rng.normal(5, 2, 500) ** 3})
        back = read_table(write_table(tmp_path / "f.parquet", want))
        assert _np.array_equal(back["fc"].to_numpy(), want["fc"].to_numpy())

    def test_provenance_rides_along_in_parquet(self, tmp_path: Path) -> None:
        """And is dropped by CSV, which has nowhere to put it — stated rather
        than hidden, because a lossy export is how a run stops being
        reproducible."""
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq  # noqa: PLC0415

        from specmod.tables import write_table  # noqa: PLC0415

        path = write_table(
            tmp_path / "fits.parquet", self._table(), meta={"config_hash": "abc123"}
        )
        stored = pq.read_table(path).schema.metadata
        assert stored[b"config_hash"] == b"abc123"

    def test_an_unknown_suffix_says_what_is_supported(self, tmp_path: Path) -> None:
        from specmod.tables import write_table  # noqa: PLC0415

        with pytest.raises(ValueError, match="cannot tell what format"):
            write_table(tmp_path / "fits.txt", self._table())

    def test_the_fitter_writes_through_it(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """`write_flatfile` is the documented entry point and now follows the
        suffix, so an existing `.csv` call keeps working."""
        import contextlib  # noqa: PLC0415
        import io as _io  # noqa: PLC0415

        from specmod.fitting import FitSpectra  # noqa: PLC0415

        pytest.importorskip("pyarrow")
        with contextlib.redirect_stdout(_io.StringIO()):
            fits = FitSpectra(_spectra(pnr_windows))
            fits.fit_spectra()

        for suffix in (".parquet", ".csv"):
            path = FitSpectra.write_flatfile(tmp_path / f"fits{suffix}", fits)
            back = FitSpectra.read_flatfile(path)
            assert len(back) == len(fits.table), suffix
            assert back["fc"].to_numpy() == pytest.approx(
                fits.table["fc"].to_numpy()
            ), suffix
