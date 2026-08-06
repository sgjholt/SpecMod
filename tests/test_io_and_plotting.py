"""The two capabilities that went with the legacy containers, rebuilt.

``spectral.Spectra`` carried both persistence (``write_spectra``/
``read_spectra``) and plotting (``quick_vis``). Deleting the class deleted
both, and a spectral package where you can neither save a result nor look at
one is not usable — so these are not optional furniture and they are tested
like the rest.
"""

from __future__ import annotations

import functools
import json
import zipfile
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


class TestTheFormat:
    def test_it_does_not_pickle(self, pnr_windows: Any, tmp_path: Path) -> None:
        """The whole reason for replacing the old format.

        Pickle stores the import path of every class, so a stored result stops
        loading the moment a class is renamed — which is why the shipped
        ``.spec`` files have been unreadable since before this refactor. Read
        back with ``allow_pickle=False``, which raises if anything in the
        archive needs unpickling.
        """
        path = save(tmp_path / "event", _spectra(pnr_windows))
        with np.load(path, allow_pickle=False) as archive:
            assert len(archive.files) > 1

    def test_the_metadata_is_readable_without_python(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """An ``.npz`` is a zip, so the header is greppable from a shell."""
        path = save(tmp_path / "event", _spectra(pnr_windows))
        with zipfile.ZipFile(path) as archive:
            assert "header.npy" in archive.namelist()
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(str(archive["header"]))
        assert header["format_version"] == FORMAT_VERSION
        assert len(header["pairs"]) == 28

    def test_an_unreadable_version_says_so(self, tmp_path: Path) -> None:
        """Naming both versions, rather than a ``KeyError`` on a renamed field."""
        path = tmp_path / "old.npz"
        np.savez_compressed(
            path, header=np.array(json.dumps({"format_version": 0, "pairs": {}}))
        )
        with pytest.raises(ValueError, match="format version 0"):
            load(path)

    def test_the_suffix_and_directory_are_supplied(
        self, pnr_windows: Any, tmp_path: Path
    ) -> None:
        """The legacy version raised ``FileNotFoundError`` from inside ``open``,
        naming the file rather than the directory that did not exist."""
        path = save(tmp_path / "nested" / "deeper" / "event", _spectra(pnr_windows))
        assert path.suffix == ".npz"
        assert path.is_file()


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
