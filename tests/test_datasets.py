"""Loading datasets: the local one, and the fetch-and-cache path for published ones.

No test reaches the network. The published path is exercised through an
injected downloader that copies from a local archive, which still runs pooch's
caching, hash check and unpacking — the parts worth testing.
"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import Any

import pytest

from specmod.datasets import (
    PNR_2019,
    Dataset,
    DatasetSpec,
    Event,
    EventDirectory,
    data_dir,
    load,
    load_pnr_2019,
)


class TestTheLocalDataset:
    def test_it_loads_without_a_network(self) -> None:
        """PNR ships with the repository, so this must not reach for a URL."""
        dataset = load_pnr_2019()
        assert dataset.event == PNR_2019
        assert dataset.paths.is_present()
        assert dataset.manifest is None

    def test_it_reads_its_waveforms_and_inventory(self) -> None:
        pytest.importorskip("obspy")
        dataset = load_pnr_2019()
        assert len(dataset.stream("*HHZ*")) > 0
        assert len(dataset.inventory()) > 0

    def test_the_waveforms_are_raw(self) -> None:
        """Counts, not a deconvolved trace.

        The response is stored beside them so removing it stays part of what
        the tests cover, rather than being frozen into the fixture.
        """
        pytest.importorskip("obspy")
        trace = load_pnr_2019().stream("*HHZ*")[0]
        assert "processing" not in trace.stats or not trace.stats.processing


class TestTheCacheLocation:
    def test_the_environment_overrides_the_platform_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cluster ``$HOME`` is often small, or not writable from a node."""
        monkeypatch.setenv("SPECMOD_DATA_DIR", str(tmp_path / "elsewhere"))
        assert data_dir() == tmp_path / "elsewhere"

    def test_without_it_pooch_decides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pooch = pytest.importorskip("pooch")
        monkeypatch.delenv("SPECMOD_DATA_DIR", raising=False)
        assert data_dir() == Path(pooch.os_cache("specmod"))


@pytest.fixture
def published(tmp_path: Path) -> tuple[DatasetSpec, Any]:
    """A tar.gz holding one event directory, and a downloader that serves it.

    Stands in for a release asset. pooch cannot fetch ``file://``, so the
    downloader copies — everything after the transfer is pooch's own code.
    """
    pooch = pytest.importorskip("pooch")

    source = tmp_path / "build" / "2020-03-18T13:09:31.000000Z"
    paths = EventDirectory(source)
    paths.waveforms.mkdir(parents=True)
    paths.stations.mkdir(parents=True)
    (paths.waveforms / "UU.NOQ..HHZ").write_bytes(b"not really miniseed")
    paths.inventory.write_text("<FDSNStationXML/>")

    archive = tmp_path / "magna_2020_v1.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname=source.name)

    def downloader(url: str, output_file: Any, pooch_instance: Any) -> None:
        shutil.copyfile(archive, output_file)

    spec = DatasetSpec(
        name="magna_2020_v1",
        url="https://example.invalid/magna_2020_v1.tar.gz",
        sha256=pooch.file_hash(str(archive)),
        event=Event(
            origin="2020-03-18T13:09:31.000000Z",
            latitude=40.751,
            longitude=-112.078,
            depth_km=9.2,
            catalogue_magnitude=5.7,
            catalogue_magnitude_type="Mww",
        ),
    )
    return spec, downloader


class TestAPublishedDataset:
    def test_it_downloads_unpacks_and_resolves_the_layout(
        self,
        published: tuple[DatasetSpec, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spec, downloader = published
        monkeypatch.setenv("SPECMOD_DATA_DIR", str(tmp_path / "cache"))
        monkeypatch.setitem(_registry(), spec.name, spec)

        dataset = load(spec.name, downloader=downloader)
        assert isinstance(dataset, Dataset)
        assert dataset.event.catalogue_magnitude == 5.7
        assert dataset.paths.is_present()
        assert dataset.paths.inventory.read_text() == "<FDSNStationXML/>"

    def test_a_substituted_archive_is_refused(
        self,
        published: tuple[DatasetSpec, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The hash is the whole point: without it a swapped archive would
        quietly become the new expected answer."""
        spec, downloader = published
        monkeypatch.setenv("SPECMOD_DATA_DIR", str(tmp_path / "cache"))
        wrong = DatasetSpec(
            name=spec.name, url=spec.url, sha256="sha256:" + "0" * 64, event=spec.event
        )
        monkeypatch.setitem(_registry(), spec.name, wrong)

        with pytest.raises(ValueError, match="hash"):
            load(spec.name, downloader=downloader)

    def test_the_second_call_does_not_download_again(
        self,
        published: tuple[DatasetSpec, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Offline after the first fetch is the reason for the cache."""
        spec, downloader = published
        monkeypatch.setenv("SPECMOD_DATA_DIR", str(tmp_path / "cache"))
        monkeypatch.setitem(_registry(), spec.name, spec)

        calls = []

        def counting(url: str, output_file: Any, pooch_instance: Any) -> None:
            calls.append(url)
            downloader(url, output_file, pooch_instance)

        load(spec.name, downloader=counting)
        load(spec.name, downloader=counting)
        assert len(calls) == 1


def _registry() -> dict[str, DatasetSpec]:
    from specmod import datasets  # noqa: PLC0415

    return datasets.REGISTRY


class TestAnUnknownName:
    def test_it_names_what_is_available(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            load("not_a_dataset")

    def test_it_points_at_the_local_loader(self) -> None:
        """The likeliest mistake is reaching for `load("pnr_2019")`."""
        with pytest.raises(ValueError, match="load_pnr_2019"):
            load("pnr_2019")
