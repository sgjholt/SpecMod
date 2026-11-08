"""Times in file and directory names, and moving data off the pre-0.7 names.

Before 0.7 an event directory was named by its origin as ObsPy prints it,
``2019-08-26T07:30:47.000000Z``, and its waveform and pick files carried the
same string. The colons are not legal in a Windows file name, and uv 0.13
refuses to build a source distribution that contains one, which stopped the
release. Names now go through :func:`specmod.datasets.path_name`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import pytest
from click.testing import CliRunner

from specmod.acquire import verify
from specmod.cli import main
from specmod.datasets import EVENTS, PNR_2019, Event, migrate_names, path_name

ROOT = Path(__file__).resolve().parent.parent

LEGACY = "2019-08-26T07:30:47.000000Z"
CURRENT = "20190826T073047.000000Z"
WAVEFORMS = ("LV.L001..HHE", "LV.L001..HHN")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_fetch(out: Path) -> Path:
    """A fetch as ``specmod fetch`` wrote it before 0.7, manifest included."""
    event = out / LEGACY
    files: dict[str, str] = {}
    for channel in WAVEFORMS:
        path = event / "waveforms" / f"{channel}_{LEGACY}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(channel.encode())
        files[str(path.relative_to(out))] = _sha256(path)
    inventory = event / "stations" / "inventory.xml"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("<FDSNStationXML/>")
    files[str(inventory.relative_to(out))] = _sha256(inventory)
    picks = event / "picks" / f"{LEGACY}.picks"
    picks.parent.mkdir(parents=True)
    picks.write_text("phase: 2019-08-26 07:30:50.5\n")

    manifest = {"resolved": {"origin": LEGACY}, "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return event


class TestPathName:
    def test_it_writes_iso_8601_basic_format(self) -> None:
        assert path_name(LEGACY) == CURRENT

    def test_it_keeps_microseconds_so_events_a_second_apart_cannot_collide(
        self,
    ) -> None:
        assert path_name("2019-08-26T07:49:24.2") == "20190826T074924.200000Z"
        assert path_name("2019-08-26T07:49:24.2") != path_name("2019-08-26T07:49:24")

    def test_it_is_idempotent(self) -> None:
        assert path_name(path_name(LEGACY)) == path_name(LEGACY)

    def test_it_takes_a_utcdatetime(self) -> None:
        obspy = pytest.importorskip("obspy")
        assert path_name(obspy.UTCDateTime(LEGACY)) == CURRENT

    def test_names_sort_in_time_order(self) -> None:
        times = ["2019-08-26T07:30:47", "2019-08-26T07:30:47.5", "2019-08-27T00:00"]
        assert sorted(map(path_name, times)) == list(map(path_name, times))


class TestTheShippedData:
    def test_no_tracked_name_contains_a_colon(self) -> None:
        """What failed the release: uv 0.13 refuses to build the sdist."""
        listed = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        if listed.returncode != 0:
            pytest.skip("not a git checkout, so there is no list of tracked files")
        tracked = listed.stdout.splitlines()
        assert tracked
        assert [name for name in tracked if ":" in name] == []

    def test_the_pnr_event_is_found_under_its_new_name(self) -> None:
        paths = PNR_2019.directory(ROOT)
        assert paths.root.name == CURRENT
        assert paths.is_present()


class TestTheLegacyFallback:
    def test_an_unmigrated_directory_is_still_found_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / EVENTS / LEGACY / "waveforms").mkdir(parents=True)
        event = Event(origin=LEGACY, latitude=0.0, longitude=0.0, depth_km=1.0)

        with pytest.warns(DeprecationWarning, match="specmod migrate-names"):
            paths = event.directory(tmp_path)
        assert paths.root.name == LEGACY

    def test_the_new_name_wins_and_says_nothing(self, tmp_path: Path) -> None:
        (tmp_path / EVENTS / LEGACY).mkdir(parents=True)
        (tmp_path / EVENTS / CURRENT).mkdir(parents=True)
        event = Event(origin=LEGACY, latitude=0.0, longitude=0.0, depth_km=1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            paths = event.directory(tmp_path)
        assert paths.root.name == CURRENT

    def test_with_neither_present_the_new_name_is_returned(
        self, tmp_path: Path
    ) -> None:
        event = Event(origin=LEGACY, latitude=0.0, longitude=0.0, depth_km=1.0)
        assert event.directory(tmp_path).root.name == CURRENT


class TestMigrateNames:
    def test_every_time_in_every_name_is_rewritten(self, tmp_path: Path) -> None:
        _legacy_fetch(tmp_path)

        migrate_names(tmp_path)

        names = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
        assert not [n for n in names if ":" in n]
        assert f"{CURRENT}/waveforms/LV.L001..HHE_{CURRENT}" in names
        assert f"{CURRENT}/picks/{CURRENT}.picks" in names
        # Names with no time in them are left as they were.
        assert f"{CURRENT}/stations/inventory.xml" in names

    def test_contents_are_untouched(self, tmp_path: Path) -> None:
        event = _legacy_fetch(tmp_path)
        before = _sha256(event / "picks" / f"{LEGACY}.picks")

        migrate_names(tmp_path)

        assert _sha256(tmp_path / CURRENT / "picks" / f"{CURRENT}.picks") == before

    def test_a_migrated_fetch_still_verifies(self, tmp_path: Path) -> None:
        _legacy_fetch(tmp_path)
        assert verify(tmp_path) == []

        migrate_names(tmp_path)

        assert verify(tmp_path) == []
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert not [name for name in manifest["files"] if ":" in name]
        # Only the paths change: the origin is a time, not a name.
        assert manifest["resolved"]["origin"] == LEGACY

    def test_a_manifest_inside_a_renamed_directory_is_updated(
        self, tmp_path: Path
    ) -> None:
        """A tree of fetches, each in a directory named by its own time."""
        _legacy_fetch(tmp_path / LEGACY)

        migrate_names(tmp_path)

        assert verify(tmp_path / CURRENT) == []

    def test_a_dry_run_changes_nothing(self, tmp_path: Path) -> None:
        _legacy_fetch(tmp_path)
        before = sorted(tmp_path.rglob("*"))
        manifest = (tmp_path / "manifest.json").read_text()

        renames = migrate_names(tmp_path, dry_run=True)

        assert len(renames) == len(WAVEFORMS) + 2  # and the picks, the event
        assert sorted(tmp_path.rglob("*")) == before
        assert (tmp_path / "manifest.json").read_text() == manifest

    def test_a_second_run_finds_nothing_to_do(self, tmp_path: Path) -> None:
        _legacy_fetch(tmp_path)
        migrate_names(tmp_path)
        assert migrate_names(tmp_path) == []

    def test_it_refuses_rather_than_overwrite(self, tmp_path: Path) -> None:
        """Both layouts of one event: which to keep is not a silent choice."""
        _legacy_fetch(tmp_path)
        (tmp_path / CURRENT).mkdir()
        before = sorted(tmp_path.rglob("*"))

        with pytest.raises(FileExistsError, match="Nothing was renamed"):
            migrate_names(tmp_path)
        assert sorted(tmp_path.rglob("*")) == before

    def test_it_refuses_a_file(self, tmp_path: Path) -> None:
        target = tmp_path / "manifest.json"
        target.write_text("{}")
        with pytest.raises(NotADirectoryError):
            migrate_names(target)


class TestTheCommand:
    def test_it_migrates_and_reports(self, tmp_path: Path) -> None:
        _legacy_fetch(tmp_path)

        result = CliRunner().invoke(main, ["migrate-names", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert result.output.endswith(f"{tmp_path}: renamed 4 path(s)\n")
        assert (tmp_path / CURRENT).is_dir()

    def test_dry_run_lists_and_leaves_everything(self, tmp_path: Path) -> None:
        _legacy_fetch(tmp_path)

        result = CliRunner().invoke(main, ["migrate-names", "--dry-run", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert f"{LEGACY} -> {CURRENT}" in result.output
        assert result.output.endswith(f"{tmp_path}: would rename 4 path(s)\n")
        assert (tmp_path / LEGACY).is_dir()
